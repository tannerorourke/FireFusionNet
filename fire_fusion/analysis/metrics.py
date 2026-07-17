from typing import Dict, List, Literal, Optional, Tuple
import math
import torch
import torch.nn as nn
import numpy as np

class TemperatureScaler(nn.Module):
    """
    Temperature scaling: logit_scaled = logit / T
    T > 0; we optimize log_T for stability: T = exp(log_T).

    A pure scale keeps the 0.5-crossing pinned at logit 0, so it cannot move a
    decision boundary that training shifted with a class weight. For that case
    use PlattScaler, whose intercept absorbs the shift.
    """
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # log T, init T=1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature)
        return logits / temperature


class PlattScaler(nn.Module):
    """
    Affine logit calibration: z_cal = a * z + b, then p = sigmoid(z_cal).

    A single-logit head trained with BCEWithLogitsLoss(pos_weight=w) converges,
    at the population optimum, to z = log(w) + logit(p_true): the class weight
    lands as a constant additive offset of log(w) in logit space. The intercept
    b is what cancels that offset; the slope a corrects any residual over- or
    under-confidence. Initializing (a=1, b=-log(w)) starts the fit exactly on
    the analytic prior correction, so an unfittable split still yields sane
    probabilities.

    a is carried as exp(log_a) to keep the map monotone, which leaves every
    ranking score (ROC-AUC, PR-AUC) invariant under calibration.
    """
    def __init__(self, prior_pos_weight: float | None = None):
        super().__init__()
        b0 = -math.log(float(prior_pos_weight)) if prior_pos_weight else 0.0
        self.log_a = nn.Parameter(torch.zeros(1))       # a = 1
        self.b = nn.Parameter(torch.full((1,), b0))     # analytic prior offset

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_a) * logits + self.b

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(logits))

    @torch.enable_grad()
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        """ Fit (a, b) by minimizing unweighted BCE on held-out cells.

        Unweighted is the point: the reliability we want is P(fire | score), so
        the fit must see the true class balance rather than the training-time
        reweighting that caused the miscalibration.
        """
        z = logits.detach().flatten().float()
        y = labels.detach().flatten().float()

        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(self.forward(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def state(self) -> Dict[str, float]:
        return {"a": float(torch.exp(self.log_a).item()), "b": float(self.b.item())}

    def load_state(self, params: Dict[str, float]):
        with torch.no_grad():
            self.log_a.fill_(math.log(float(params["a"])))
            self.b.fill_(float(params["b"]))
        return self


@torch.no_grad()
def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15
) -> float:
    """ Weighted gap between confidence and accuracy across equal-width bins.

    ECE is the single scalar that says whether a probability means what it
    claims: bin by predicted probability, and in each bin compare the mean
    prediction to the observed frequency, weighted by bin population.
    """
    p = probs.detach().flatten().float()
    y = labels.detach().flatten().float()
    if p.numel() == 0:
        return float("nan")

    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=p.device)
    ece = torch.zeros((), device=p.device)
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        # the last bin owns its right edge so p == 1.0 is counted
        in_bin = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = in_bin.sum()
        if n == 0:
            continue
        conf = p[in_bin].mean()
        acc = y[in_bin].mean()
        ece += (n.float() / p.numel()) * (conf - acc).abs()
    return float(ece.item())


class Metric:
    def __init__(self):
        self.record = []
    def reset(self) -> None:
        raise NotImplementedError
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        raise NotImplementedError
    def compute_step(self) -> Dict:
        raise NotImplementedError
    def get_history(self):
        raise NotImplementedError



class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.record = []
        self.ep_correct = 0
        self.ep_total = 0
            
    def reset(self) -> None:
        self.ep_correct = 0
        self.ep_total = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """ Update accuracy and ground truth labels
        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
            mask (torch.BoolTensor): optional (b, h, w) selection of the cells to score
        """
        if mask is not None:
            preds = preds[mask]
            labels = labels[mask]

        self.ep_correct += (preds.type_as(labels) == labels).sum().item()
        self.ep_total += labels.numel()

    def compute_step(self) -> dict[str, float]:
        """ Return scores for the epoch, reset internal state, and update p/epoch record """
        acc = self.ep_correct / (self.ep_total + 1e-6)
        self.record.append(acc)

        scores = {
            f"accuracy": acc,
            f"n_samples": self.ep_total
        }
        self.reset()

        return scores
    
    def get_history(self):
        return self.record 



class ConfusionMatrix(Metric):
    """
    Metric for computing mean IoU, accuracy, precision, recall, F1, and confusion matrix.

    Counts accumulate into the matrix itself, so memory is O(num_classes^2)
    rather than O(cells seen): an epoch of per-pixel predictions is billions of
    entries at the finer grid resolutions. Every score below is a function of
    the matrix alone.
    """

    def __init__(self, num_classes: int = 3):
        """
        Args:
            num_classes: number of label classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.record: List[Dict] = []
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Update using predicted and ground truth class indices.

        Args:
            preds:  (B, ...) class indices (see MetricsManager._logits_to_preds)
            labels: (B, ...) ground truth class indices
            mask:   optional (B, ...) selection of the cells to score
        """
        if mask is not None:
            preds = preds[mask]
            labels = labels[mask]

        preds_np = preds.detach().cpu().numpy().astype(np.int64).ravel()
        labels_np = labels.detach().cpu().numpy().astype(np.int64).ravel()

        # cells carrying no class (cause is -1 wherever no ignition is labeled)
        # sit outside the matrix and are not scored
        keep = (
            (labels_np >= 0) & (labels_np < self.num_classes)
            & (preds_np >= 0) & (preds_np < self.num_classes)
        )
        if not keep.any():
            return

        flat = labels_np[keep] * self.num_classes + preds_np[keep]
        counts = np.bincount(flat, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute_step(
        self,
        roc_auc: Optional[float] = None,
        pr_auc: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics for the epoch, append to record, and reset internal storage.

        roc_auc / pr_auc:
            Optional AUC scores for this epoch (computed elsewhere from raw logits).
        """
        cm = self.matrix
        total = float(cm.sum())

        if total == 0:
            scores = {
                "mean_iou": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
            self.record.append({**scores, "matrix": cm.copy()})
            self.reset()
            return scores

        # rows are ground truth, columns are predictions
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp

        def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
            # a class absent from both truth and predictions scores 0, not NaN
            return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        iou = _ratio(tp, tp + fp + fn)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2.0 * precision * recall, precision + recall)

        scores = {
            "mean_iou": float(iou.mean()),
            "accuracy": float(tp.sum() / total),
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

        self.record.append({**scores, "matrix": cm.copy()})
        self.reset()

        return scores

    def get_history(self) -> Tuple[np.ndarray | None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], List[Dict]]:
        """
        Returns:
            last_cm: latest confusion matrix (or None if no epochs)
            rates:  (tpr, tnr, fpr, fnr) arrays over epochs (binary-only; empty otherwise)
            record: raw record list
        """
        if not self.record:
            empty = np.zeros(0, dtype=float)
            return None, (empty, empty, empty, empty), []

        matrices = [r["matrix"] for r in self.record]
        last_cm = matrices[-1]

        # For non-binary heads, skip rate computation but still return matrices
        if self.num_classes != 2:
            empty = np.zeros(0, dtype=float)
            return last_cm, (empty, empty, empty, empty), self.record

        # Binary rates per epoch
        tpr_list, tnr_list, fpr_list, fnr_list = [], [], [], []
        for cm in matrices:
            # cm: [[TN, FP],
            #      [FN, TP]]
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn + 1e-6)
            tnr = tn / (tn + fp + 1e-6)
            fpr = fp / (fp + tn + 1e-6)
            fnr = fn / (fn + tp + 1e-6)
            tpr_list.append(tpr)
            tnr_list.append(tnr)
            fpr_list.append(fpr)
            fnr_list.append(fnr)

        rates = (
            np.asarray(tpr_list, dtype=float),
            np.asarray(tnr_list, dtype=float),
            np.asarray(fpr_list, dtype=float),
            np.asarray(fnr_list, dtype=float),
        )
        return last_cm, rates, self.record




class BinaryAUC(Metric):
    """
    ROC-AUC and PR-AUC for a single-logit head, accumulated as per-class
    histograms of the score.

    Threshold-free ranking scores are the only ones that separate a useful
    ignition model from one that answers "no fire" everywhere, since accuracy
    reads ~1.0 at the class ratio this dataset carries. Bucketing scores holds
    memory at O(num_bins) instead of an epoch of per-pixel logits.
    """

    def __init__(self, num_bins: int = 4096, logit_range: Tuple[float, float] = (-16.0, 16.0)):
        super().__init__()
        self.num_bins = num_bins
        self.lo, self.hi = logit_range
        self.pos = np.zeros(num_bins, dtype=np.int64)
        self.neg = np.zeros(num_bins, dtype=np.int64)

    def reset(self) -> None:
        self.pos[:] = 0
        self.neg[:] = 0

    @torch.no_grad()
    def add(self, logits: torch.Tensor, labels: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W) raw scores for the positive class
            labels: (B, H, W) ground truth in {0, 1}
            mask:   optional (B, H, W) selection of the cells to score
        """
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        if labels.numel() == 0:
            return

        # scores outside the range already rank above/below every in-range
        # score, so clamping preserves the ordering the AUCs depend on
        scores = logits.detach().float().flatten().clamp(self.lo, self.hi)
        width = (self.hi - self.lo) / self.num_bins
        bins = ((scores - self.lo) / width).long().clamp(0, self.num_bins - 1)

        bins_np = bins.cpu().numpy()
        labels_np = labels.detach().flatten().cpu().numpy()

        self.pos += np.bincount(bins_np[labels_np == 1], minlength=self.num_bins)
        self.neg += np.bincount(bins_np[labels_np != 1], minlength=self.num_bins)

    def compute_step(self) -> Dict[str, float]:
        n_pos, n_neg = int(self.pos.sum()), int(self.neg.sum())

        if n_pos == 0 or n_neg == 0:
            scores = {"roc_auc": float("nan"), "pr_auc": float("nan")}
            self.record.append(scores)
            self.reset()
            return scores

        # sweep the decision threshold from the highest-scoring bucket downward
        tp = np.cumsum(self.pos[::-1]).astype(np.float64)
        fp = np.cumsum(self.neg[::-1]).astype(np.float64)

        recall = tp / n_pos
        precision = tp / np.maximum(tp + fp, 1.0)
        fpr = fp / n_neg

        # trapezoid over the ROC curve, anchored at the origin
        r = np.concatenate([[0.0], recall])
        f = np.concatenate([[0.0], fpr])
        roc_auc = float(np.sum(np.diff(f) * (r[1:] + r[:-1]) / 2.0))

        # average precision: each threshold's precision weighted by the recall it adds
        pr_auc = float(np.sum(np.diff(r) * precision))

        scores = {"roc_auc": roc_auc, "pr_auc": pr_auc}
        self.record.append(scores)
        self.reset()

        return scores

    def get_history(self) -> List[Dict]:
        return self.record



class MetricsManager:
    def __init__(
        self,
        num_classes: Tuple = (2,),
        select_by: Literal["pr_auc", "val_loss"] = "pr_auc",
    ):
        """
        num_classes: Tuple with number of classes per prediction head
            e.g. one binary classifier + separate 4-class head -> (2, 4)
        select_by: what "best epoch" and early stopping key off.
            "pr_auc"   - masked PR-AUC of the ignition head (maximized)
            "val_loss" - total validation loss (minimized), which sums both
                         heads and so mixes the sparse cause term into the
                         choice of checkpoint
        """
        assert num_classes and num_classes[0] == 2, \
            "head 0 is the binary ignition head; its ranking scores assume a single logit"

        self.num_classes = num_classes
        self.num_heads = len(num_classes)
        self.select_by = select_by

        self.trn_accuracies = [Accuracy() for _ in range(self.num_heads)]
        self.val_accuracies = [Accuracy() for _ in range(self.num_heads)]

        # One confusion matrix per head, with correct class count
        self.val_cm = [ConfusionMatrix(nc) for nc in self.num_classes]

        # Ranking scores for the binary ignition head
        self.val_auc = BinaryAUC()

        # Loss history: (num_loss_terms, num_epochs)
        self.trn_losses: Optional[np.ndarray] = None
        self.val_losses: Optional[np.ndarray] = None

        self.best = {
            "epoch": 0,
            "score": -float("inf") if select_by == "pr_auc" else float("inf"),
        }
        self.epoch = 1
        self.no_improve = 0

    def _is_improvement(self, score: float) -> bool:
        if self.select_by == "pr_auc":
            # an epoch whose eval split carried no positives scores nan and
            # cannot be ranked against anything
            return bool(np.isfinite(score)) and score > self.best["score"]
        return score < self.best["score"]

    @staticmethod
    def _logits_to_preds(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
        """
        Convert a head's logits to class indices.
        Handles:
          - multi-class logits (B, C, ...) -> argmax over C
          - binary logits (B, 1, ...)      -> threshold at 0

        A head whose channel count contradicts its class count is a wiring
        error, and silently reading logits as indices hides it until the
        shapes collide somewhere less obvious.
        """
        if logits.dim() > 1 and logits.size(1) == n_classes and n_classes > 1:
            return torch.argmax(logits, dim=1)

        if logits.dim() > 1 and logits.size(1) == 1 and n_classes == 2:
            return (logits.squeeze(1) > 0).long()

        raise ValueError(
            f"logits {tuple(logits.shape)} do not describe {n_classes} classes; "
            f"expected (B, {n_classes}, ...), or (B, 1, ...) for a binary head"
        )

    @staticmethod
    def _last_day(labels: torch.Tensor) -> torch.Tensor:
        """ Loaders emit (B, H, W) for the window's final day; tolerate (B, T, H, W). """
        return labels[:, -1] if labels.dim() == 4 else labels

    def add(
        self,
        type: Literal["train", "eval"],
        logits: List[torch.Tensor],
        golds: List[torch.Tensor],
        masks: Optional[List[torch.Tensor]] = None,
    ):
        """
        One entry per output head. Each mask selects the cells that head is
        supervised on, so the scores describe the same population as the loss;
        unmasked scores over every cell are dominated by ocean and by the
        no-ignition class.
        """
        assert len(logits) == self.num_heads, f"send one logit tensor for each ({self.num_heads}) output head"
        assert len(golds) == self.num_heads, f"send one golds tensor for each ({self.num_heads}) output head"
        assert masks is None or len(masks) == self.num_heads, \
            f"send one mask tensor for each ({self.num_heads}) output head"

        def mask_for(head: int) -> Optional[torch.Tensor]:
            return masks[head] if masks is not None else None

        accuracies = self.trn_accuracies if type == "train" else self.val_accuracies
        for i, acc in enumerate(accuracies):
            preds_i = self._logits_to_preds(logits[i], self.num_classes[i])
            labels_i = self._last_day(golds[i]).long()
            acc.add(preds_i, labels_i, mask_for(i))

        if type != "eval":
            return

        for i, cm in enumerate(self.val_cm):
            preds_i = self._logits_to_preds(logits[i], self.num_classes[i])
            labels_i = self._last_day(golds[i]).long()
            cm.add(preds_i, labels_i, mask_for(i))

        self.val_auc.add(logits[0], self._last_day(golds[0]).long(), mask_for(0))

    def add_epoch_totals(
        self,
        type: Literal["train", "eval"],
        losses: np.ndarray,
    ):
        """
        losses: 1D array of loss terms for this epoch, e.g. [total, ign, cause]
        Stored as columns in (num_loss_terms, num_epochs).
        """
        new_col = np.asarray(losses).reshape(-1, 1)

        if type == "train":
            if self.trn_losses is None:
                self.trn_losses = new_col
            else:
                self.trn_losses = np.concatenate([self.trn_losses, new_col], axis=1)
        elif type == "eval":
            if self.val_losses is None:
                self.val_losses = new_col
            else:
                self.val_losses = np.concatenate([self.val_losses, new_col], axis=1)

    def epoch_forward(self):
        """
        Print losses for this epoch, update best score, and increment epoch counter.
        Assumes add_epoch_totals() has been called for both train and val.
        Also finalizes per-epoch accuracies and confusion matrices.
        """
        assert self.trn_losses is not None and self.val_losses is not None, "Call add_epoch_totals() before epoch_forward"

        # Finalize accuracies for this epoch (fills .record in Accuracy)
        for acc in self.trn_accuracies:
            acc.compute_step()
        for acc in self.val_accuracies:
            acc.compute_step()

        # Finalize confusion matrices for this epoch (fills .record in ConfusionMatrix).
        # The ignition head carries the epoch's ranking scores alongside its matrix.
        ign_scores = self.val_cm[0].compute_step(**self.val_auc.compute_step())
        for cm in self.val_cm[1:]:
            cm.compute_step()

        trn_last = self.trn_losses[:, -1]
        val_last = self.val_losses[:, -1]

        # the ignition head's ranking quality is the claim under test; total
        # validation loss also carries the sparse, high-variance cause term
        score = float(
            ign_scores["pr_auc"] if self.select_by == "pr_auc" else val_last[0]
        )

        trn_total, trn_ign, trn_cause = trn_last[:3]
        val_total, val_ign, val_cause = val_last[:3]

        print(
            f"[Epoch {self.epoch}]\n"
            f"Train >> mL (total): {trn_total:.4f}, "
            f"mL (ign): {trn_ign:.4f}, "
            f"mL (cause): {trn_cause:.3f}\n"
            f"Eval   >> mL (total): {val_total:.4f}, "
            f"mL (ign): {val_ign:.4f}, "
            f"mL (cause): {val_cause:.3f}\n"
            f"Ignition (supervised cells) >> "
            f"PR-AUC: {ign_scores['pr_auc']:.5f}, "
            f"ROC-AUC: {ign_scores['roc_auc']:.4f}, "
            f"F1: {ign_scores['f1']:.4f}, "
            f"recall: {ign_scores['recall']:.4f}\n"
            f"         SCORE ({self.select_by}): {score:.5f}"
        )

        new_best = False
        if self._is_improvement(score):
            print(f"NEW BEST! SCORE={score:.5f}\n")
            new_best = True
            self.best["epoch"] = self.epoch
            self.best["train_loss"] = trn_last.copy()
            self.best["eval_loss"] = val_last.copy()
            self.best["score"] = score
            self.no_improve = 0
        else:
            self.no_improve += 1

        self.epoch += 1
        return score, new_best, trn_last, val_last
    
    def get_history(self):
        return self.trn_losses, self.val_losses
from typing import Dict, List, Literal, Optional, Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    jaccard_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

class TemperatureScaler(nn.Module):
    """
    Temperature scaling: logit_scaled = logit / T
    T > 0; we optimize log_T for stability: T = exp(log_T).
    """
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # log T, init T=1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature)
        return logits / temperature


class Metric:
    def __init__(self):
        self.record = []
    def reset(self) -> None:
        raise NotImplementedError
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
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
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """ Update accuracy and ground truth labels
        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
        """
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
    Uses sklearn under the hood.
    """

    def __init__(self, num_classes: int = 3):
        """
        Args:
            num_classes: number of label classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.record: List[Dict] = []
        self.y_true: List[np.ndarray] = []
        self.y_pred: List[np.ndarray] = []

    def reset(self):
        self.y_true = []
        self.y_pred = []

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Update using predictions and ground truth labels.

        Args:
            preds:  logits or class indices
                    - (B, C, ...)  -> argmax over C
                    - (B, ...)     -> treated as class indices
            labels: (B, ...) with ground truth class indices
        """
        # Keep labels as a Tensor, derive a NumPy view
        labels_flat = labels.view(-1)
        labels_np = labels_flat.cpu().numpy().astype(int)

        # Convert logits -> predicted classes
        if preds.dim() > 1:
            # Multi-class logits: (B, C, ...)
            if preds.size(1) == self.num_classes:
                preds = torch.argmax(preds, dim=1)
            # Binary logits with single channel: (B, 1, ...)
            elif preds.size(1) == 1 and self.num_classes == 2:
                preds = (preds.squeeze(1) > 0).long()

        preds_flat = preds.view(-1)
        preds_np = preds_flat.cpu().numpy().astype(int)

        self.y_true.append(labels_np)
        self.y_pred.append(preds_np)

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
        if not self.y_true:
            self.record.append({
                "mean_iou": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "matrix": np.zeros((self.num_classes, self.num_classes), dtype=int),
            })
            return {"iou": 0.0, "accuracy": 0.0}

        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)
        labels = np.arange(self.num_classes)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # IoU (Jaccard) per class + macro mean
        iou_per_class = jaccard_score(
            y_true, y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        mean_iou = float(iou_per_class)

        # Precision / Recall / F1 (macro)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        self.record.append({
            "mean_iou": mean_iou,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "matrix": cm,
        })

        self.reset()

        return {
            "iou": mean_iou,
            "accuracy": float(accuracy),
        }
    
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
    





class MetricsManager:
    def __init__(self, num_classes: Tuple = (2,)):
        """
        num_classes: Tuple with number of classes per prediction head
            e.g. one binary classifier + separate 4-class head -> (2, 4)
        """
        self.num_classes = num_classes
        self.num_heads = len(num_classes)

        self.trn_logits: List[List[torch.Tensor]] = []
        self.val_logits: List[List[torch.Tensor]] = []
        self.trn_labels: List[List[torch.Tensor]] = []
        self.val_labels: List[List[torch.Tensor]] = []

        self.trn_accuracies = [Accuracy() for _ in range(self.num_heads)]
        self.val_accuracies = [Accuracy() for _ in range(self.num_heads)]

        # One confusion matrix per head, with correct class count
        self.val_cm = [ConfusionMatrix(nc) for nc in self.num_classes]

        # Loss history: (num_loss_terms, num_epochs)
        self.trn_losses: Optional[np.ndarray] = None
        self.val_losses: Optional[np.ndarray] = None

        self.best = {
            "epoch": 0,
            "score": float("inf"),
            "ign_err": float("inf"),
        }
        self.epoch = 1
        self.no_improve = 0

    @staticmethod
    def _logits_to_preds(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
        """
        Convert logits tensor to class index tensor.
        Handles:
          - binary (single-channel logits) -> threshold at 0
          - multi-class logits -> argmax
          - already-indexed tensors (fallback)
        """
        if logits.dim() == 1:
            return logits.long()

        # Multi-class logits: (B, C, ...)
        if logits.size(1) == n_classes and n_classes > 1:
            return torch.argmax(logits, dim=1)

        # Binary logits with single channel: (B, 1, ...)
        if logits.size(1) == 1 and n_classes == 2:
            return (logits.squeeze(1) > 0).long()

        # Fallback: assume already indices
        return logits.long()

    def add(
        self,
        type: Literal["train", "eval"],
        logits: List[torch.Tensor],
        golds: List[torch.Tensor],
    ):
        assert len(logits) == self.num_heads, f"send one logit tensor for each ({self.num_heads}) output head"
        assert len(golds) == self.num_heads, f"send one golds tensor for each ({self.num_heads}) output head"

        if type == "train":
            self.trn_logits.append(logits)
            self.trn_labels.append(golds)
            for i, acc in enumerate(self.trn_accuracies):
                preds_i = self._logits_to_preds(logits[i], self.num_classes[i])
                labels_i = golds[i].long()
                acc.add(preds_i, labels_i[:, -1])

        elif type == "eval":
            self.val_logits.append(logits)
            self.val_labels.append(golds)

            for i, acc in enumerate(self.val_accuracies):
                preds_i = self._logits_to_preds(logits[i], self.num_classes[i])
                labels_i = golds[i].long()
                acc.add(preds_i, labels_i[:, -1])

            for i, cm in enumerate(self.val_cm):
                preds_i = self._logits_to_preds(logits[i], self.num_classes[i])
                labels_i = golds[i].long()
                cm.add(preds_i, labels_i[: -1])

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

    def compute_val_auc(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ROC-AUC and PR-AUC per head across the full validation epoch.

        Uses stored self.val_logits and self.val_labels.
        Returns:
            roc_aucs: (num_heads,)
            pr_aucs:  (num_heads,)
        NOTE: This assumes you reshape logits/labels appropriately before calling sklearn.
              Currently not used in training loop; safe to leave as-is or extend later.
        """
        roc_aucs: List[float] = []
        pr_aucs: List[float] = []

        for head_idx, n_classes in enumerate(self.num_classes):
            if not self.val_logits:
                roc_aucs.append(float("nan"))
                pr_aucs.append(float("nan"))
                continue

            logits_head = torch.cat([b[head_idx] for b in self.val_logits], dim=0)
            labels_head = torch.cat([b[head_idx] for b in self.val_labels], dim=0)

            # Flatten spatial dimensions if present
            if logits_head.dim() > 2:
                # (B, C, H, W) -> (B*H*W, C)
                c = logits_head.size(1)
                logits_head = logits_head.permute(0, 2, 3, 1).reshape(-1, c)
                labels_head = labels_head.view(-1)

            probs = torch.softmax(logits_head, dim=1).cpu().numpy()
            y = labels_head.cpu().numpy().astype(int)

            if n_classes == 2:
                scores = probs[:, 1]
                roc = roc_auc_score(y, scores)
                pr = average_precision_score(y, scores)
            else:
                y_one_hot = np.eye(n_classes)[y]
                roc = roc_auc_score(
                    y_one_hot, probs, multi_class="ovr", average="macro"
                )
                pr = average_precision_score(
                    y_one_hot, probs, average="macro"
                )

            roc_aucs.append(float(roc))
            pr_aucs.append(float(pr))

        return np.asarray(roc_aucs), np.asarray(pr_aucs)

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

        # Finalize confusion matrices for this epoch (fills .record in ConfusionMatrix)
        for cm in self.val_cm:
            cm.compute_step()

        trn_last = self.trn_losses[:, -1]
        val_last = self.val_losses[:, -1]
        score = float(val_last[0])  # total validation loss

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
            f"         SCORE: {score:.4f}"
        )

        new_best = False
        if score < self.best["score"]:
            print(f"NEW BEST! SCORE={score:.5f}\n")
            new_best = True
            self.best["epoch"] = self.epoch
            self.best["train_loss"] = trn_last.copy()
            self.best["eval_loss"] = val_last.copy()
            self.best["score"] = score
            self.no_improve = 0
        else:
            self.no_improve += 1

        # Clear stored logits/labels so we don't accumulate across epochs
        self.trn_logits.clear()
        self.trn_labels.clear()
        self.val_logits.clear()
        self.val_labels.clear()

        self.epoch += 1
        return score, new_best, trn_last, val_last
    
    def get_history(self):
        return self.trn_losses, self.val_losses
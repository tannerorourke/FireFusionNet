import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp.autocast_mode import autocast

import json
import numpy as np
from typing import Literal, Dict
from tqdm import tqdm
from time import perf_counter

from .dataset.data_loader import init_data_loader
from .model.model import FireFusionModel
from .config.path_config import MODEL_DIR
from .train_utils import (
    estimate_model_size_mb, set_global_seed, get_device_config, 
    save_model, WarmupCosineAnnealingLR
)
from .analysis.metrics import MetricsManager
from .analysis.plots import plot_class_accuracy, plot_loss_curves, plot_rates_per_epoch



class WRMTrainer:
    def __init__(self,
        model_params: Dict,
        training_params: Dict,
        device: torch.device,
        num_workers: int = 0,
        dataset_name: str = "wa2000",
        mode: Literal['train', 'test'] = 'train',
        debug: bool = False
    ):
        self.device = device;
        self.use_amp = bool(device.type == "cuda")
        self.debug = debug

        # fp16 gradients underflow well before the ignition loss does (pos_weight
        # is O(1e4)); the scaler keeps them representable and drops any step that
        # still overflows
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.train_loader = init_data_loader(
            "train", dataset_name, num_workers, training_params["batch_size"]
        )
        self.eval_loader = init_data_loader(
            "eval", dataset_name, num_workers, training_params["batch_size"]
        )

        # channel count, output size, cause classes, and class balance come from
        # the built dataset's manifest, not from hardcoded params
        train_set = self.train_loader.dataset
        model_params = dict(model_params)
        model_params["out_size"] = list(train_set.out_size)
        model_params["n_cause_classes"] = train_set.n_cause_classes
        in_channels = train_set.in_channels
        ign_pos_weight = train_set.ign_pos_weight
        print(f"[WRMTrainer] dataset={dataset_name} in_channels={in_channels} "
              f"out_size={model_params['out_size']} "
              f"cause_classes={model_params['n_cause_classes']} "
              f"pos_weight={ign_pos_weight:.2f} "
              f"prevalence={1.0 / max(ign_pos_weight, 1e-9):.2e} (PR-AUC baseline)")

        self.model = FireFusionModel(in_channels, mp=model_params).to(self.device)


        ep = training_params["epochs"]
        self.ep_warmup, self.ep_max, self.ep_early_stop = ep[0], ep[1], ep[2]
        self.min_lr = training_params["min_lr"]
        self.base_lr = training_params["base_lr"]
        self.weight_decay = training_params["weight_decay"]
        self.grad_clip = training_params["grad_clip"]
        
        self.ign_pos_weight = torch.as_tensor(
            [float(ign_pos_weight)], dtype=torch.float32, device=device
        )
        self.bcewl_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.ign_pos_weight)
        # best epoch / early stopping key off the ignition head's masked PR-AUC
        # rather than total validation loss: it is the reported claim, and it
        # excludes the sparse cause term from the choice of checkpoint
        self.mm = MetricsManager(
            num_classes=(2, train_set.n_cause_classes), select_by="pr_auc"
        )

        if mode == "train": self.train()
        else: self.test()

    @staticmethod
    def _last_day(t: torch.Tensor) -> torch.Tensor:
        """ Loaders emit (B, H, W) for the window's final day; tolerate (B, T, H, W). """
        return t[:, -1] if t.ndim == 4 else t

    def _prepare_targets(self, golds: Dict, masks: Dict):
        """ Collapse any window-time dimension and build the per-head masks of
            supervised cells. Loss and metrics both read these, so the reported
            scores describe the population the model was actually trained on.
        """
        ign_golds = self._last_day(golds["ign_next"])
        cause_golds = self._last_day(golds["ign_next_cause"])
        no_act_fire_mask = self._last_day(masks["no_act_fire_mask"])
        land_mask = self._last_day(masks["land_mask"])

        # masks read 1 where the cell is usable, so this is an AND of the two
        ign_mask = (land_mask == 1) & (no_act_fire_mask == 1)

        # cause is only defined where an ignition at t+1 carries a known cause
        cause_mask = (ign_golds == 1) & (cause_golds != -1) & ign_mask

        return ign_golds, cause_golds, ign_mask, cause_mask

    def _compute_loss(self,
        ign_logits: torch.Tensor, ign_golds: torch.Tensor,  # (B, 1, H, W), (B, H, W)
        cause_logits: torch.Tensor,                         # (B, num_classes, H, W)
        cause_golds: torch.Tensor,                          # (B, H, W)
        ign_mask: torch.Tensor,                             # (B, H, W)
        cause_mask: torch.Tensor,                           # (B, H, W)
        alpha_ign: float = 1.0,
        alpha_cause: float = 1.0
    ):
        """ Compute BCELogitsLoss on ignition at time t + 1,
            as well as cross entropy loss on ignition TYPE given an ignition
        """
        # === Ignition Loss: on ignition at t = t+1 ===============
        ign_logits_flat = ign_logits.squeeze(1)
        ign_targets = ign_golds.float()
        ign_loss = self.bcewl_loss(
            ign_logits_flat,
            ign_targets
        )

        masked_ign_loss = ign_loss * ign_mask
        ign_loss = (
            (masked_ign_loss.sum()) / 
            (ign_mask.sum() + 1e-6)
        )

        # === Cause loss: =========================================
        if cause_mask.any():
            cause_logits_flat = cause_logits.permute(0, 2, 3, 1)[cause_mask]
            cause_targets_flat = cause_golds[cause_mask].long()

            cause_loss = nn.functional.cross_entropy(
                cause_logits_flat, 
                cause_targets_flat, 
                reduction="mean"
            )
        else:
            cause_loss = torch.tensor(0.0, device=ign_logits.device)

        total_loss = (ign_loss * alpha_ign) + (cause_loss * alpha_cause)
        return total_loss, ign_loss, cause_loss

    def train_epoch(self, epoch: int):
        self.model.train()
        ep_total_loss: float = 0.0
        ep_ign_loss: float = 0.0
        ep_cause_loss: float = 0.0
        n_samples: int = 0
        
        for X, golds, masks in tqdm(self.train_loader, desc="Training...", leave=False):
            X = X.to(self.device)
            golds = { k: v.to(self.device) for k, v in golds.items() }
            masks = { k: v.to(self.device) for k, v in masks.items() }

            if epoch == 1 and self.debug:
                print(f"[DataCheck] X shape: {tuple(X.shape)}  (expected: B, T, C, H, W)")
                # Basic tensor stats
                print(f"[DataCheck] Feature min/max: {X.min().item():.4f} / {X.max().item():.4f}")
                print(f"[DataCheck] Feature mean/std: {X.mean().item():.4f} / {X.std().item():.4f}")
                print(f"[DataCheck] Feature NaNs: {torch.isnan(X).sum().item()}")
                print(f"[DataCheck] Feature Infs: {torch.isinf(X).sum().item()}")
                print(f"[DataCheck] golds shape:", golds["ign_next"].shape)
                print(f"[DataCheck] golds dtype:", golds["ign_next"].dtype)
                print(f"[DataCheck] target min/max:", golds["ign_next"].min().item(), golds["ign_next"].max().item())
                print(f"[DataCheck] unique golds (sample):", torch.unique(golds["ign_next"]).cpu()[:10])
                # Per-label checks
                for name, y in golds.items():
                    print(f"[DataCheck] Label '{name}' shape: {tuple(y.shape)}  (expected: B, H, W)")
                    print(f"             unique vals: {torch.unique(y).tolist()}")
                # Per-mask checks
                for name, m in masks.items():
                    print(f"[DataCheck] Mask '{name}' shape: {tuple(m.shape)}  (expected: B, H, W)")
                    print(f"             unique vals: {torch.unique(m).tolist()}")
                # Memory estimate
                bytes_per_batch = X.numel() * X.element_size()
                print(f"[DataCheck] Approx batch memory: {bytes_per_batch/1e6:.2f} MB")
                print("[DataCheck] ✓ Batch looks good.")
            
            self.optimizer.zero_grad(set_to_none=True)
            # lr_used = self.optimizer.param_groups[0]["lr"]

            ign_golds, cause_golds, ign_mask, cause_mask = self._prepare_targets(golds, masks)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                ign_logits, cause_logits = self.model(X)         # (B, 1, H, W), (B, num_classes, H, W)

                tot_loss, ign_loss, cause_loss = self._compute_loss(
                    ign_logits, ign_golds, cause_logits, cause_golds,
                    ign_mask, cause_mask
                )

            # Log total loss
            ep_total_loss += tot_loss.item()
            ep_ign_loss += ign_loss.item()
            ep_cause_loss += cause_loss.item()
            n_samples += golds["ign_next"].size(0)
            self.mm.add('train',
                logits=[ign_logits.detach().cpu(), cause_logits.detach().cpu()],
                golds =[ign_golds.detach().cpu(), cause_golds.detach().cpu()],
                masks =[ign_mask.detach().cpu(), cause_mask.detach().cpu()]
            )

            # Backpropogate -> unscale for clipping -> clip gradients -> step optimizer
            self.scaler.scale(tot_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        self.mm.add_epoch_totals("train", 
            losses=np.array([ep_total_loss, ep_ign_loss, ep_cause_loss])
        )

    def eval_epoch(self, calibration = False):
        self.model.eval()
        ep_total_loss: float = 0.0
        ep_ign_loss: float = 0.0
        ep_cause_loss: float = 0.0
        n_samples: int = 0
        with torch.inference_mode():
            for features, golds, masks in tqdm(self.eval_loader, desc="Evaluating...", leave=False):
                features = features.to(self.device)
                golds = { k: v.to(self.device) for k, v in golds.items() }
                masks = { k: v.to(self.device) for k, v in masks.items() }
                
                ign_golds, cause_golds, ign_mask, cause_mask = self._prepare_targets(golds, masks)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    ign_logits, cause_logits = self.model(features)

                    tot_loss, ign_loss, cause_loss = self._compute_loss(
                        ign_logits, ign_golds, cause_logits, cause_golds,
                        ign_mask, cause_mask
                    )

                # Log total loss for epoch
                ep_total_loss += tot_loss.item()
                ep_ign_loss += ign_loss.item()
                ep_cause_loss += cause_loss.item()
                n_samples += golds["ign_next"].size(0)

                self.mm.add('eval',
                    logits=[ign_logits.detach().cpu(), cause_logits.detach().cpu()],
                    golds =[ign_golds.detach().cpu(), cause_golds.detach().cpu()],
                    masks =[ign_mask.detach().cpu(), cause_mask.detach().cpu()]
                )

        self.mm.add_epoch_totals("eval", 
            np.array([ep_total_loss, ep_ign_loss, ep_cause_loss])
        )

    def train(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.base_lr, 
            weight_decay=self.weight_decay
        )
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer, 
            warmup_steps=self.ep_warmup * max(1, len(self.train_loader)), 
            total_steps=self.ep_max * max(1, len(self.train_loader)), 
            min_lr=self.min_lr
        )

        print(f"Starting training with parameters:\n"
            f"- model size: {estimate_model_size_mb(self.model):.2f}mb\n",
            f"- epochs: {self.ep_warmup} (warmup) {self.ep_max} (total) {self.ep_early_stop} (early stop)\n",
            f"- min lr: {self.min_lr}, base lr: {self.base_lr}, grad clip: {self.grad_clip}, weight decay: {self.weight_decay}\n",
        )

        time0 = perf_counter()
        epochs_ran = 0
        for epoch in range (1, self.ep_max + 1):
            self.train_epoch(epoch)
            self.eval_epoch()

            score, new_best, trn_last, val_last = self.mm.epoch_forward()
            epochs_ran += 1

            # kept under a fixed name so the best weights survive the epochs that follow
            if new_best:
                best_path = save_model(self.model, name_base="wf_risk_model_best", overwrite=True)
                print(f"Saved best weights >> {best_path}\n")

            if self.mm.no_improve > self.ep_early_stop:
                print(f"Stopped training for early stop")
                break

        elapsed_min = (perf_counter() - time0) // 60
        elapsed_sec = (perf_counter() - time0) % 60
        print(f"Finished training in {elapsed_min:.0f} min {elapsed_sec:.0f} sec")
        print(f"Best score @epoch {self.mm.best['epoch']} >> score: {self.mm.best['score']:.5f}")

        final_path = save_model(self.model)
        print(f"Saved final weights >> {final_path}")

        # Do some plotting and fun visualizations!
        

        trn_losses, val_losses = self.mm.get_history()

        trn_ignit_acc, trn_cause_acc = self.mm.trn_accuracies[0], self.mm.trn_accuracies[1]
        val_ignit_acc, val_cause_acc = self.mm.val_accuracies[0], self.mm.val_accuracies[1]

        val_ignit_cm = self.mm.val_cm[0]
        last_ign_cm, ign_rates, ign_cm_record = val_ignit_cm.get_history()

        epochs_axis = list(range(1, epochs_ran + 1))
        
        # Train vs. Eval
        plot_class_accuracy(
            epochs_axis, 
            val_ignit_acc, val_cause_acc, 
            trn_ignit_acc, trn_cause_acc, 
            save=True
        )
        plot_loss_curves(
            epochs_axis, 
            trn_losses, val_losses, 
            save=True
        )
        
        tpr, tnr, fpr, fnr = ign_rates
        plot_rates_per_epoch(epochs_axis, ign_rates, save=True)

    def test(self):
        return
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train FireFusionNet on a built dataset")
    parser.add_argument("--dataset", default="wa2000")
    parser.add_argument("--profile", default="sanity", help="params.json profile")
    args = parser.parse_args()

    set_global_seed(42)
    device, num_workers = get_device_config(maximum=2)

    """ Model Params """
    with open(f'{MODEL_DIR}/params.json') as file:
        data = json.load(file)
        params = data[args.profile]

    model_params        = params["model"]
    training_params     = params["training"]

    wt = WRMTrainer(
        model_params, training_params,
        device, num_workers,
        dataset_name = args.dataset,
        mode = "train",
        debug = False
    )
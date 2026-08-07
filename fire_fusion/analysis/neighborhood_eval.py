"""Neighborhood-tolerant scoring of a saved checkpoint on the eval split.

Ignition placement within a few cells is partly aleatoric, so per-cell scores
carry a noise floor that grows as cells shrink. This scores the neighborhood
event 'any supervised ignition within r km' alongside the per-cell metrics;
fixed-km radii keep the numbers comparable across resolution tiers.

  python -m fire_fusion.analysis.neighborhood_eval --experiment wa4000-s1
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

from ..config.dataset_config import get_dataset_config
from ..config.path_config import MODEL_DIR, MODEL_SAVE_DIR
from ..dataset.data_loader import init_data_loader
from ..model.model import FireFusionModel
from ..train_utils import checkpoint_name, get_device_config, load_model, set_global_seed
from .metrics import BinaryAUC, MeanIgnorance


def _last_day(t: torch.Tensor) -> torch.Tensor:
    return t[:, -1] if t.ndim == 4 else t


def _feed(acc: dict, z: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    acc["auc"].add(z, labels, mask)
    acc["ign"].add(z, labels, mask)
    sel = labels[mask]
    acc["n"] += int(sel.numel())
    acc["pos"] += int((sel == 1).sum().item())


def main():
    ap = argparse.ArgumentParser(description="Score a checkpoint per-cell and per km-neighborhood")
    ap.add_argument("--experiment", required=True, help="params.json experiment name")
    ap.add_argument("--dataset", default=None, help="override the dataset the experiment names")
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint base name (default <experiment>_model, the best weights)")
    ap.add_argument("--radius-km", type=float, nargs="+", default=[4.0, 8.0])
    ap.add_argument("--calibrated", action="store_true",
                    help="apply the fitted calibrator sidecar instead of the analytic +log r shift")
    args = ap.parse_args()

    with open(f"{MODEL_DIR}/params.json") as f:
        data = json.load(f)
        if args.experiment not in data:
            raise SystemExit(f"Unknown experiment '{args.experiment}'. Options: {sorted(data)}")
        params = data[args.experiment]

    tp = params["training"]
    mp = dict(params["model"])
    dataset = args.dataset if args.dataset is not None else params["dataset"]
    seed = tp["seed"]
    device, num_workers = get_device_config(maximum=tp.get("max_workers", 8))
    set_global_seed(seed)

    loader = init_data_loader(
        "eval", dataset, num_workers, tp["batch_size"],
        window_size=tp.get("window_size", 10), window_stride=tp.get("window_stride", 2),
        seed=seed, encoder_depth=mp.get("encoder_depth", 1),
        attn_window=mp["win_spatial_mixing"]["window_size"],
    )
    ds = loader.dataset
    mp["n_cause_classes"] = ds.n_cause_classes
    mp["dyn_groups"] = ds.dyn_groups

    model = FireFusionModel(ds.dyn_channels, ds.static_channels, mp=mp).to(device)
    ckpt = args.checkpoint or checkpoint_name(args.experiment)
    load_model(model, f"{ckpt}.th", map_location=device)
    model.eval()

    # -- either path maps raw logits to real-prior probabilities: the analytic
    #    subsampling shift, or the fitted affine whose intercept subsumes it
    a, b = 1.0, math.log(float(tp.get("neg_keep_rate", 1.0)))
    if args.calibrated:
        with open(MODEL_SAVE_DIR / f"{ckpt}.calib.json") as f:
            calib = json.load(f)
        a, b = float(calib["a"]), float(calib["b"])

    res_m = get_dataset_config(dataset).resolution
    halos = {f"{r:g}km": max(1, round(r * 1000 / res_m)) for r in args.radius_km}
    scopes = {name: {"auc": BinaryAUC(), "ign": MeanIgnorance(), "n": 0, "pos": 0}
              for name in ("cell", *halos)}

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for (x_dyn, x_static), golds, masks in tqdm(loader, desc="Scoring...", leave=False):
            x_dyn, x_static = x_dyn.to(device), x_static.to(device)
            ign_golds = _last_day(golds["ign_next"]).to(device)
            ign_mask = (
                (_last_day(masks["land_mask"]).to(device) == 1)
                & (_last_day(masks["no_act_fire_mask"]).to(device) == 1)
            )

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                ign_logits, _ = model(x_dyn, x_static)

            z = a * ign_logits.squeeze(1).float() + b
            _feed(scopes["cell"], z, ign_golds, ign_mask)

            # -- P(any ignition in the window) composes supervised cells under
            #    independence; log-space keeps the product exact at ~1e-3 rates
            p = torch.sigmoid(z.clamp(max=16.0)) * ign_mask.float()
            log_keep = torch.log1p(-p).unsqueeze(1)
            pos = (ign_golds.float() * ign_mask.float()).unsqueeze(1)

            for name, h in halos.items():
                k = 2 * h + 1
                # zero padding reads as certain-no-fire outside the frame, which
                # the strict mask below excludes from scoring anyway
                s = (F.avg_pool2d(log_keep, k, stride=1, padding=h) * (k * k)).squeeze(1)
                z_nb = (torch.log((-torch.expm1(s)).clamp_min(1e-12)) - s).clamp(-30.0, 30.0)
                lab_nb = (F.max_pool2d(pos, k, stride=1, padding=h).squeeze(1) > 0).long()

                # -- scored centers need every neighbor supervised, else censored
                #    labels would read as calm neighborhoods
                m = F.pad(ign_mask.float().unsqueeze(1), (h, h, h, h), value=0.0)
                mask_nb = (-F.max_pool2d(-m, k, stride=1)).squeeze(1) == 1
                _feed(scopes[name], z_nb, lab_nb, mask_nb)

    mode = "calibrated" if args.calibrated else "analytic +log r"
    print(f"[NeighborhoodEval] experiment={args.experiment} dataset={dataset} "
          f"checkpoint={ckpt} logits={mode} resolution={res_m:g}m")
    print(f"{'scope':>8} | {'ign (bits)':>10} | {'recall@prev':>11} | "
          f"{'ROC-AUC':>8} | {'PR-AUC':>8} | {'prevalence':>10} | {'cells':>12}")
    for name, acc in scopes.items():
        s = acc["auc"].compute_step()
        bits = acc["ign"].compute_step()["ign_bits"]
        prev = acc["pos"] / max(acc["n"], 1)
        print(f"{name:>8} | {bits:>10.5f} | {s['recall_at_prev']:>11.4f} | "
              f"{s['roc_auc']:>8.4f} | {s['pr_auc']:>8.5f} | {prev:>10.2e} | {acc['n']:>12,}")


if __name__ == "__main__":
    main()

"""Turn a trained FireFusion checkpoint into per-cell ignition probabilities.

An input is a spatiotemporal cube (B, T, C, H, W) spanning days [t_0..t_n]; the
output is a (B, 1, H, W) map of P(fresh ignition at t_{n+1}) in [0, 1]. The model
emits raw logits; a fitted Platt calibrator maps them to probabilities. Absent a
fitted sidecar, the analytic prior correction for the training class weight
(sigmoid(z - log(pos_weight))) stands in, so a checkpoint is always usable.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config.dataset_config import get_dataset_config
from .config.path_config import MODEL_DIR, PLOTS_DIR
from .dataset.data_loader import init_data_loader
from .model.model import FireFusionModel
from .analysis.metrics import PlattScaler
from .train_utils import load_model, load_calibration, get_device_config


class FirePredictor:
    """ A checkpoint plus its calibrator, applied to input cubes. """
    def __init__(self, model: FireFusionModel, calibrator: PlattScaler, device: torch.device):
        self.model = model.eval()
        self.calibrator = calibrator
        self.device = device

    @torch.no_grad()
    def predict_proba(
        self, cube: torch.Tensor, land_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ cube: (B, T, C, H, W) -> (B, 1, H, W) probabilities for t_{n+1}.

        land_mask (1 where usable) marks non-land cells NaN so an ocean cell is
        never read as a fire probability.
        """
        cube = cube.to(self.device)
        ign_logits, _ = self.model(cube)              # (B, 1, H, W)
        probs = self.calibrator.probs(ign_logits.float())

        if land_mask is not None:
            lm = land_mask.to(probs.device)
            if lm.dim() == probs.dim() - 1:           # (B, H, W) -> (B, 1, H, W)
                lm = lm.unsqueeze(1)
            probs = probs.masked_fill(lm != 1, float("nan"))
        return probs


def load_predictor(
    dataset_name: str = "wa2000",
    profile: str = "optimal",
    checkpoint: str = "specialized_model.th",
    calib: str | None = None,
    device: torch.device | None = None,
) -> FirePredictor:
    """ Rebuild the model at its trained shape, load weights, attach a calibrator.

    Channel count, grid size, cause classes, and the class weight come from the
    dataset manifest (as at train time); the attention/embedding shape comes from
    the params.json `profile` the checkpoint was trained with, so a mismatched
    profile surfaces immediately as a strict state_dict error.
    """
    if device is None:
        device, _ = get_device_config(maximum=1)

    manifest = json.loads(get_dataset_config(dataset_name).manifest_path.read_text())
    in_channels = int(manifest["in_channels"])
    pos_weight = float(manifest["ign_pos_weight"])

    with open(f"{MODEL_DIR}/params.json") as f:
        model_params = dict(json.load(f)[profile]["model"])
    model_params["out_size"] = [int(manifest["grid"]["height"]), int(manifest["grid"]["width"])]
    model_params["n_cause_classes"] = int(manifest["n_cause_classes"])

    model = FireFusionModel(in_channels, mp=model_params).to(device)
    load_model(model, checkpoint, map_location=device)
    model.eval()

    scaler = PlattScaler(prior_pos_weight=pos_weight).to(device)
    sidecar = calib if calib is not None else Path(checkpoint).stem
    params = load_calibration(sidecar)
    if params is not None:
        scaler.load_state(params)
        print(f"[predict] calibration a={params['a']:.4f} b={params['b']:.4f} "
              f"(ECE {params.get('ece_before', float('nan')):.4f} -> "
              f"{params.get('ece_after', float('nan')):.4f})")
    else:
        print(f"[predict] no calibration sidecar for '{sidecar}'; analytic prior "
              f"b=-log(pos_weight)={-math.log(pos_weight):.4f}")

    return FirePredictor(model, scaler, device)


def _last_day(t: torch.Tensor) -> torch.Tensor:
    return t[:, -1] if t.dim() == 4 else t


if __name__ == "__main__":
    from .analysis.plots import plot_XY_grid

    parser = argparse.ArgumentParser(
        description="Predict per-cell ignition probability for t_{n+1}"
    )
    parser.add_argument("--dataset", default="wa2000")
    parser.add_argument("--profile", default="optimal",
                        help="params.json profile the checkpoint was trained with")
    parser.add_argument("--checkpoint", default="specialized_model.th")
    parser.add_argument("--calib", default=None,
                        help="calibration sidecar name; defaults to the checkpoint's")
    parser.add_argument("--split", default="eval", choices=["train", "eval", "test"])
    parser.add_argument("--batches", type=int, default=1,
                        help="how many batches to summarize and plot")
    args = parser.parse_args()

    predictor = load_predictor(args.dataset, args.profile, args.checkpoint, args.calib)
    loader = init_data_loader(args.split, args.dataset, num_workers=0, batch_size=1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, (features, _golds, masks) in enumerate(loader):
        land = _last_day(masks["land_mask"])
        probs = predictor.predict_proba(features, land_mask=land)   # (B, 1, H, W)

        finite = probs[torch.isfinite(probs)]
        print(f"[predict] batch {i}: P(fire) over land  min={finite.min():.3e}  "
              f"mean={finite.mean():.3e}  max={finite.max():.3e}")

        grid = probs[0, 0].cpu().numpy()
        vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
        plot_XY_grid(
            grid, land_mask=land[0],
            title=f"P(fire at t_+1)  [{args.split} #{i}]",
            vmin=0.0, vmax=vmax,
            save_path=str(PLOTS_DIR / f"pred_proba_{args.split}_{i}.png"),
        )

        if i + 1 >= args.batches:
            break

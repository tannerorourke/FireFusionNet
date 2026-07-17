import os
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from .config.path_config import MODEL_SAVE_DIR


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    """ Naive way to estimate model size """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_config(maximum: int | None = None, utilization: float | None = 0.75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpus = os.cpu_count() or 1

    if utilization is not None:
        workers = math.floor(cpus * utilization)
    else:
        workers = 1
    if maximum is not None:
        workers = max(1, min(workers, maximum))

    if torch.cuda.is_available():
        print(f"Device: {device}, {torch.cuda.get_device_name(0)}")
    print(f"Using {workers}/{cpus} CPUs")
    return device, workers


def save_model(
    model: torch.nn.Module,
    name_base: str = "wf_risk_model",
    overwrite: bool = False,
) -> str:
    """ Use this function to save your model in train.py

    overwrite: write to `<name_base>.th` instead of taking the next free
    `<name_base>_<i>.th`, for checkpoints that are re-saved as a run improves.
    """
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if overwrite:
        output_path = MODEL_SAVE_DIR / f"{name_base}.th"
    else:
        i = 1
        while (Path(MODEL_SAVE_DIR / f"{name_base}_{i}.th").exists()):
            i += 1
        output_path = MODEL_SAVE_DIR / f"{name_base}_{i}.th"

    torch.save(model.state_dict(), output_path)

    return str(output_path)


def load_model(
    model: torch.nn.Module,
    path: str | os.PathLike,
    map_location=None,
    strict: bool = True,
):
    """ Restore weights written by save_model into an existing model.

    A bare name (no directory component) is resolved against MODEL_SAVE_DIR,
    mirroring save_model's output layout, so `load_model(m, "main_model.th")`
    pairs with `save_model(m, name_base="main_model", overwrite=True)`.

    strict=False tolerates a checkpoint that only covers part of the model
    (e.g. a backbone loaded into a model with freshly initialized heads); the
    returned value lists whatever keys were missing or unexpected.
    """
    p = Path(path)
    if p.parent == Path("."):
        p = MODEL_SAVE_DIR / p

    state = torch.load(p, map_location=map_location)
    return model.load_state_dict(state, strict=strict)


def save_calibration(params: dict, name_base: str = "specialized_model") -> str:
    """ Write a calibrator sidecar next to its `<name_base>.th` checkpoint.

    The probabilities a checkpoint produces depend on both its weights and the
    fitted calibration, so the two travel together under a shared name.
    """
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_SAVE_DIR / f"{name_base}.calib.json"
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    return str(output_path)


def load_calibration(name_base: str = "specialized_model") -> dict | None:
    """ Load a calibrator sidecar by checkpoint name, or None if absent.

    A bare `<name_base>` (no directory) resolves against MODEL_SAVE_DIR; a path
    ending in `.calib.json` is read as given. Absent means "no fit available",
    which the predictor answers with the analytic prior correction.
    """
    p = Path(name_base)
    if p.suffix != ".json":
        p = MODEL_SAVE_DIR / f"{p.name}.calib.json"
    elif p.parent == Path("."):
        p = MODEL_SAVE_DIR / p
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


class WarmupCosineAnnealingLR:
    """ 
    PyTorch CosineAnnealing learning rate, with a linear warmup step
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
        https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """
    def __init__(self,
        optimizer,
        warmup_steps: int, total_steps: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer

        w_steps = max(0, warmup_steps)
        base_lr = float(optimizer.param_groups[0]["lr"])

        # LinearLR scales base_lr by start_factor and requires it in (0, 1]. A
        # base_lr of 0 (the null-learning profile) leaves the ratio undefined,
        # and any factor of zero is still zero, so warm up at full scale.
        start_factor = min_lr / base_lr if base_lr > 0 else 1.0
        start_factor = float(min(1.0, max(start_factor, 1e-8)))

        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = start_factor if w_steps > 0 else 1.0,
            total_iters = warmup_steps if warmup_steps > 0 else 1
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = max(1, int(total_steps - warmup_steps)),
            eta_min = min_lr,
        )
        self.sched = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[w_steps]
        )

    def step(self): self.sched.step()
    def state_dict(self): return self.sched.state_dict()
    def load_state_dict(self, sd): self.sched.load_state_dict(sd)
    def get_last_lr(self): return self.sched.get_last_lr()
    @property
    def last_epoch(self): return self.sched.last_epoch
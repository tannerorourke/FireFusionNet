from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.calibration import calibration_curve
import torch

from .metrics import Accuracy

from ..config.path_config import PLOTS_DIR


def plot_class_accuracy(
    epochs: List,
    val_ignit_acc: Accuracy,
    val_cause_acc: Accuracy,
    trn_ignit_acc: Accuracy,
    trn_cause_acc: Accuracy,
    save: bool = True,
):
    plt.figure()
    plt.plot(epochs, trn_ignit_acc.record, label="Ignition acc (train)")
    plt.plot(epochs, trn_cause_acc.record, label="Cause acc (train)")
    plt.plot(epochs, val_ignit_acc.record, label="Ignition acc (val)")
    plt.plot(epochs, val_cause_acc.record, label="Cause acc (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation accuracy per epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(exist_ok=True, parents=True)
        plt.savefig(PLOTS_DIR / "class_accuracy.png", bbox_inches="tight", dpi=200)
    else:
        plt.show()


def plot_loss_curves(
    epochs: List,
    trn_losses,
    val_losses,
    term_names: Tuple[str, ...] = ("total", "ignition", "cause"),
    save: bool = True,
):
    # histories arrive as (num_loss_terms, num_epochs); one line per term
    trn_losses = np.atleast_2d(np.asarray(trn_losses, dtype=float))
    val_losses = np.atleast_2d(np.asarray(val_losses, dtype=float))

    def term_label(i: int) -> str:
        return term_names[i] if i < len(term_names) else f"term {i}"

    plt.figure()
    for i, curve in enumerate(trn_losses):
        plt.plot(epochs, curve, label=f"Train loss ({term_label(i)})")
    for i, curve in enumerate(val_losses):
        plt.plot(epochs, curve, linestyle="--", label=f"Val loss ({term_label(i)})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(exist_ok=True, parents=True)
        plt.savefig(PLOTS_DIR / "losses.png", bbox_inches="tight", dpi=200)
    else:
        plt.show()


def plot_rates_per_epoch(epochs: List, rates: Tuple, save=True):
    tpr, tnr, fpr, fnr = rates

    plt.figure()
    plt.plot(epochs, tpr, label="TPR (recall)")
    plt.plot(epochs, tnr, label="TNR (specificity)")
    plt.plot(epochs, fpr, label="FPR")
    plt.plot(epochs, fnr, label="FNR")
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("Ignition rates per epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(exist_ok=True, parents=True)
        plt.savefig(PLOTS_DIR / "rates.png", bbox_inches="tight", dpi=200)
    else:
        plt.show()



def precision_recall_history(record: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given confusion-matrix record entries, return macro precision and recall for each epoch.
    """
    precisions = np.array([r["precision"] for r in record], dtype=float)
    recalls = np.array([r["recall"] for r in record], dtype=float)
    return precisions, recalls


def auc_history(record: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract PR-AUC and ROC-AUC per epoch from record (if present).
    """
    pr_aucs = [r["pr_auc"] for r in record if r.get("pr_auc") is not None]
    roc_aucs = [r["roc_auc"] for r in record if r.get("roc_auc") is not None]
    return np.asarray(pr_aucs, dtype=float), np.asarray(roc_aucs, dtype=float)


def auc_vs_f1_history(
    record: List[Dict],
    which_auc: str = "roc_auc",  # or "pr_auc"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (AUC, F1) pairs per epoch for plotting.
    """
    aucs = []
    f1s = []
    for r in record:
        auc_val = r.get(which_auc, None)
        if auc_val is None:
            continue
        aucs.append(float(auc_val))
        f1s.append(float(r["f1"]))
    return np.asarray(aucs), np.asarray(f1s)


def reliability_diagram(
    probs: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    num_bins: int = 10,
    title: str = "Reliability diagram",
    save_path: str | None = None,
):
    """
    Compute and plot a reliability diagram (calibration curve) using sklearn.calibration_curve.
    probs:  [N] predicted probabilities in [0,1]
    labels: [N] true labels {0,1}
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    frac_pos, mean_pred = calibration_curve(
        labels, probs, n_bins=num_bins, strategy="uniform"
    )

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_XY_grid(
    grid_2d: np.ndarray,
    water_mask: torch.Tensor | np.ndarray | None = None,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
):
    """
    Plot a heatmap of a continuous field [H, W] with water cells in black.
    """
    data = np.array(grid_2d, dtype=float)

    if water_mask is not None:
        if isinstance(water_mask, torch.Tensor):
            water_mask = water_mask.detach().cpu().numpy()
        water_mask = np.array(water_mask).astype(bool)
        data = np.ma.masked_where(water_mask, data)

    cmap = plt.cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="black")

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    plt.figure()
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Value")
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()



def plot_label_grid_time_t(
    labels: torch.Tensor | np.ndarray,
    water_mask: torch.Tensor | np.ndarray | None = None,
    time_idx: int | None = None,
    title: str = "Fire labels",
    save_path: str | None = None,
):
    """
    Plot binary labels as a discrete heatmap:
      - 0 (no fire) = blue
      - 1 (fire)    = red
      - water       = black
    labels can be:
      - [T, H, W]
      - [H, W]
      - [1, H, W]
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.array(labels)

    if labels.ndim == 3 and labels.shape[0] != 1:
        # assume [T, H, W]
        if time_idx is None:
            raise ValueError("time_idx must be provided for label grid with shape [T, H, W]")
        grid_2d = labels[time_idx]
    elif labels.ndim == 2:
        grid_2d = labels
    elif labels.ndim == 3 and labels.shape[0] == 1:
        grid_2d = labels[0]
    else:
        raise ValueError(f"Unsupported labels shape: {labels.shape}")

    grid_2d = grid_2d.astype(float)

    if water_mask is not None:
        if isinstance(water_mask, torch.Tensor):
            water_mask = water_mask.detach().cpu().numpy()
        water_mask = np.array(water_mask).astype(bool)
        grid_2d = np.ma.masked_where(water_mask, grid_2d)

    cmap = ListedColormap(["blue", "red"])
    cmap.set_bad(color="black")

    plt.figure()
    im = plt.imshow(grid_2d, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, ticks=[0, 1])
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_probability_surface_3d(
    probs_2d: torch.Tensor | np.ndarray,
    title: str = "Ignition probability surface",
    save_path: str | None = None
):
    """
    probs_2d: [H, W] probability grid in [0,1]
    """
    if isinstance(probs_2d, torch.Tensor):
        probs_2d = probs_2d.detach().cpu().numpy()
    probs_2d = np.array(probs_2d)

    H, W = probs_2d.shape
    xs = np.arange(W)
    ys = np.arange(H)
    X, Y = np.meshgrid(xs, ys)

    Z = probs_2d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)

    ax.set_xlabel("X (col)")
    ax.set_ylabel("Y (row)")
    ax.set_zlabel("P(fire)")
    ax.set_title(title)
    ax.view_init(elev=30, azim=225)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()
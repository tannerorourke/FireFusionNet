"""Per-channel feature diagnostic for the built cube.

Answers whether input channels carry no independent, target-relevant
signal and could be dropped for training. 

Outputs: Three (not per-pixel) per-channel views :

  1. Spread     -- global std over land cells; a near-constant channel is dead.
  2. Redundancy -- channel-channel correlation + PCA loadings; a channel that is
                   a linear combination of others adds no independent variance.
  3. Relevance  -- supervised signal against the real per-cell `ign_next` label:
                   mutual information and random-forest permutation importance.

  python -m fire_fusion.analysis.pca --dataset wa2000 --out <dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config.path_config import PROCESSED_DATA_DIR


DERIVED = {
    "lightning_load", "fire_spatial_roll", "precip_2d", "precip_5d",
    "fosberg_fwi", "dead_fmo_100hr", "ndvi_anomaly", "doy_sin",
    "modis_months_since_last_burn", "kde_debris", "kde_human",
    "kde_industrial", "kde_natural_lightning",
}

DEAD_STD = 1e-6
REDUNDANT_R = 0.95 # |corr| >= is near-duplication
NEG_CAP = 400_000 # -- reservoir bound on sampled negative cells


def _split_path(dataset: str, split: str) -> Path:
    return PROCESSED_DATA_DIR / dataset / f"{split}.zarr"


def collect(dataset: str, split: str, device: str, neg_per_pos: int,
            time_chunk: int, seed: int):
    """ Stream the split cube in time chunks; return a class-balanced per-cell
        channel matrix over land cells plus global per-channel spread. """
    ds = xr.open_zarr(_split_path(dataset, split))
    if "channel" in ds["X"].coords:
        names = [str(n) for n in ds["X"].coords["channel"].values]
    else:
        names = [str(n) for n in ds.attrs.get("channels",
                                              range(ds["X"].sizes["channel"]))]
    C = ds["X"].sizes["channel"]
    T = ds["X"].sizes["time"]
    rng = np.random.default_rng(seed)

    g_sum = torch.zeros(C, dtype=torch.float64, device=device)
    g_sqsum = torch.zeros(C, dtype=torch.float64, device=device)
    g_n = 0
    pos_rows = []
    neg_rows = []
    # -- each time chunk holds the same number of days, so an equal per-chunk cap
    # yields a roughly uniform negative sample bounded by NEG_CAP overall
    n_chunks = (T + time_chunk - 1) // time_chunk
    per_chunk_neg = max(1, NEG_CAP // n_chunks)

    for t0 in range(0, T, time_chunk):
        t1 = min(t0 + time_chunk, T)
        X = torch.as_tensor(
            ds["X"].isel(time=slice(t0, t1)).values, device=device
        )                                            # (tc, C, H, W)
        ign = torch.as_tensor(
            ds["ign_next"].isel(time=slice(t0, t1)).values, device=device
        )                                            # (tc, H, W)
        land = torch.as_tensor(
            ds["land_mask"].isel(time=slice(t0, t1)).values, device=device
        ).bool()

        Xf = X.permute(0, 2, 3, 1).reshape(-1, C)    # (tc*H*W, C)
        landf = land.reshape(-1)
        yf = ign.reshape(-1) > 0
        Xland = Xf[landf]
        yland = yf[landf]

        g_sum += Xland.sum(0).double()
        g_sqsum += (Xland.double() ** 2).sum(0)
        g_n += Xland.shape[0]

        pos_rows.append(Xland[yland].cpu().numpy())

        neg = Xland[~yland].cpu().numpy()
        if neg.shape[0] > per_chunk_neg:
            sel = rng.choice(neg.shape[0], size=per_chunk_neg, replace=False)
            neg = neg[sel]
        neg_rows.append(neg)
        del X, ign, land, Xf, Xland

    Xpos = np.concatenate(pos_rows, axis=0)
    neg_all = np.concatenate(neg_rows, axis=0)
    g_mean = (g_sum / g_n).cpu().numpy()
    g_std = torch.sqrt(
        torch.clamp((g_sqsum / g_n) - (g_sum / g_n) ** 2, min=0.0)
    ).cpu().numpy()

    n_pos = Xpos.shape[0]
    n_neg = min(neg_all.shape[0], neg_per_pos * max(n_pos, 1))
    idx = rng.choice(neg_all.shape[0], size=n_neg, replace=False)
    Xneg = neg_all[idx]

    Xc = np.concatenate([Xpos, Xneg], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    return Xc, y, np.asarray(g_std), np.asarray(g_mean), names, int(g_n)


def run(dataset: str, split: str, out: Path, device: str, neg_per_pos: int,
        time_chunk: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    Xc, y, g_std, g_mean, names, n_land = collect(
        dataset, split, device, neg_per_pos, time_chunk, seed)
    C = len(names)
    n_pos = int(y.sum())
    print(f"[pca] {dataset}/{split}: {n_land:,} land cell-days, "
          f"{n_pos:,} positives, matrix {Xc.shape}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)

    # -- redundancy: correlation + PCA loadings
    corr = np.corrcoef(Xs.T)
    off = corr.copy()
    np.fill_diagonal(off, 0.0)
    off = np.nan_to_num(off)
    max_abs_corr = np.abs(off).max(axis=1)
    corr_partner = [names[i] for i in np.abs(off).argmax(axis=1)]

    pca = PCA(random_state=seed).fit(Xs)
    scree = pca.explained_variance_ratio_
    loadings = pca.components_                    # (C, C)

    # -- relevance: mutual information + RF permutation importance
    mi = mutual_info_classif(Xs, y, discrete_features=False, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.3, random_state=seed, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    # -- single-process: loky worker spawning exhausts the POSIX semaphore
    # namespace under WSL and fails with ENOSPC despite free memory
    perm = permutation_importance(
        rf, Xte, yte, n_repeats=10, random_state=seed,
        scoring="average_precision", n_jobs=1).importances_mean
    gini = rf.feature_importances_

    dead = g_std < DEAD_STD
    mi_q = np.quantile(mi, 0.25)
    perm_q = np.quantile(perm, 0.25)
    drop = dead | ((mi <= mi_q) & (perm <= perm_q) & (max_abs_corr >= REDUNDANT_R))

    rows = []
    for i, nm in enumerate(names):
        rows.append(dict(
            channel=nm, derived=nm in DERIVED, global_std=float(g_std[i]),
            dead=bool(dead[i]), max_abs_corr=float(max_abs_corr[i]),
            corr_partner=corr_partner[i], mi=float(mi[i]),
            rf_perm_ap=float(perm[i]), rf_gini=float(gini[i]),
            pc1_loading=float(loadings[0, i]), drop_candidate=bool(drop[i]),
        ))
    order = np.argsort(perm)            # least important first
    rows = [rows[i] for i in order]

    _write_csv(out / f"{dataset}_channels.csv", rows)
    np.savez(out / f"{dataset}_raw.npz", corr=corr, scree=scree,
             loadings=loadings, g_std=g_std, g_mean=g_mean,
             mi=mi, perm=perm, gini=gini, names=np.asarray(names))
    (out / f"{dataset}_summary.json").write_text(json.dumps(dict(
        dataset=dataset, split=split, n_land_cell_days=n_land, n_pos=n_pos,
        n_channels=C, matrix_shape=list(Xc.shape),
        cum_var_at_k={str(k): float(np.cumsum(scree)[k - 1])
                      for k in (5, 10, 20, min(C, 30))},
        drop_candidates=[r["channel"] for r in rows if r["drop_candidate"]],
    ), indent=2))
    _plots(out, dataset, names, g_std, corr, scree, loadings, mi, perm)

    print(f"\n[pca] {dataset} channel ranking (least -> most RF-important):")
    print(f"  {'channel':<28}{'MI':>8}{'perm':>9}{'|r|max':>8}  flags")
    for r in rows:
        flags = []
        if r["dead"]: flags.append("DEAD")
        if r["drop_candidate"]: flags.append("DROP?")
        if r["derived"]: flags.append("derived")
        print(f"  {r['channel']:<28}{r['mi']:>8.4f}{r['rf_perm_ap']:>9.4f}"
              f"{r['max_abs_corr']:>8.3f}  {','.join(flags)}")
    cands = [r["channel"] for r in rows if r["drop_candidate"]]
    print(f"\n[pca] drop candidates ({len(cands)}): {cands or 'none'}")
    print(f"[pca] artifacts >> {out}")
    return rows


def _write_csv(path: Path, rows):
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def _plots(out, dataset, names, g_std, corr, scree, loadings, mi, perm):
    C = len(names)
    order = np.argsort(perm)

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, C + 1), np.cumsum(scree), marker="o", ms=3)
    plt.axhline(0.95, ls="--", c="grey", lw=0.8)
    plt.xlabel("PCs"); plt.ylabel("cumulative explained var")
    plt.title(f"{dataset} PCA scree"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out / f"{dataset}_scree.png", dpi=150); plt.close()

    for tag, val, lbl in [("mi", mi, "mutual information"),
                          ("perm", perm, "RF permutation AP")]:
        plt.figure(figsize=(7, max(4, C * 0.22)))
        o = np.argsort(val)
        plt.barh(range(C), val[o])
        plt.yticks(range(C), [names[i] for i in o], fontsize=7)
        plt.xlabel(lbl); plt.title(f"{dataset} {lbl} vs ign_next")
        plt.tight_layout(); plt.savefig(out / f"{dataset}_{tag}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(7, max(4, C * 0.22)))
    o = np.argsort(g_std)
    plt.barh(range(C), np.maximum(g_std[o], 1e-9))
    plt.xscale("log"); plt.yticks(range(C), [names[i] for i in o], fontsize=7)
    plt.axvline(DEAD_STD, ls="--", c="red", lw=0.8)
    plt.xlabel("global std (log)"); plt.title(f"{dataset} channel spread")
    plt.tight_layout(); plt.savefig(out / f"{dataset}_std.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 7))
    plt.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(C), [names[i] for i in order], rotation=90, fontsize=6)
    plt.yticks(range(C), [names[i] for i in order], fontsize=6)
    plt.title(f"{dataset} channel correlation"); plt.tight_layout()
    plt.savefig(out / f"{dataset}_corr.png", dpi=150); plt.close()

    k = min(10, C)
    vmax = np.abs(loadings[:k]).max()
    plt.figure(figsize=(8, max(4, C * 0.22)))
    plt.imshow(loadings[:k].T, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(fraction=0.046)
    plt.xticks(range(k), [f"PC{i+1}" for i in range(k)])
    plt.yticks(range(C), names, fontsize=7)
    plt.title(f"{dataset} PCA loadings"); plt.tight_layout()
    plt.savefig(out / f"{dataset}_loadings.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wa2000")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--neg-per-pos", type=int, default=30)
    ap.add_argument("--time-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[pca] device={device}")
    torch.set_grad_enabled(False)
    run(args.dataset, args.split, args.out, device,
        args.neg_per_pos, args.time_chunk, args.seed)


if __name__ == "__main__":
    main()

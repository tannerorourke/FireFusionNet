import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch

from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from ..dataset.data_loader import FireDataset, init_data_loader
from ..config.path_config import TRAIN_DATA_DIR, EVAL_DATA_DIR


def extract_features_and_labels_from_batch(batch):
    """
    Extract a feature matrix and a 1D label vector from a single batch.

    Parameters
    ----------
    batch : tuple
        Expected format from FireDataset.collate_batch:
        (X, label_tensors, mask_tensors)
        where:
            X              : (B, T, C, H, W)
            label_tensors  : dict[str, Tensor] each (B, T, H, W)
            mask_tensors   : dict[str, Tensor] each (B, T, H, W)

    Returns
    -------
    X_flat : np.ndarray
        Shape (B, F) where F = T * C * H * W.
    y_window : np.ndarray
        Shape (B,), binary label for each window:
        1 if ANY cell/time in that window is positive in the first label tensor, else 0.
    """
    X, label_tensors, mask_tensors = batch

    # # Ensure we have the expected 5D tensor
    # if X.dim() != 5:
    #     raise ValueError(f"Expected X to have 5 dimensions (B, T, C, H, W), got {X.shape}")

    batch_size = X.shape[0]

    # Move features to CPU and flatten: (B, T, C, H, W) -> (B, F)
    X_np = X.detach().cpu().numpy()
    X_flat = X_np.reshape(batch_size, -1)

    # # Take the first label tensor in the dict as "the" label source
    # if len(label_tensors) == 0:
    #     raise ValueError("label_tensors is empty; need at least one label for classification.")

    first_label_tensor = next(iter(label_tensors.values()))
    label_np = first_label_tensor.detach().cpu().numpy()  # shape (B, T, H, W)

    # Window-level binary label: 1 if ANY cell/time is positive, else 0
    # Axis order: (B, T, H, W) -> reduce across (T, H, W) for each B
    y_window = (label_np > 0).any(axis=(1, 2, 3)).astype(np.int64)  # shape (B,)
    return X_flat, y_window


def build_feature_matrix_and_labels_from_loaders(
    loaders,
    max_samples: int = 20_000,
):
    """
    Build a 2D feature matrix and a 1D label vector from a list of loaders.

    Each batch corresponds to one or more spatiotemporal windows.
    For each window, we produce:
        - One flattened feature vector.
        - One binary label.

    Parameters
    ----------
    loaders : list[DataLoader]
        e.g. [train_loader, eval_loader]
    max_samples : int
        Upper bound on total number of windows to use.

    Returns
    -------
    X_all : np.ndarray, shape (n_samples, n_features)
    y_all : np.ndarray, shape (n_samples,)
    """
    feature_chunks = []
    label_chunks = []
    total_samples = 0

    for loader in loaders:
        for batch in loader:
            X_flat, y_window = extract_features_and_labels_from_batch(batch)

            feature_chunks.append(X_flat)
            label_chunks.append(y_window)

            total_samples += X_flat.shape[0]
            if total_samples >= max_samples:
                break

        if total_samples >= max_samples:
            break

    if not feature_chunks:
        raise RuntimeError("No samples collected; check your loaders or max_samples.")

    X_all = np.concatenate(feature_chunks, axis=0)[:max_samples]
    y_all = np.concatenate(label_chunks, axis=0)[:max_samples]

    return X_all, y_all
    


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    
    # -------------------------------------------------------------------------
    # 1) Build datasets and loaders (batch_size=1 for quick sampling)
    # -------------------------------------------------------------------------
    batch_size = 1
    window_size = 10
    window_stride = 2
    max_samples = 20_000  # adjust if you want more/less

    use_gpu = torch.cuda.is_available()

    # Create loaders with batch_size=1 and no multiprocessing (simplest / least headache)
    train_loader = init_data_loader("train", num_workers=1, batch_size=1)
    eval_loader = init_data_loader("eval", num_workers=1, batch_size=1)

    # -------------------------------------------------------------------------
    # 2) Build feature matrix and labels from loaders
    # -------------------------------------------------------------------------
    X, y = build_feature_matrix_and_labels_from_loaders(
        loaders=[train_loader, eval_loader],
        max_samples=max_samples,
    )

    print(f"Feature matrix shape: {X.shape}")  # (n_samples, n_features)
    print(f"Label vector shape:   {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip(unique.tolist(), counts.tolist())))

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    #    Option A: fixed number of components
    #    Option B: variance threshold, e.g. n_components=0.95
    pca = PCA(n_components=32, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    print("\nPCA:")
    print("  PCA-transformed shape:", X_pca.shape)
    print("  Explained variance ratio (first 10 PCs):", explained_var[:10])
    print("  Cumulative explained variance (first 10 PCs):", cumulative_var[:10])

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------------------------------------------------
    # 6) Very basic Logistic Regression classifier
    # -------------------------------------------------------------------------
    logistic_clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )
    logistic_clf.fit(X_train, y_train)
    y_pred_logistic = logistic_clf.predict(X_test)

    print("\nLogistic Regression on PCA features:")
    print(classification_report(y_test, y_pred_logistic, digits=3))

    # -------------------------------------------------------------------------
    # 7) Very basic Random Forest classifier
    # -------------------------------------------------------------------------
    rf_clf = RandomForestClassifier(
        n_estimators=100,      # small forest, just to get a quick sense
        max_depth=None,        # let trees grow fully (can overfit; fine for exploration)
        n_jobs=-1,
        random_state=42,
    )
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    print("\nRandom Forest on PCA features:")
    print(classification_report(y_test, y_pred_rf, digits=3))

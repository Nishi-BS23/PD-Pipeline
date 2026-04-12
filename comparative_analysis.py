"""Comparative analysis: Wav2Vec2 vs HuBERT.

Usage:
    python comparative_analysis.py --mode full     # uses embeddings_full/
    python comparative_analysis.py --mode segment  # uses embeddings_aggregated/

Sections:
    1. Data loading
    2. Dimensionality reduction (PCA, t-SNE, UMAP)
    3. DBSCAN clustering
    4. MLP classifier + random hyperparameter search
    5. Training monitoring (loss curves)
    6. Evaluation metrics (confusion matrix, precision/recall/F1, AUROC, AUPRC)
    7. Comparative summary
"""

import argparse
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("WARNING: umap-learn not installed. UMAP plots will be skipped.")
    print("         Install with: pip install umap-learn")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

MODEL_CONFIGS = {
    "Wav2Vec2": {
        "segment_meta":  os.path.join(PROJECT_ROOT, "Wav2Vec2", "metadata_aggregated.csv"),
        "full_meta":     os.path.join(PROJECT_ROOT, "Wav2Vec2", "metadata_full.csv"),
        "color":         "#1f77b4",
    },
    "HuBERT": {
        "segment_meta":  os.path.join(PROJECT_ROOT, "HuBERT", "metadata_aggregated.csv"),
        "full_meta":     os.path.join(PROJECT_ROOT, "HuBERT", "metadata_full.csv"),
        "color":         "#d62728",
    },
}

CLASS_COLORS = {"PD": "#e74c3c", "HC": "#2ecc71"}
RANDOM_STATE  = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_output_dir(mode: str) -> str:
    out = os.path.join(PROJECT_ROOT, "results", "comparative_analysis", mode)
    os.makedirs(out, exist_ok=True)
    return out


def load_embeddings(
    meta_path: str,
    label_col: str = "class",
    max_per_class: int | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load mean-pooled embeddings and binary labels from a metadata CSV.

    Returns (X, y) where X is (N, D) float32 and y is (N,) int (PD=1, HC=0).
    """
    df = pd.read_csv(meta_path)

    if max_per_class is not None:
        # Balanced cap: take up to N samples from each class independently.
        df = df.copy()
        df[label_col] = df[label_col].astype(str).str.strip()
        rng = np.random.default_rng(seed)
        keep_parts = []
        for cls_name in ["PD", "HC"]:
            cls_df = df[df[label_col] == cls_name]
            if len(cls_df) > max_per_class:
                # Sample rows by index to keep output deterministic with seed.
                idx = rng.choice(cls_df.index.to_numpy(), size=max_per_class, replace=False)
                cls_df = cls_df.loc[np.sort(idx)]
            keep_parts.append(cls_df)
        df = pd.concat(keep_parts, ignore_index=True)
    emb_col = "embedding_path"

    X, y = [], []
    for _, row in df.iterrows():
        path = row[emb_col]
        if not os.path.isfile(path):
            continue
        X.append(np.load(path).astype(np.float32))
        y.append(1 if str(row[label_col]).strip() == "PD" else 0)

    if not X:
        raise FileNotFoundError(f"No valid embedding files found from {meta_path}")
    return np.stack(X, axis=0), np.array(y, dtype=np.int64)


def scale(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val)


# ---------------------------------------------------------------------------
# Section 1: Data loading
# ---------------------------------------------------------------------------

def load_all_embeddings(
    mode: str,
    max_per_class: int | None = None,
    seed: int = RANDOM_STATE,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load embeddings for both models. Returns {model_name: (X, y)}."""
    meta_key = "full_meta" if mode == "full" else "segment_meta"
    data = {}
    for name, cfg in MODEL_CONFIGS.items():
        meta_path = cfg[meta_key]
        if not os.path.isfile(meta_path):
            print(f"WARNING: {meta_path} not found. Run {name}/pipeline.py --mode {mode} first.")
            continue
        X, y = load_embeddings(meta_path, max_per_class=max_per_class, seed=seed)
        print(f"  {name}: {X.shape[0]} samples, dim={X.shape[1]}, "
              f"PD={y.sum()}, HC={(y==0).sum()}")
        data[name] = (X, y)
    return data


# ---------------------------------------------------------------------------
# Section 2: Dimensionality reduction
# ---------------------------------------------------------------------------

def plot_dim_reduction(data: dict, out_dir: str) -> None:
    """PCA, t-SNE, UMAP scatter plots for both models, coloured by class."""
    reducers = {
        "PCA":  lambda: PCA(n_components=2, random_state=RANDOM_STATE),
        "t-SNE": lambda: TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                               max_iter=1000),
    }
    if UMAP_AVAILABLE:
        reducers["UMAP"] = lambda: umap.UMAP(n_components=2, random_state=RANDOM_STATE)

    n_models  = len(data)
    n_reducers = len(reducers)
    fig, axes = plt.subplots(n_models, n_reducers,
                             figsize=(5 * n_reducers, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (model_name, (X, y)) in enumerate(data.items()):
        for col, (rname, builder) in enumerate(reducers.items()):
            ax = axes[row, col]
            reducer = builder()
            Z = reducer.fit_transform(X)
            for cls_idx, (label, cls_name) in enumerate([(0, "HC"), (1, "PD")]):
                mask = y == cls_idx
                ax.scatter(Z[mask, 0], Z[mask, 1], c=CLASS_COLORS[cls_name],
                           label=cls_name, alpha=0.5, s=10, linewidths=0)
            ax.set_title(f"{model_name} — {rname}", fontsize=11)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            if row == 0 and col == 0:
                ax.legend(markerscale=3)

    plt.suptitle("Dimensionality Reduction — PD vs HC", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "dim_reduction_class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Section 3: DBSCAN clustering
# ---------------------------------------------------------------------------

def run_dbscan_configs(X: np.ndarray, y: np.ndarray,
                       eps_cosine: float = 0.3, eps_euclidean: float = 5.0,
                       min_samples: int = 5) -> list[dict]:
    """Run 4 DBSCAN configs and return summary dicts."""
    # UMAP reduction for the reduced-input configs
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=10, random_state=RANDOM_STATE)
        X_reduced = reducer.fit_transform(X)
    else:
        pca = PCA(n_components=10, random_state=RANDOM_STATE)
        X_reduced = pca.fit_transform(X)

    configs = [
        ("raw", "cosine",    X,       DBSCAN(eps=eps_cosine,    min_samples=min_samples, metric="cosine")),
        ("raw", "euclidean", X,       DBSCAN(eps=eps_euclidean, min_samples=min_samples, metric="euclidean")),
        ("reduced", "cosine",    X_reduced, DBSCAN(eps=eps_cosine, min_samples=min_samples, metric="cosine")),
        ("reduced", "euclidean", X_reduced, DBSCAN(eps=10.0,       min_samples=min_samples, metric="euclidean")),
    ]

    results = []
    for space, metric, X_in, db in configs:
        labels = db.fit_predict(X_in)
        n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = (labels == -1).mean()
        ari = adjusted_rand_score(y, labels) if n_clusters > 0 else 0.0
        nmi = normalized_mutual_info_score(y, labels) if n_clusters > 0 else 0.0
        results.append({
            "space":        space,
            "metric":       metric,
            "n_clusters":   n_clusters,
            "noise_pct":    round(noise_ratio * 100, 1),
            "ARI":          round(ari, 4),
            "NMI":          round(nmi, 4),
        })
    return results


def run_clustering_analysis(data: dict, out_dir: str) -> None:
    all_rows = []
    for model_name, (X, y) in data.items():
        print(f"  Running DBSCAN for {model_name}...")
        rows = run_dbscan_configs(X, y)
        for r in rows:
            r["model"] = model_name
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)[["model", "space", "metric",
                                          "n_clusters", "noise_pct", "ARI", "NMI"]]
    path = os.path.join(out_dir, "dbscan_summary.csv")
    summary_df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Section 4: MLP classifier
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP with two hidden layers and Softmax output.

    Architecture:
        Linear(in_dim, hidden) -> ReLU -> Dropout
        Linear(hidden, hidden) -> ReLU -> Dropout
        Linear(hidden, 2)      -> (logits, used with CrossEntropyLoss)
    """

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights to handle PD/HC imbalance."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    # return as float tensor ordered by class index 0,1
    w = np.ones(len(classes))
    for cls, wt in zip(classes, weights):
        w[cls] = wt
    return torch.tensor(w, dtype=torch.float32)


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              hidden: int, dropout: float, lr: float,
              batch_size: int, max_epochs: int = 50,
              patience: int = 5, device: str = "cpu") -> tuple[nn.Module, list, list]:
    """Train MLP; return (model, train_losses, val_losses)."""
    X_tr_sc, X_val_sc = scale(X_tr, X_val)

    tensor = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
    itensor = lambda a: torch.tensor(a, dtype=torch.long).to(device)

    train_ds = TensorDataset(tensor(X_tr_sc), itensor(y_tr))
    val_ds   = TensorDataset(tensor(X_val_sc), itensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    model = MLP(X_tr.shape[1], hidden, dropout).to(device)
    cw    = compute_class_weights(y_tr).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        train_losses.append(ep_loss / len(train_ds))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                vl += criterion(model(xb), yb).item() * len(xb)
        val_loss = vl / len(val_ds)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


def random_hparam_search(X_tr: np.ndarray, y_tr: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          n_trials: int = 20, device: str = "cpu") -> tuple[dict, pd.DataFrame]:
    """Random search over MLP hyperparameters.

    Search space:
        lr          : log-uniform [1e-5, 1e-3]
        hidden_size : {128, 256, 512}
        dropout     : uniform [0.1, 0.5]
        batch_size  : {32, 64, 128}
    """
    rng = np.random.default_rng(RANDOM_STATE)
    records = []
    best_val_acc = -1.0
    best_cfg: dict = {}

    X_val_sc = StandardScaler().fit(X_tr).transform(X_val)
    Xv_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
    yv_t = torch.tensor(y_val, dtype=torch.long).to(device)

    for trial in range(n_trials):
        cfg = {
            "lr":          float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-3)))),
            "hidden_size": int(rng.choice([128, 256, 512])),
            "dropout":     float(rng.uniform(0.1, 0.5)),
            "batch_size":  int(rng.choice([32, 64, 128])),
        }
        model, _, _ = train_mlp(
            X_tr, y_tr, X_val, y_val,
            hidden=cfg["hidden_size"], dropout=cfg["dropout"],
            lr=cfg["lr"], batch_size=cfg["batch_size"],
            max_epochs=50, patience=5, device=device,
        )
        model.eval()
        with torch.no_grad():
            preds = model(Xv_t).argmax(dim=1).cpu().numpy()
        val_acc = (preds == y_val).mean()
        rec = {"trial": trial + 1, "val_acc": round(val_acc, 4), **cfg}
        records.append(rec)
        print(f"    Trial {trial+1:2d}/{n_trials}  acc={val_acc:.4f}  {cfg}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cfg = cfg

    trials_df = pd.DataFrame(records)
    print(f"  Best val_acc={best_val_acc:.4f}  cfg={best_cfg}")
    return best_cfg, trials_df


# ---------------------------------------------------------------------------
# Section 5: Training monitoring
# ---------------------------------------------------------------------------

def plot_training_curves(curves: dict[str, tuple[list, list]], out_dir: str) -> None:
    """Plot train/val loss curves for both models side-by-side."""
    fig, axes = plt.subplots(1, len(curves), figsize=(7 * len(curves), 5))
    if len(curves) == 1:
        axes = [axes]

    for ax, (model_name, (train_losses, val_losses)) in zip(axes, curves.items()):
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train loss", color="#1f77b4")
        ax.plot(epochs, val_losses,   label="Val loss",   color="#d62728")
        ax.set_title(f"{model_name} — Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend()
        # overfitting marker
        if val_losses:
            best_ep = int(np.argmin(val_losses)) + 1
            ax.axvline(best_ep, color="grey", linestyle="--", alpha=0.7,
                       label=f"Best val epoch={best_ep}")
            ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Section 6: Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, X_val: np.ndarray, y_val: np.ndarray,
             X_tr: np.ndarray, device: str) -> dict:
    """Compute all evaluation metrics on validation set."""
    X_val_sc = StandardScaler().fit(X_tr).transform(X_val)
    Xv_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(Xv_t).cpu().numpy()

    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
    preds = probs.argmax(axis=1)
    prob_pd = probs[:, 1]  # probability of PD class

    cm = confusion_matrix(y_val, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary",
                                                         pos_label=1, zero_division=0)
    auroc = roc_auc_score(y_val, prob_pd) if len(np.unique(y_val)) > 1 else 0.0
    auprc = average_precision_score(y_val, prob_pd) if len(np.unique(y_val)) > 1 else 0.0
    acc   = (preds == y_val).mean()

    return {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "auroc":     round(auroc, 4),
        "auprc":     round(auprc, 4),
        "cm":        cm,
        "y_val":     y_val,
        "prob_pd":   prob_pd,
        "preds":     preds,
    }


def plot_evaluation(results: dict[str, dict], out_dir: str) -> None:
    """Confusion matrices, ROC curves, PR curves."""
    model_names = list(results.keys())
    n = len(model_names)

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        cm = results[name]["cm"]
        disp = ConfusionMatrixDisplay(cm, display_labels=["HC", "PD"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- ROC curves ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, cfg in MODEL_CONFIGS.items():
        if name not in results:
            continue
        r = results[name]
        fpr, tpr, _ = roc_curve(r["y_val"], r["prob_pd"])
        ax.plot(fpr, tpr, label=f"{name} (AUROC={r['auroc']:.3f})", color=cfg["color"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- PR curves ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, cfg in MODEL_CONFIGS.items():
        if name not in results:
            continue
        r = results[name]
        prec_arr, rec_arr, _ = precision_recall_curve(r["y_val"], r["prob_pd"])
        ax.plot(rec_arr, prec_arr,
                label=f"{name} (AUPRC={r['auprc']:.3f})", color=cfg["color"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    path = os.path.join(out_dir, "pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Section 7: Comparative summary
# ---------------------------------------------------------------------------

def print_summary(metrics: dict[str, dict], dbscan_df: pd.DataFrame, out_dir: str) -> None:
    rows = []
    for model_name, m in metrics.items():
        # best ARI from DBSCAN
        best_ari = dbscan_df[dbscan_df["model"] == model_name]["ARI"].max() \
            if not dbscan_df.empty else float("nan")
        rows.append({
            "Model":     model_name,
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1":        m["f1"],
            "AUROC":     m["auroc"],
            "AUPRC":     m["auprc"],
            "Best ARI":  round(best_ari, 4),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "comparative_summary.csv")
    df.to_csv(path, index=False)

    print(f"\n{'='*70}")
    print("  COMPARATIVE SUMMARY")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"\nSaved: {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparative analysis: Wav2Vec2 vs HuBERT."
    )
    parser.add_argument(
        "--mode", choices=["segment", "full"], default="segment",
        help="Which embeddings to load: 'segment' = aggregated, 'full' = full audio"
    )
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of random hyperparameter search trials per model")
    parser.add_argument("--final-epochs", type=int, default=100,
                        help="Epochs for final model training after hparam search")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for validation")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Balanced cap per class: N means use up to N PD and N HC samples (0 = no cap)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used for capped class sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = make_output_dir(args.mode)

    print(f"\n{'='*60}")
    print(f"  Comparative Analysis  [mode={args.mode}]")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    print(f"  Output    : {out_dir}")
    print(f"  Trials    : {args.n_trials}")
    print(f"  Test size : {args.test_size}")
    print(f"  Max/class : {args.max_per_class if args.max_per_class > 0 else 'all'}")
    print(f"  Seed      : {args.seed}")

    # -- Section 1: load embeddings --
    print(f"\n[1/7] Loading embeddings...")
    max_per_class = args.max_per_class if args.max_per_class > 0 else None
    data = load_all_embeddings(args.mode, max_per_class=max_per_class, seed=args.seed)
    if not data:
        print("ERROR: No embeddings loaded. Run pipeline.py first.")
        sys.exit(1)

    # -- Section 2: dim reduction --
    print(f"\n[2/7] Dimensionality reduction (PCA, t-SNE{', UMAP' if UMAP_AVAILABLE else ''})...")
    plot_dim_reduction(data, out_dir)

    # -- Section 3: DBSCAN --
    print(f"\n[3/7] DBSCAN clustering...")
    dbscan_df = pd.DataFrame()
    all_dbscan = []
    for model_name, (X, y) in data.items():
        rows = run_dbscan_configs(X, y)
        for r in rows:
            r["model"] = model_name
        all_dbscan.extend(rows)
    if all_dbscan:
        dbscan_df = pd.DataFrame(all_dbscan)[["model", "space", "metric",
                                               "n_clusters", "noise_pct", "ARI", "NMI"]]
        csv_path = os.path.join(out_dir, "dbscan_summary.csv")
        dbscan_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        print(dbscan_df.to_string(index=False))

    # -- Sections 4-6: per-model MLP training, monitoring, evaluation --
    training_curves: dict[str, tuple[list, list]] = {}
    eval_results: dict[str, dict] = {}
    hparam_dfs: dict[str, pd.DataFrame] = {}

    for model_name, (X, y) in data.items():
        print(f"\n[4/7] Hyperparameter search — {model_name}...")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
        )
        best_cfg, trials_df = random_hparam_search(
            X_tr, y_tr, X_val, y_val, n_trials=args.n_trials, device=device
        )
        hparam_dfs[model_name] = trials_df
        csv_path = os.path.join(out_dir, f"hparam_search_{model_name.lower()}.csv")
        trials_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        print(f"[5/7] Final training — {model_name} (epochs={args.final_epochs})...")
        best_model, tr_losses, val_losses = train_mlp(
            X_tr, y_tr, X_val, y_val,
            hidden=best_cfg["hidden_size"], dropout=best_cfg["dropout"],
            lr=best_cfg["lr"], batch_size=best_cfg["batch_size"],
            max_epochs=args.final_epochs, patience=10, device=device,
        )
        training_curves[model_name] = (tr_losses, val_losses)

        print(f"[6/7] Evaluation — {model_name}...")
        metrics = evaluate(best_model, X_val, y_val, X_tr, device)
        eval_results[model_name] = metrics
        print(f"  Accuracy={metrics['accuracy']}  Precision={metrics['precision']}  "
              f"Recall={metrics['recall']}  F1={metrics['f1']}  "
              f"AUROC={metrics['auroc']}  AUPRC={metrics['auprc']}")

    # -- Section 5 plots (all models together) --
    if training_curves:
        plot_training_curves(training_curves, out_dir)

    # -- Section 6 plots --
    if eval_results:
        plot_evaluation(eval_results, out_dir)

        metrics_rows = []
        for model_name, m in eval_results.items():
            metrics_rows.append({
                "model": model_name,
                "accuracy":  m["accuracy"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "f1":        m["f1"],
                "auroc":     m["auroc"],
                "auprc":     m["auprc"],
            })
        mdf = pd.DataFrame(metrics_rows)
        mdf.to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False)

    # -- Section 7: summary --
    print(f"\n[7/7] Comparative summary...")
    if eval_results:
        print_summary(eval_results, dbscan_df, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

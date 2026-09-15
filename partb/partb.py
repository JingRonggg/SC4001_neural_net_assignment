"""
Part B: Classification Problem — Chest X-ray Binary Classification
SC4001 Neural Networks and Deep Learning

Classifies 64x64 greyscale chest X-ray images into healthy (0) vs pneumonia (1).
Covers questions B1 (AlexNet-like), B2 (VGG-like), B3 (ResNet-like), B4 (Grad-CAM).

Usage:
    uv run partb.py
"""

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
MAX_EPOCHS = 20
PATIENCE = 4
MIN_DELTA = 1e-4
DATA_PATH = "chestxray_binary_64.npz"
OUT_DIR = "results"


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            print("CUDA available but unusable (no compatible kernels). Falling back to CPU.")
    return torch.device("cpu")

DEVICE = _get_device()

# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_data():
    data = np.load(DATA_PATH)

    train_img = data["train_images"].astype(np.float32) / 255.0
    val_img   = data["val_images"].astype(np.float32) / 255.0
    test_img  = data["test_images"].astype(np.float32) / 255.0

    train_labels = data["train_labels"].astype(np.float32)
    val_labels   = data["val_labels"].astype(np.float32)
    test_labels  = data["test_labels"].astype(np.float32)

    # Normalize using training-set statistics
    mean = train_img.mean()
    std  = train_img.std()
    print(f"Train mean={mean:.4f}, std={std:.4f}")

    train_img = (train_img - mean) / std
    val_img   = (val_img  - mean) / std
    test_img  = (test_img - mean) / std

    # (N,H,W) -> (N,1,H,W)
    X_train = torch.tensor(train_img[:, None, :, :])
    y_train = torch.tensor(train_labels)
    X_val   = torch.tensor(val_img[:, None, :, :])
    y_val   = torch.tensor(val_labels)
    X_test  = torch.tensor(test_img[:, None, :, :])
    y_test  = torch.tensor(test_labels)

    print(f"Train: {X_train.shape}  (0={int((y_train==0).sum())}, 1={int((y_train==1).sum())})")
    print(f"Val:   {X_val.shape}  (0={int((y_val==0).sum())}, 1={int((y_val==1).sum())})")
    print(f"Test:  {X_test.shape}  (0={int((y_test==0).sum())}, 1={int((y_test==1).sum())})")

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)

    return train_ds, val_ds, test_ds, X_test, y_test, mean, std


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels):
    preds  = np.asarray(preds)
    labels = np.asarray(labels)
    acc = float((preds == labels).mean())
    TP = int(((preds == 1) & (labels == 1)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return acc, float(sens), float(spec)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb).view(-1)
            total_loss += criterion(out, yb).item() * xb.size(0)
            preds = (torch.sigmoid(out) >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA,
                verbose=True):
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = copy.deepcopy(model.state_dict())
    best_epoch    = 0

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [],
        "val_sens": [], "val_spec": [],
    }

    for epoch in range(1, max_epochs + 1):
        # --- train ---
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb).view(-1), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        # --- validate ---
        val_loss, vp, vl = evaluate_model(model, val_loader, criterion)
        acc, sens, spec = compute_metrics(vp, vl)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_sens"].append(sens)
        history["val_spec"].append(spec)

        if verbose:
            print(f"  Epoch {epoch:2d}/{max_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={acc:.4f}")

        # --- early stopping ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model, history


# ---------------------------------------------------------------------------
# Parameter / FLOPs counting
# ---------------------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size=(1, 1, 64, 64)):
    """Run a single forward pass on CPU to count FLOPs via hooks."""
    model_cpu = copy.deepcopy(model).cpu()
    total = {"v": 0}
    hooks = []

    def _conv(m, _i, o):
        kh, kw = m.kernel_size
        total["v"] += 2 * kh * kw * (m.in_channels // m.groups) * o.shape[1] * o.shape[2] * o.shape[3]

    def _fc(m, _i, _o):
        total["v"] += 2 * m.in_features * m.out_features

    for m in model_cpu.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_fc))

    model_cpu.eval()
    with torch.no_grad():
        model_cpu(torch.randn(*input_size))
    for h in hooks:
        h.remove()
    return total["v"]


# ===================================================================
# MODEL DEFINITIONS
# ===================================================================

# ---------------------------------------------------------------------------
# B1 — AlexNet-like
# ---------------------------------------------------------------------------

class AlexNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)

        self.pool    = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 16×32×32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 32×16×16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 64×8×8
        x = self.avgpool(x)                               # 64×4×4
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# B2 — VGG-like
# ---------------------------------------------------------------------------

class VGGLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 8, 3, 1, 1, bias=False),  nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, 1, 1, bias=False),  nn.BatchNorm2d(8),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(8, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# B3 — ResNet-like
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)

        self.block1 = ResidualBlock(16, 16, stride=1)

        self.block2 = ResidualBlock(16, 32, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(16, 32, 1, stride=2, bias=False),
                nn.BatchNorm2d(32),
            ))

        self.block3 = ResidualBlock(32, 64, stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(32, 64, 1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            ))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 16×32×32
        x = self.block1(x)                               # 16×32×32
        x = self.block2(x)                               # 32×16×16
        x = self.block3(x)                               # 64×8×8
        x = self.avgpool(x)                               # 64×1×1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ===================================================================
# B1 — LEARNING RATE SEARCH (AlexNet-like)
# ===================================================================

def run_b1(train_ds, val_ds, test_ds):
    print("\n" + "=" * 60)
    print("B1 — AlexNet-like: learning-rate search")
    print("=" * 60)

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    BS = 128
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False)

    results = {}
    for lr in learning_rates:
        print(f"\n--- LR={lr} ---")
        set_seed(SEED)
        model = AlexNetLike().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model, hist = train_model(model, train_loader, val_loader, criterion, opt)

        be = hist["best_epoch"]
        results[lr] = {
            "model": model, "history": hist, "best_epoch": be,
            "val_loss":  hist["val_loss"][be - 1],
            "val_acc":   hist["val_acc"][be - 1],
            "val_error": 1 - hist["val_acc"][be - 1],
            "sens":      hist["val_sens"][be - 1],
            "spec":      hist["val_spec"][be - 1],
        }

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for lr in learning_rates:
        h = results[lr]["history"]
        ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"],
                marker="o", markersize=3, label=f"LR={lr}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Loss")
    ax.set_title("B1: Validation Loss vs Epoch (different learning rates)")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "b1_val_loss.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for lr in learning_rates:
        h = results[lr]["history"]
        ax.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"],
                marker="o", markersize=3, label=f"LR={lr}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
    ax.set_title("B1: Validation Accuracy vs Epoch (different learning rates)")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "b1_val_acc.png"), dpi=150)
    plt.close()

    # ---- Table ----
    print(f"\n{'LR':>10} | {'Best Ep':>7} | {'Val Loss':>9} | {'Val Err':>9} | {'Sens':>9} | {'Spec':>9}")
    print("-" * 65)
    for lr in learning_rates:
        r = results[lr]
        print(f"{lr:>10.0e} | {r['best_epoch']:>7d} | {r['val_loss']:>9.4f} | "
              f"{r['val_error']:>9.4f} | {r['sens']:>9.4f} | {r['spec']:>9.4f}")

    best_lr = min(learning_rates, key=lambda lr: results[lr]["val_error"])
    print(f"\nSelected LR = {best_lr}  "
          f"(val classification error = {results[best_lr]['val_error']:.4f})")

    # ---- Test ----
    best_model = results[best_lr]["model"]
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)
    _, tp, tl = evaluate_model(best_model, test_loader, criterion)
    acc, sens, spec = compute_metrics(tp, tl)
    print(f"\nB1 Test  —  Accuracy: {acc:.4f}  Sensitivity: {sens:.4f}  Specificity: {spec:.4f}")

    return best_lr, best_model, acc, sens, spec


# ===================================================================
# B2 — BATCH SIZE SEARCH (VGG-like)
# ===================================================================

def run_b2(train_ds, val_ds, test_ds, best_lr):
    print("\n" + "=" * 60)
    print(f"B2 — VGG-like: batch-size search  (LR={best_lr})")
    print("=" * 60)

    batch_sizes = [16, 32, 64, 128, 256]
    criterion = nn.BCEWithLogitsLoss()

    results = {}
    for bs in batch_sizes:
        print(f"\n--- BS={bs} ---")
        set_seed(SEED)
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True)
        vl = DataLoader(val_ds,   batch_size=bs, shuffle=False)

        model = VGGLike().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=best_lr)
        model, hist = train_model(model, tl, vl, criterion, opt)

        be = hist["best_epoch"]
        results[bs] = {
            "model": model, "history": hist, "best_epoch": be,
            "val_loss":  hist["val_loss"][be - 1],
            "val_acc":   hist["val_acc"][be - 1],
            "val_error": 1 - hist["val_acc"][be - 1],
            "sens":      hist["val_sens"][be - 1],
            "spec":      hist["val_spec"][be - 1],
        }

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for bs in batch_sizes:
        h = results[bs]["history"]
        ax.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"],
                marker="o", markersize=3, label=f"BS={bs}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Loss")
    ax.set_title("B2: Validation Loss vs Epoch (different batch sizes)")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "b2_val_loss.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    for bs in batch_sizes:
        h = results[bs]["history"]
        ax.plot(range(1, len(h["val_acc"]) + 1), h["val_acc"],
                marker="o", markersize=3, label=f"BS={bs}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
    ax.set_title("B2: Validation Accuracy vs Epoch (different batch sizes)")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "b2_val_acc.png"), dpi=150)
    plt.close()

    # ---- Table ----
    print(f"\n{'BS':>6} | {'Best Ep':>7} | {'Val Loss':>9} | {'Val Err':>9} | {'Sens':>9} | {'Spec':>9}")
    print("-" * 60)
    for bs in batch_sizes:
        r = results[bs]
        print(f"{bs:>6d} | {r['best_epoch']:>7d} | {r['val_loss']:>9.4f} | "
              f"{r['val_error']:>9.4f} | {r['sens']:>9.4f} | {r['spec']:>9.4f}")

    best_bs = min(batch_sizes, key=lambda bs: results[bs]["val_error"])
    print(f"\nSelected BS = {best_bs}  "
          f"(val classification error = {results[best_bs]['val_error']:.4f})")

    # ---- Test ----
    best_model = results[best_bs]["model"]
    test_loader = DataLoader(test_ds, batch_size=best_bs, shuffle=False)
    _, tp, tl = evaluate_model(best_model, test_loader, criterion)
    acc, sens, spec = compute_metrics(tp, tl)
    print(f"\nB2 Test  —  Accuracy: {acc:.4f}  Sensitivity: {sens:.4f}  Specificity: {spec:.4f}")

    return best_bs, best_model, acc, sens, spec


# ===================================================================
# B3 — ResNet-like + COMPARISON TABLE
# ===================================================================

def run_b3(train_ds, val_ds, test_ds, best_lr, best_bs,
           b1_model, b1_metrics, b2_model, b2_metrics):
    print("\n" + "=" * 60)
    print(f"B3 — ResNet-like  (LR={best_lr}, BS={best_bs})")
    print("=" * 60)

    criterion = nn.BCEWithLogitsLoss()
    set_seed(SEED)
    model = ResNetLike().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=best_lr)

    tl = DataLoader(train_ds, batch_size=best_bs, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=best_bs, shuffle=False)
    model, hist = train_model(model, tl, vl, criterion, opt)

    test_loader = DataLoader(test_ds, batch_size=best_bs, shuffle=False)
    _, tp, tlab = evaluate_model(model, test_loader, criterion)
    acc, sens, spec = compute_metrics(tp, tlab)
    print(f"\nB3 Test  —  Accuracy: {acc:.4f}  Sensitivity: {sens:.4f}  Specificity: {spec:.4f}")

    # ---- Comparison table ----
    rows = [
        ("B1 AlexNet", b1_model, b1_metrics),
        ("B2 VGG",     b2_model, b2_metrics),
        ("B3 ResNet",  model,    (acc, sens, spec)),
    ]
    print(f"\n{'Model':>12} | {'Acc':>8} | {'Sens':>8} | {'Spec':>8} | "
          f"{'Cls Err':>8} | {'Params':>10} | {'FLOPs':>14}")
    print("-" * 85)
    for name, m, (a, sn, sp) in rows:
        p = count_parameters(m)
        f = count_flops(m)
        print(f"{name:>12} | {a:>8.4f} | {sn:>8.4f} | {sp:>8.4f} | "
              f"{1-a:>8.4f} | {p:>10,} | {f:>14,}")

    return model, (acc, sens, spec)


# ===================================================================
# B4 — GRAD-CAM
# ===================================================================

def run_b4(test_ds, X_test, y_test, train_mean, train_std,
           b1_model, b2_model, b3_model):
    print("\n" + "=" * 60)
    print("B4 — Grad-CAM Visualization")
    print("=" * 60)

    from captum.attr import LayerGradCam

    set_seed(SEED)
    pneumonia_idx = np.where(y_test.numpy() == 1)[0]
    selected = np.random.choice(pneumonia_idx, size=5, replace=False)
    print(f"Selected pneumonia indices: {selected}")

    target_layers = {
        "B1 AlexNet": b1_model.conv3,
        "B2 VGG":     b2_model.features[17],  # last Conv2d in the Sequential
        "B3 ResNet":  b3_model.block3.conv2,
    }
    models_dict = {
        "B1 AlexNet": b1_model,
        "B2 VGG":     b2_model,
        "B3 ResNet":  b3_model,
    }

    fig, axes = plt.subplots(5, 4, figsize=(14, 18))
    col_titles = ["Original", "B1 AlexNet", "B2 VGG", "B3 ResNet"]

    for i, idx in enumerate(selected):
        img_tensor = X_test[idx].unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad = True

        orig = X_test[idx, 0].numpy() * train_std + train_mean
        orig = np.clip(orig, 0, 1)

        axes[i, 0].imshow(orig, cmap="gray")
        if i == 0:
            axes[i, 0].set_title(col_titles[0])
        axes[i, 0].axis("off")

        for j, name in enumerate(models_dict):
            m = models_dict[name]
            m.eval()
            layer = target_layers[name]
            gc = LayerGradCam(m, layer)
            attr = gc.attribute(img_tensor, target=0)
            heatmap = attr.squeeze().cpu().detach().numpy()
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()

            hm_t = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            hm_up = F.interpolate(hm_t, size=(64, 64), mode="bilinear", align_corners=False)
            hm_up = hm_up.squeeze().numpy()

            axes[i, j + 1].imshow(orig, cmap="gray")
            axes[i, j + 1].imshow(hm_up, cmap="jet", alpha=0.4)
            if i == 0:
                axes[i, j + 1].set_title(col_titles[j + 1])
            axes[i, j + 1].axis("off")

    plt.suptitle("Grad-CAM Heatmaps on Pneumonia X-rays", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "b4_gradcam.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {os.path.join(OUT_DIR, 'b4_gradcam.png')}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, X_test, y_test, train_mean, train_std = load_data()

    # Print model summaries
    print(f"\nAlexNet-like  params: {count_parameters(AlexNetLike()):,}  "
          f"FLOPs: {count_flops(AlexNetLike()):,}")
    print(f"VGG-like      params: {count_parameters(VGGLike()):,}  "
          f"FLOPs: {count_flops(VGGLike()):,}")
    print(f"ResNet-like   params: {count_parameters(ResNetLike()):,}  "
          f"FLOPs: {count_flops(ResNetLike()):,}")

    # B1
    best_lr, b1_model, b1_acc, b1_sens, b1_spec = run_b1(train_ds, val_ds, test_ds)

    # B2
    best_bs, b2_model, b2_acc, b2_sens, b2_spec = run_b2(train_ds, val_ds, test_ds, best_lr)

    # B3
    b3_model, (b3_acc, b3_sens, b3_spec) = run_b3(
        train_ds, val_ds, test_ds, best_lr, best_bs,
        b1_model, (b1_acc, b1_sens, b1_spec),
        b2_model, (b2_acc, b2_sens, b2_spec),
    )

    # B4
    run_b4(test_ds, X_test, y_test, train_mean, train_std,
           b1_model, b2_model, b3_model)

    print("\nDone. All plots saved in", OUT_DIR)


if __name__ == "__main__":
    main()

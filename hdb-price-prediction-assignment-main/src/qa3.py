"""
Question A3: Wide-and-Deep Architecture
========================================
- Both branches receive the same concatenated embeddings + continuous features.
- Linear branch: single linear layer with one output.
- MLP branch: same architecture as A1 baseline.
- output = MLP output + lambda * linear output.
- Find optimal lambda in {0, 0.25, 0.5, 1, 2} using train/val split.
- Evaluate on 5 seeds with combined train+val. Report mean/std RMSE and R^2.

Run from the repository root:
    uv run python src/A3_wide_and_deep.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import Config
from data import prepare_data
from model import PriceModel
from utils import (
    calculate_rmse,
    calculate_r2,
    prepare_combined_data,
    train_with_history,
    set_seed,
    count_parameters,
    load_best_config,
    make_config_from_a1,
    DATA_PATH,
    SEEDS,
)

OUTPUT_DIR = Path("outputs/A3")
LAMBDA_VALUES = [0, 0.25, 0.5, 1, 2]


# ── Wide-and-Deep model ──────────────────────────────────────────────────────

class WideAndDeepModel(nn.Module):
    """Wide-and-deep: MLP branch + lambda * linear branch."""

    def __init__(self, cardinalities, target_mean, target_std, config, lam=1.0):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(cardinalities, config.embedding_dims)
        )

        input_width = sum(config.embedding_dims) + len(config.continuous_features)

        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.ReLU(),
            nn.Linear(config.hidden_width, 1),
        )

        self.linear = nn.Linear(input_width, 1)
        self.lam = lam
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, categorical, continuous):
        embedded = [
            emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        features = torch.cat(embedded + [continuous], dim=1)
        mlp_out = self.mlp(features).squeeze(1)
        linear_out = self.linear(features).squeeze(1)
        return mlp_out + self.lam * linear_out

    def predict_price(self, categorical, continuous):
        return self(categorical, continuous) * self.target_std + self.target_mean


# ── Lambda search (train/val split) ──────────────────────────────────────────

def search_lambda(config):
    """Search for optimal lambda using train/validation split."""
    print("\n--- Lambda Search (train/val split) ---")
    print(f"{'lambda':>8} {'best_epoch':>12} {'val_RMSE':>12}")
    print("-" * 40)

    best_lam, best_rmse = None, float("inf")
    results = {}

    for lam in LAMBDA_VALUES:
        set_seed(config.seed)
        data = prepare_data(DATA_PATH, config)

        model = WideAndDeepModel(
            data["cardinalities"], data["target_mean"], data["target_std"],
            config, lam=lam,
        )

        ep, train_hist, val_hist = train_with_history(
            model, data["train"], data["validation"], config
        )

        val_rmse = calculate_rmse(model, data["validation"], config.batch_size)
        results[lam] = {
            "best_epoch": ep,
            "val_rmse": val_rmse,
            "train_hist": train_hist,
            "val_hist": val_hist,
        }

        tag = ""
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_lam = lam
            tag = " <-- best"

        print(f"{lam:>8.2f} {ep:>12} {val_rmse:>12,.0f}{tag}")

    print(f"\nOptimal lambda: {best_lam}")
    return best_lam, results


def plot_lambda_search(lam_results, output_dir):
    """Plot val RMSE curves for each lambda value."""
    plt.figure(figsize=(8, 5))
    for lam, res in lam_results.items():
        epochs = range(1, len(res["val_hist"]) + 1)
        plt.plot(epochs, res["val_hist"], "o-", label=f"lambda={lam}", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE (SGD)")
    plt.title("A3: Lambda Search — Validation RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "a3_lambda_search.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Multi-seed evaluation ────────────────────────────────────────────────────

def evaluate_seeds(config, best_lam, seeds):
    """Train wide-and-deep on combined data with multiple seeds."""
    print(f"\n--- Multi-seed evaluation (lambda={best_lam}) ---")
    rmses, r2s = [], []
    all_train_hist, all_test_hist = [], []

    for seed in seeds:
        set_seed(seed)
        data = prepare_combined_data(DATA_PATH, config)

        model = WideAndDeepModel(
            data["cardinalities"], data["target_mean"], data["target_std"],
            config, lam=best_lam,
        )

        _, train_hist, test_hist = train_with_history(
            model, data["train"], data["test"], config
        )

        rmse = calculate_rmse(model, data["test"], config.batch_size)
        r2 = calculate_r2(model, data["test"], config.batch_size)
        rmses.append(rmse)
        r2s.append(r2)
        all_train_hist.append(train_hist)
        all_test_hist.append(test_hist)

        print(f"  seed={seed}: RMSE={rmse:,.2f}, R^2={r2:.4f}")

    return {
        "rmses": rmses,
        "r2s": r2s,
        "mean_rmse": np.mean(rmses),
        "std_rmse": np.std(rmses, ddof=1),
        "mean_r2": np.mean(r2s),
        "std_r2": np.std(r2s, ddof=1),
        "train_histories": all_train_hist,
        "test_histories": all_test_hist,
    }


def evaluate_baseline_seeds(config, seeds):
    """Train baseline on combined data with same seeds for comparison."""
    print("\n--- Baseline multi-seed evaluation ---")
    rmses, r2s = [], []

    for seed in seeds:
        set_seed(seed)
        data = prepare_combined_data(DATA_PATH, config)

        model = PriceModel(
            data["cardinalities"], data["target_mean"], data["target_std"], config
        )

        _, _, _ = train_with_history(model, data["train"], data["test"], config)

        rmse = calculate_rmse(model, data["test"], config.batch_size)
        r2 = calculate_r2(model, data["test"], config.batch_size)
        rmses.append(rmse)
        r2s.append(r2)
        print(f"  seed={seed}: RMSE={rmse:,.2f}, R^2={r2:.4f}")

    return {
        "mean_rmse": np.mean(rmses),
        "std_rmse": np.std(rmses, ddof=1),
        "mean_r2": np.mean(r2s),
        "std_r2": np.std(r2s, ddof=1),
    }


def plot_comparison(wd_res, bl_res, best_lam, output_dir):
    """Bar chart comparing wide-and-deep vs baseline."""
    labels = ["Baseline", f"Wide-and-Deep (lambda={best_lam})"]
    mean_rmse = [bl_res["mean_rmse"], wd_res["mean_rmse"]]
    std_rmse = [bl_res["std_rmse"], wd_res["std_rmse"]]
    mean_r2 = [bl_res["mean_r2"], wd_res["mean_r2"]]
    std_r2 = [bl_res["std_r2"], wd_res["std_r2"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels))

    axes[0].bar(x, mean_rmse, yerr=std_rmse, capsize=5, color=["#4C72B0", "#C44E52"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Test RMSE (SGD)")
    axes[0].set_title("Test RMSE (mean +/- std, 5 seeds)")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, mean_r2, yerr=std_r2, capsize=5, color=["#4C72B0", "#C44E52"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Test R^2")
    axes[1].set_title("Test R^2 (mean +/- std, 5 seeds)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "a3_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_wd_rmse_curves(wd_res, best_lam, output_dir):
    """Plot average train/test RMSE curves for wide-and-deep."""
    max_len = max(len(h) for h in wd_res["train_histories"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for hist_list, label in [
        (wd_res["train_histories"], "Train RMSE"),
        (wd_res["test_histories"], "Test RMSE"),
    ]:
        padded = [h + [h[-1]] * (max_len - len(h)) for h in hist_list]
        mean_curve = np.mean(padded, axis=0)
        std_curve = np.std(padded, axis=0)
        epochs = range(1, max_len + 1)
        ax.plot(epochs, mean_curve, "o-", label=label, markersize=4)
        ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (SGD)")
    ax.set_title(f"A3: Wide-and-Deep (lambda={best_lam}) RMSE Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "a3_wd_rmse_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    a1_cfg = load_best_config()
    config = make_config_from_a1(a1_cfg)

    print("=" * 60)
    print("A3: Wide-and-Deep Architecture")
    print("=" * 60)
    print(f"Using A1 config: hidden_width={a1_cfg['hidden_width']}, "
          f"embedding_dim={a1_cfg['embedding_dim']}")

    # Print parameter count
    set_seed(42)
    data = prepare_data(DATA_PATH, config)
    baseline = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )
    wd = WideAndDeepModel(
        data["cardinalities"], data["target_mean"], data["target_std"],
        config, lam=1.0,
    )
    print(f"Baseline params:        {count_parameters(baseline):,}")
    print(f"Wide-and-Deep params:   {count_parameters(wd):,}")

    # 1. Search lambda
    best_lam, lam_results = search_lambda(config)
    plot_lambda_search(lam_results, OUTPUT_DIR)

    # Save best lambda for A4
    import json
    with open(OUTPUT_DIR / "best_lambda.json", "w") as f:
        json.dump({"best_lambda": best_lam}, f)

    # 2. Multi-seed evaluation
    wd_res = evaluate_seeds(config, best_lam, SEEDS)
    bl_res = evaluate_baseline_seeds(config, SEEDS)

    # 3. Summary
    print("\n" + "=" * 70)
    print("A3 Results Summary")
    print("=" * 70)
    print(f"{'Model':<30} {'RMSE (mean +/- std)':>22} {'R^2 (mean +/- std)':>22}")
    print("-" * 80)
    print(
        f"{'Baseline':<30} "
        f"{bl_res['mean_rmse']:>10,.2f} +/- {bl_res['std_rmse']:<8,.2f} "
        f"{bl_res['mean_r2']:>10.4f} +/- {bl_res['std_r2']:<8.4f}"
    )
    print(
        f"{'Wide-and-Deep (lam=' + str(best_lam) + ')':<30} "
        f"{wd_res['mean_rmse']:>10,.2f} +/- {wd_res['std_rmse']:<8,.2f} "
        f"{wd_res['mean_r2']:>10.4f} +/- {wd_res['std_r2']:<8.4f}"
    )

    # 4. Plots
    plot_comparison(wd_res, bl_res, best_lam, OUTPUT_DIR)
    plot_wd_rmse_curves(wd_res, best_lam, OUTPUT_DIR)


if __name__ == "__main__":
    main()

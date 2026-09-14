"""
Question A4: Feature Ablation (Captum)
======================================
- Use Captum's FeatureAblation on the test set for:
  (1) Final baseline model from A1
  (2) Wide-and-deep model from A3
- Train both models with the same seed.
- Compute mean absolute attribution per feature across all test samples.
- Present feature importance in SGD in a bar plot (highest to lowest).
- Report top 3 features for each model.

Run from the repository root:
    uv run python src/A4_feature_ablation.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from captum.attr import FeatureAblation

from config import Config
from model import PriceModel
from qa3 import WideAndDeepModel
from utils import (
    prepare_combined_data,
    train_with_history,
    set_seed,
    load_best_config,
    make_config_from_a1,
    DATA_PATH,
)

OUTPUT_DIR = Path("outputs/A4")

FEATURE_NAMES = [
    "month",
    "town",
    "flat_model_type",
    "storey_range",
    "dist_to_nearest_stn",
    "dist_to_dhoby",
    "degree_centrality",
    "eigenvector_centrality",
    "remaining_lease_years",
    "floor_area_sqm",
]

SEED = 42


# ── Train models ──────────────────────────────────────────────────────────────

def train_baseline(config, data):
    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )
    train_with_history(model, data["train"], data["test"], config)
    return model


def train_wide_and_deep(config, data, best_lam):
    model = WideAndDeepModel(
        data["cardinalities"], data["target_mean"], data["target_std"],
        config, lam=best_lam,
    )
    train_with_history(model, data["train"], data["test"], config)
    return model


# ── Feature ablation ──────────────────────────────────────────────────────────

def compute_feature_importance(model, test_dataset):
    """Compute mean absolute FeatureAblation attribution for each of the 10 features."""
    model.eval()

    cat_data = test_dataset.tensors[0]   # (N, 4) long
    cont_data = test_dataset.tensors[1]  # (N, 6) float

    cat_mask = torch.tensor([0, 1, 2, 3]).unsqueeze(0).expand_as(cat_data)
    cont_mask = torch.tensor([4, 5, 6, 7, 8, 9]).unsqueeze(0).expand_as(cont_data)

    cat_baseline = torch.zeros_like(cat_data)
    cont_baseline = torch.zeros_like(cont_data)

    def forward_fn(cat, cont):
        return model.predict_price(cat, cont).unsqueeze(1)

    fa = FeatureAblation(forward_fn)

    attr_cat, attr_cont = fa.attribute(
        inputs=(cat_data, cont_data),
        baselines=(cat_baseline, cont_baseline),
        feature_mask=(cat_mask, cont_mask),
        perturbations_per_eval=512,
        show_progress=True,
    )

    importances = torch.zeros(10)
    for i in range(4):
        importances[i] = attr_cat[:, i].abs().mean().item()
    for i in range(6):
        importances[4 + i] = attr_cont[:, i].abs().mean().item()

    return importances.numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_feature_importance(importances, model_name, output_dir):
    """Bar plot of feature importances sorted highest to lowest."""
    sorted_idx = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
    sorted_vals = importances[sorted_idx]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(sorted_names)), sorted_vals, color="#4C72B0")
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
    plt.ylabel("Mean |Attribution| (SGD)")
    plt.title(f"A4: Feature Importance — {model_name}")
    plt.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, sorted_vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + sorted_vals.max() * 0.01,
            f"${val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("-", "_")
    path = output_dir / f"a4_importance_{fname}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    return sorted_idx


def plot_side_by_side(bl_imp, wd_imp, output_dir):
    """Side-by-side bar plots for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, imp, title, color in [
        (axes[0], bl_imp, "Baseline", "#4C72B0"),
        (axes[1], wd_imp, "Wide-and-Deep", "#C44E52"),
    ]:
        sorted_idx = np.argsort(imp)[::-1]
        sorted_names = [FEATURE_NAMES[i] for i in sorted_idx]
        sorted_vals = imp[sorted_idx]

        bars = ax.bar(range(len(sorted_names)), sorted_vals, color=color)
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha="right")
        ax.set_ylabel("Mean |Attribution| (SGD)")
        ax.set_title(f"{title}")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, sorted_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sorted_vals.max() * 0.01,
                f"${val:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.suptitle("A4: Feature Importance Comparison", fontsize=14)
    plt.tight_layout()
    path = output_dir / "a4_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    a1_cfg = load_best_config()
    config = make_config_from_a1(a1_cfg)

    print("=" * 60)
    print("A4: Feature Ablation (Captum)")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print(f"Config: hidden_width={a1_cfg['hidden_width']}, "
          f"embedding_dim={a1_cfg['embedding_dim']}")

    # Prepare data
    set_seed(SEED)
    data = prepare_combined_data(DATA_PATH, config)

    # Train baseline
    print("\nTraining baseline model...")
    set_seed(SEED)
    baseline = train_baseline(config, data)

    # Train wide-and-deep (use lambda=1.0 as default; update after running A3)
    # Load the optimal lambda from A3 if available, otherwise default to 1.0
    import json
    a3_lam = 1.0
    try:
        with open("outputs/A3/best_lambda.json") as f:
            a3_lam = json.load(f)["best_lambda"]
        print(f"Loaded optimal lambda={a3_lam} from A3")
    except FileNotFoundError:
        print(f"A3 results not found, using default lambda={a3_lam}")
        print("(Re-run after A3 for optimal lambda)")

    print(f"\nTraining wide-and-deep model (lambda={a3_lam})...")
    set_seed(SEED)
    wd_model = train_wide_and_deep(config, data, a3_lam)

    # Compute feature importance
    print("\nComputing feature importance for baseline...")
    bl_imp = compute_feature_importance(baseline, data["test"])

    print("\nComputing feature importance for wide-and-deep...")
    wd_imp = compute_feature_importance(wd_model, data["test"])

    # Report
    print("\n" + "=" * 70)
    print("Feature Importance (Mean |Attribution| in SGD)")
    print("=" * 70)
    print(f"{'Feature':<25} {'Baseline':>12} {'Wide-and-Deep':>14}")
    print("-" * 55)
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{name:<25} {bl_imp[i]:>12,.0f} {wd_imp[i]:>14,.0f}")

    # Top 3
    bl_top3 = np.argsort(bl_imp)[::-1][:3]
    wd_top3 = np.argsort(wd_imp)[::-1][:3]
    print(f"\nBaseline top 3:       {[FEATURE_NAMES[i] for i in bl_top3]}")
    print(f"Wide-and-Deep top 3:  {[FEATURE_NAMES[i] for i in wd_top3]}")

    # Plots
    plot_feature_importance(bl_imp, "Baseline", OUTPUT_DIR)
    plot_feature_importance(wd_imp, "Wide-and-Deep", OUTPUT_DIR)
    plot_side_by_side(bl_imp, wd_imp, OUTPUT_DIR)


if __name__ == "__main__":
    main()

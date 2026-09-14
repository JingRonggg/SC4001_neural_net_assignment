"""
Question A2: Ablation Experiments
=================================
a) Remove all categorical embeddings — use only the 6 continuous features.
b) Replace hidden ReLU activation with Sigmoid.

Train baseline + both ablations on combined train+val with 5 seeds.
Report mean and std of test RMSE and R^2.

Run from the repository root:
    uv run python src/qa2.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import Config
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

OUTPUT_DIR = Path("outputs/A2")


# ── Ablation model variants ──────────────────────────────────────────────────

class PriceModelNoCat(nn.Module):
    """Ablation (a): no categorical embeddings, continuous features only."""

    def __init__(self, target_mean, target_std, config):
        super().__init__()
        input_width = len(config.continuous_features)
        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.ReLU(),
            nn.Linear(config.hidden_width, 1),
        )
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, categorical, continuous):
        return self.mlp(continuous).squeeze(1)

    def predict_price(self, categorical, continuous):
        return self(categorical, continuous) * self.target_std + self.target_mean


class PriceModelSigmoid(nn.Module):
    """Ablation (b): Sigmoid activation instead of ReLU."""

    def __init__(self, cardinalities, target_mean, target_std, config):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(cardinalities, config.embedding_dims)
        )
        input_width = sum(config.embedding_dims) + len(config.continuous_features)
        self.mlp = nn.Sequential(
            nn.Linear(input_width, config.hidden_width),
            nn.LayerNorm(config.hidden_width),
            nn.Sigmoid(),
            nn.Linear(config.hidden_width, 1),
        )
        self.target_mean = target_mean
        self.target_std = target_std

    def forward(self, categorical, continuous):
        embedded = [
            emb(categorical[:, i]) for i, emb in enumerate(self.embeddings)
        ]
        features = torch.cat(embedded + [continuous], dim=1)
        return self.mlp(features).squeeze(1)

    def predict_price(self, categorical, continuous):
        return self(categorical, continuous) * self.target_std + self.target_mean


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(model_factory, config, seeds, label):
    """Train a model across multiple seeds and return test metrics."""
    rmses, r2s = [], []
    all_train_hist, all_test_hist = [], []

    for seed in seeds:
        set_seed(seed)
        data = prepare_combined_data(DATA_PATH, config)
        model = model_factory(data, config)

        _, train_hist, test_hist = train_with_history(
            model, data["train"], data["test"], config
        )

        rmse = calculate_rmse(model, data["test"], config.batch_size)
        r2 = calculate_r2(model, data["test"], config.batch_size)
        rmses.append(rmse)
        r2s.append(r2)
        all_train_hist.append(train_hist)
        all_test_hist.append(test_hist)

        print(f"  {label} seed={seed}: RMSE={rmse:,.2f}, R^2={r2:.4f}")

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


# ── Model factories ───────────────────────────────────────────────────────────

def make_baseline(data, config):
    return PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )


def make_no_cat(data, config):
    return PriceModelNoCat(data["target_mean"], data["target_std"], config)


def make_sigmoid(data, config):
    return PriceModelSigmoid(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_rmse_curves(results_dict, output_dir):
    """Plot RMSE curves for all variants (average across seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, res in results_dict.items():
        max_len = max(len(h) for h in res["train_histories"])
        for hist_list, ax, title in [
            (res["train_histories"], axes[0], "Train RMSE"),
            (res["test_histories"], axes[1], "Test RMSE"),
        ]:
            padded = []
            for h in hist_list:
                padded.append(h + [h[-1]] * (max_len - len(h)))
            mean_curve = np.mean(padded, axis=0)
            std_curve = np.std(padded, axis=0)
            epochs = range(1, max_len + 1)
            ax.plot(epochs, mean_curve, "o-", label=label, markersize=4)
            ax.fill_between(
                epochs,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.15,
            )

    for ax, title in zip(axes, ["Train RMSE", "Test RMSE"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE (SGD)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "a2_rmse_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_bar_comparison(results_dict, output_dir):
    """Bar chart comparing RMSE and R^2 across variants."""
    labels = list(results_dict.keys())
    mean_rmse = [results_dict[l]["mean_rmse"] for l in labels]
    std_rmse = [results_dict[l]["std_rmse"] for l in labels]
    mean_r2 = [results_dict[l]["mean_r2"] for l in labels]
    std_r2 = [results_dict[l]["std_r2"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    axes[0].bar(x, mean_rmse, yerr=std_rmse, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Test RMSE (SGD)")
    axes[0].set_title("Test RMSE (mean +/- std, 5 seeds)")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, mean_r2, yerr=std_r2, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Test R^2")
    axes[1].set_title("Test R^2 (mean +/- std, 5 seeds)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = output_dir / "a2_bar_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    a1_cfg = load_best_config()
    config = make_config_from_a1(a1_cfg)

    print("=" * 60)
    print("A2: Ablation Experiments")
    print("=" * 60)
    print(f"Using A1 config: hidden_width={a1_cfg['hidden_width']}, "
          f"embedding_dim={a1_cfg['embedding_dim']}")
    print(f"Seeds: {SEEDS}")

    # Print parameter counts
    set_seed(42)
    data = prepare_combined_data(DATA_PATH, config)
    print(f"\nBaseline params:     {count_parameters(make_baseline(data, config)):,}")
    print(f"No-cat params:       {count_parameters(make_no_cat(data, config)):,}")
    print(f"Sigmoid params:      {count_parameters(make_sigmoid(data, config)):,}")

    # Run experiments
    print("\n--- Baseline ---")
    baseline_res = run_experiment(make_baseline, config, SEEDS, "Baseline")

    print("\n--- Ablation (a): No Categorical Embeddings ---")
    nocat_res = run_experiment(make_no_cat, config, SEEDS, "NoCat")

    print("\n--- Ablation (b): Sigmoid Activation ---")
    sigmoid_res = run_experiment(make_sigmoid, config, SEEDS, "Sigmoid")

    results = {
        "Baseline": baseline_res,
        "No Embeddings": nocat_res,
        "Sigmoid": sigmoid_res,
    }

    # Summary table
    print("\n" + "=" * 70)
    print("A2 Results Summary")
    print("=" * 70)
    print(f"{'Model':<20} {'RMSE (mean +/- std)':>22} {'R^2 (mean +/- std)':>22}")
    print("-" * 70)
    for label, res in results.items():
        print(
            f"{label:<20} "
            f"{res['mean_rmse']:>10,.2f} +/- {res['std_rmse']:<8,.2f} "
            f"{res['mean_r2']:>10.4f} +/- {res['std_r2']:<8.4f}"
        )

    # Plots
    plot_rmse_curves(results, OUTPUT_DIR)
    plot_bar_comparison(results, OUTPUT_DIR)


if __name__ == "__main__":
    main()

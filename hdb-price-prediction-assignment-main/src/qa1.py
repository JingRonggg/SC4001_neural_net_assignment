"""
Question A1: Hyperparameter Optimisation via Grid Search (Ray Tune)
===================================================================
- Report baseline trainable parameters.
- Grid search: hidden_width in {16, 32, 64, 128, 256},
               embedding_dim in {2, 4, 8, 12, 16} (same for all 4 categoricals).
- Select config with lowest best validation RMSE.
- Plot train/val RMSE curves for best config.
- Retrain on combined train+val, evaluate on test. Plot train/test RMSE.
- Report final RMSE and R^2.

Run from the repository root:
    uv run python src/qa1.py
"""

import copy
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ray import tune

from config import Config
from data import prepare_data
from model import PriceModel
from utils import (
    count_parameters,
    calculate_rmse,
    calculate_r2,
    prepare_combined_data,
    train_with_history,
    set_seed,
    save_best_config,
    DATA_PATH,
)

OUTPUT_DIR = Path("outputs/A1")

HIDDEN_WIDTHS = [16, 32, 64, 128, 256]
EMBEDDING_DIMS = [2, 4, 8, 12, 16]


# ── Ray Tune trainable ────────────────────────────────────────────────────────

def ray_trainable(ray_config):
    """Training function for each Ray Tune trial.
    Returns a dict with the best validation RMSE and the epoch it occurred at.
    Using return-based reporting (no session.report / train.report needed).
    """
    csv_path = ray_config["csv_path"]
    config = Config(
        hidden_width=ray_config["hidden_width"],
        embedding_dims=tuple([ray_config["embedding_dim"]] * 4),
    )
    set_seed(config.seed)
    data = prepare_data(csv_path, config)

    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )

    loader = DataLoader(data["train"], config.batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val = float("inf")
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for cat, cont, tgt in loader:
            opt.zero_grad()
            sp = model(cat, cont)
            st = (tgt - model.target_mean) / model.target_std
            loss = loss_fn(sp, st)
            loss.backward()
            opt.step()

        val_rmse = calculate_rmse(model, data["validation"], config.batch_size)

        if val_rmse < best_val - config.early_stopping_threshold:
            best_val = val_rmse
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.early_stopping_patience:
                break

    return {"best_val_rmse": best_val, "best_epoch": best_epoch}


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid_search():
    csv_path = os.path.abspath(DATA_PATH)

    search_space = {
        "hidden_width": tune.grid_search(HIDDEN_WIDTHS),
        "embedding_dim": tune.grid_search(EMBEDDING_DIMS),
        "csv_path": csv_path,
    }

    tuner = tune.Tuner(
        ray_trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="best_val_rmse",
            mode="min",
        ),
    )

    results = tuner.fit()
    return results


def analyse_grid_results(results):
    """Print a results table, find best config, return (best_hw, best_ed, best_val_rmse)."""
    print("\n" + "=" * 70)
    print("Grid Search Results (sorted by best validation RMSE)")
    print("=" * 70)
    print(f"{'hidden_width':>14} {'embedding_dim':>14} {'best_val_RMSE':>14} {'best_epoch':>12}")
    print("-" * 70)

    rows = []
    for result in results:
        hw = result.config["hidden_width"]
        ed = result.config["embedding_dim"]
        bv = result.metrics["best_val_rmse"]
        be = result.metrics["best_epoch"]
        rows.append((hw, ed, bv, be))

    rows.sort(key=lambda r: r[2])
    for hw, ed, bv, be in rows:
        tag = " <-- best" if (hw, ed, bv) == (rows[0][0], rows[0][1], rows[0][2]) else ""
        print(f"{hw:>14} {ed:>14} {bv:>14,.0f} {be:>12}{tag}")

    best = results.get_best_result("best_val_rmse", "min")
    best_hw = best.config["hidden_width"]
    best_ed = best.config["embedding_dim"]
    best_val = best.metrics["best_val_rmse"]
    print(f"\nOptimal: hidden_width={best_hw}, embedding_dim={best_ed}")
    print(f"Best validation RMSE: {best_val:,.0f}")

    return best_hw, best_ed, best_val


# ── Retrain best config locally (for RMSE curves) ────────────────────────────

def retrain_best_with_val(best_hw, best_ed, output_dir):
    """Retrain the best config on train/val split to get RMSE curves."""
    config = Config(
        hidden_width=best_hw,
        embedding_dims=tuple([best_ed] * 4),
    )
    set_seed(config.seed)
    data = prepare_data(DATA_PATH, config)

    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )

    best_epoch, train_hist, val_hist = train_with_history(
        model, data["train"], data["validation"], config
    )

    epochs = range(1, len(train_hist) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_hist, "o-", label="Train RMSE", markersize=4)
    plt.plot(epochs, val_hist, "s-", label="Validation RMSE", markersize=4)
    plt.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5,
                label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (SGD)")
    plt.title(f"A1 Best Config: hidden_width={best_hw}, embedding_dim={best_ed}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "a1_train_val_rmse.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    return best_epoch


# ── Retrain on combined data ──────────────────────────────────────────────────

def retrain_and_evaluate(best_hw, best_ed, output_dir):
    """Retrain the best config on train+val, evaluate on test."""
    config = Config(
        hidden_width=best_hw,
        embedding_dims=tuple([best_ed] * 4),
    )
    set_seed(config.seed)
    data = prepare_combined_data(DATA_PATH, config)

    model = PriceModel(
        data["cardinalities"], data["target_mean"], data["target_std"], config
    )
    print(f"\nSelected model trainable parameters: {count_parameters(model):,}")

    best_epoch, train_hist, test_hist = train_with_history(
        model, data["train"], data["test"], config
    )

    test_rmse = calculate_rmse(model, data["test"], config.batch_size)
    test_r2 = calculate_r2(model, data["test"], config.batch_size)

    print(f"\n--- Retrained on combined train+val ---")
    print(f"Best epoch: {best_epoch}")
    print(f"Test RMSE:  {test_rmse:,.2f} SGD")
    print(f"Test R^2:   {test_r2:.4f}")

    epochs = range(1, len(train_hist) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_hist, "o-", label="Train RMSE", markersize=4)
    plt.plot(epochs, test_hist, "s-", label="Test RMSE", markersize=4)
    plt.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5,
                label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (SGD)")
    plt.title(f"A1 Retrained: hidden_width={best_hw}, embedding_dim={best_ed}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "a1_train_test_rmse.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")

    return best_epoch, test_rmse, test_r2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Report baseline parameters
    print("=" * 60)
    print("A1: Baseline Model Info")
    print("=" * 60)

    base_config = Config()
    set_seed(base_config.seed)
    base_data = prepare_data(DATA_PATH, base_config)

    baseline = PriceModel(
        base_data["cardinalities"],
        base_data["target_mean"],
        base_data["target_std"],
        base_config,
    )
    print(f"Default baseline trainable parameters: {count_parameters(baseline):,}")
    print(f"Cardinalities (incl. unseen): {base_data['cardinalities']}")
    print(f"Embedding dims: {base_config.embedding_dims}")
    print(f"Input width: {sum(base_config.embedding_dims) + len(base_config.continuous_features)}")
    print(f"Hidden width: {base_config.hidden_width}")
    print(f"\nModel architecture:\n{baseline}")

    # 2. Grid search
    print("\n" + "=" * 60)
    print("A1: Running Ray Tune Grid Search")
    print("=" * 60)

    results = run_grid_search()
    best_hw, best_ed, _ = analyse_grid_results(results)

    # 3. Retrain best config on train/val split locally (for RMSE curves)
    print("\n" + "=" * 60)
    print("A1: Retraining Best Config (train/val split, for curves)")
    print("=" * 60)

    retrain_best_with_val(best_hw, best_ed, OUTPUT_DIR)

    # 4. Report selected model parameters
    sel_config = Config(
        hidden_width=best_hw,
        embedding_dims=tuple([best_ed] * 4),
    )
    set_seed(sel_config.seed)
    sel_data = prepare_data(DATA_PATH, sel_config)
    sel_model = PriceModel(
        sel_data["cardinalities"], sel_data["target_mean"],
        sel_data["target_std"], sel_config,
    )
    print(f"\nSelected model trainable parameters: {count_parameters(sel_model):,}")

    # 5. Retrain on combined train+val, evaluate on test
    print("\n" + "=" * 60)
    print("A1: Retrain on Combined Data")
    print("=" * 60)

    best_epoch, test_rmse, test_r2 = retrain_and_evaluate(
        best_hw, best_ed, OUTPUT_DIR
    )

    # 6. Save config for A2/A3/A4
    save_best_config(OUTPUT_DIR, best_hw, best_ed, best_epoch)

    print("\n" + "=" * 60)
    print("A1 Summary")
    print("=" * 60)
    print(f"Optimal hidden_width:  {best_hw}")
    print(f"Optimal embedding_dim: {best_ed}")
    print(f"Best epoch (retrain):  {best_epoch}")
    print(f"Final test RMSE:       {test_rmse:,.2f} SGD")
    print(f"Final test R^2:        {test_r2:.4f}")


if __name__ == "__main__":
    main()

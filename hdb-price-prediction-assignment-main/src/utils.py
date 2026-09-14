"""Shared utilities for Part A experiments."""

import copy
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import Config
from data import make_dataset

DATA_PATH = "hdb_price_prediction.csv"
SEEDS = [42, 123, 456, 789, 2024]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def calculate_rmse(model, dataset, batch_size=512):
    model.eval()
    preds, tgts = [], []
    for cat, cont, tgt in DataLoader(dataset, batch_size=batch_size):
        preds.append(model.predict_price(cat, cont))
        tgts.append(tgt)
    p, t = torch.cat(preds), torch.cat(tgts)
    return (p - t).square().mean().sqrt().item()


@torch.no_grad()
def calculate_r2(model, dataset, batch_size=512):
    model.eval()
    preds, tgts = [], []
    for cat, cont, tgt in DataLoader(dataset, batch_size=batch_size):
        preds.append(model.predict_price(cat, cont))
        tgts.append(tgt)
    p, t = torch.cat(preds), torch.cat(tgts)
    ss_res = (t - p).square().sum().item()
    ss_tot = (t - t.mean()).square().sum().item()
    return 1.0 - ss_res / ss_tot


def prepare_combined_data(csv_path, config):
    """Prepare datasets with 2017-2021 as training and 2022 as test.
    Preprocessing is fitted on the combined train+validation data."""
    data = pd.read_csv(csv_path)
    rows = data.to_dict(orient="records")

    combined_years = set(config.train_years) | {config.validation_year}
    train_rows = [r for r in rows if int(r["year"]) in combined_years]
    test_rows = [r for r in rows if int(r["year"]) == config.test_year]

    category_maps = {}
    for name in config.categorical_features:
        values = sorted({r[name] for r in train_rows})
        category_maps[name] = {v: i + 1 for i, v in enumerate(values)}

    train_cont = torch.tensor(
        [[float(r[name]) for name in config.continuous_features] for r in train_rows]
    )
    means = train_cont.mean(dim=0)
    stds = train_cont.std(dim=0).clamp_min(1e-8)

    train_tgt = torch.tensor([float(r["resale_price"]) for r in train_rows])
    target_mean = train_tgt.mean().item()
    target_std = train_tgt.std(correction=0).clamp_min(1e-8).item()

    return {
        "train": make_dataset(train_rows, category_maps, means, stds, config),
        "test": make_dataset(test_rows, category_maps, means, stds, config),
        "cardinalities": [
            len(category_maps[n]) + 1 for n in config.categorical_features
        ],
        "target_mean": target_mean,
        "target_std": target_std,
    }


def train_with_history(model, train_data, eval_data, config, fixed_epochs=None):
    """Train model and return (best_epoch, train_rmse_history, eval_rmse_history).

    If fixed_epochs is None, uses early stopping on eval RMSE.
    If fixed_epochs is set, trains for exactly that many epochs (no early stopping).
    """
    loader = DataLoader(train_data, config.batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_hist, eval_hist = [], []
    best_rmse = float("inf")
    best_weights = None
    best_epoch = 0
    patience_ctr = 0
    n_epochs = fixed_epochs if fixed_epochs else config.max_epochs
    use_es = fixed_epochs is None

    for epoch in range(1, n_epochs + 1):
        model.train()
        se, n = 0.0, 0
        for cat, cont, tgt in loader:
            opt.zero_grad()
            sp = model(cat, cont)
            st = (tgt - model.target_mean) / model.target_std
            loss = loss_fn(sp, st)
            loss.backward()
            opt.step()
            with torch.no_grad():
                p = sp * model.target_std + model.target_mean
                se += (p - tgt).square().sum().item()
                n += len(tgt)

        tr = (se / n) ** 0.5
        er = calculate_rmse(model, eval_data, config.batch_size)
        train_hist.append(tr)
        eval_hist.append(er)

        if use_es:
            if best_weights is None or er < best_rmse - config.early_stopping_threshold:
                best_rmse = er
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= config.early_stopping_patience:
                    break
        else:
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_epoch, train_hist, eval_hist


def save_best_config(output_dir, hidden_width, embedding_dim, best_epoch):
    """Save best A1 config to JSON for use by A2/A3/A4."""
    cfg = {
        "hidden_width": hidden_width,
        "embedding_dim": embedding_dim,
        "best_epoch": best_epoch,
    }
    path = Path(output_dir) / "best_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved best config to {path}")
    return path


def load_best_config(a1_output_dir="outputs/A1"):
    """Load best A1 config from JSON."""
    with open(Path(a1_output_dir) / "best_config.json") as f:
        return json.load(f)


def make_config_from_a1(a1_cfg):
    """Create a Config object using the best hyperparameters from A1."""
    return Config(
        hidden_width=a1_cfg["hidden_width"],
        embedding_dims=tuple([a1_cfg["embedding_dim"]] * 4),
    )

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse_loss(pred, target, eps=1e-8):
    """
    Root Mean Squared Error loss.

    eps prevents numerical issues if MSE becomes exactly zero.
    """
    mse = torch.mean((pred - target) ** 2)

    return torch.sqrt(
        mse + eps
    )


def _epoch(
    model,
    loader,
    optimizer,
    device,
    training,
):
    model.train(training)

    total = 0.0
    count = 0

    ctx = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with ctx:
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            sector = batch["sector_id"].to(device)

            if training:
                optimizer.zero_grad()

            pred = model(
                x,
                sector,
            )

            loss = rmse_loss(
                pred,
                y,
            )

            if training:
                loss.backward()
                optimizer.step()

            total += float(loss.item()) * len(y)
            count += len(y)

    return total / max(count, 1)


def fit_model(
    model,
    train_ds,
    val_ds,
    batch_size,
    epochs,
    lr,
    weight_decay,
    patience,
    device,
):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    best_val = float("inf")
    stale = 0
    hist = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        train_loss = _epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            training=True,
        )

        val_loss = _epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            training=False,
        )

        hist.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val:
            best_val = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            stale = 0

        else:
            stale += 1

            if stale >= patience:
                break

    model.load_state_dict(
        best_state
    )

    return pd.DataFrame(
        hist
    )


@torch.no_grad()
def predict(
    model,
    dataset,
    batch_size,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    rows = []

    for batch in loader:
        x = batch["x"].to(device)

        sector = batch[
            "sector_id"
        ].to(device)

        pred = model(
            x,
            sector,
        ).cpu().numpy()

        y = batch[
            "y"
        ].numpy()

        for date, symbol, y_true, y_pred in zip(
            batch["date"],
            batch["symbol"],
            y,
            pred,
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                }
            )

    return pd.DataFrame(
        rows
    )
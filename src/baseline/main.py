import argparse
from pathlib import Path

import pandas as pd
import torch

from .common.backtest import BacktestConfig, backtest_predictions
from .common.config import Config
from .common.data import (
    FEATURE_COLUMNS,
    StockWindowDataset,
    fit_feature_scaler,
    load_market_data,
)
from .common.metrics import daily_cross_sectional_metrics, regression_metrics
from .model import VanillaLSTM
from .common.phases import build_paper_phases
from ..train import fit_model, predict, seed_everything


PAPER_LSTM = {
    "Return": 0.036,
    "IC": -0.009,
    "Rank_IC": -0.185,
    "Prec@N": 0.544,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Run only one phase, e.g. --phase 1",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Examples: cuda, cpu, cuda:0",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )

    seed_everything(cfg.seed)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    market = load_market_data(
        cfg.data_path,
        cfg.ticker_path,
    )

    phases = build_paper_phases()

    if args.phase is not None:
        phases = [
            phase
            for phase in phases
            if phase.number == args.phase
        ]

        if not phases:
            raise ValueError("--phase must be between 1 and 12")

    print(
        f"device={device} "
        f"stocks={len(market.symbols)} "
        f"sectors={len(market.sector_to_id)}"
    )

    out_dir = Path("results") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_summaries = []

    for phase in phases:
        print(f"\nPhase {phase.number}: {phase}")

        # Fit normalization ONLY on the phase training data.
        mean, std = fit_feature_scaler(
            market=market,
            symbols=market.symbols,
            start=phase.train_start,
            end=phase.train_end,
        )

        # Build datasets.
        train_ds = StockWindowDataset(
            market=market,
            symbols=market.symbols,
            start=phase.train_start,
            end=phase.train_end,
            lookback=cfg.lookback,
            lookahead=cfg.lookahead,
            mean=mean,
            std=std,
        )

        val_ds = StockWindowDataset(
            market=market,
            symbols=market.symbols,
            start=phase.val_start,
            end=phase.val_end,
            lookback=cfg.lookback,
            lookahead=cfg.lookahead,
            mean=mean,
            std=std,
        )

        test_ds = StockWindowDataset(
            market=market,
            symbols=market.symbols,
            start=phase.test_start,
            end=phase.test_end,
            lookback=cfg.lookback,
            lookahead=cfg.lookahead,
            mean=mean,
            std=std,
        )

        print(
            f"samples "
            f"train={len(train_ds):,} "
            f"val={len(val_ds):,} "
            f"test={len(test_ds):,}"
        )

        model = VanillaLSTM(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

        model.to(device)

        # Train model.
        history = fit_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            device=device,
        )

        # Generate test predictions.
        preds = predict(
            model=model,
            dataset=test_ds,
            batch_size=cfg.batch_size,
            device=device,
        )

        # ---------------------------------------------------------
        # Prediction metrics:
        # IC, ICIR, Rank_IC, Rank_ICIR, Prec@N
        #
        # Ignore the Return generated here because we replace it
        # below with the backtest result.
        # ---------------------------------------------------------
        daily_metrics, summary = daily_cross_sectional_metrics(
            predictions=preds,
            top_n=cfg.top_k,
        )

        # Regression diagnostics.
        summary.update(
            regression_metrics(preds)
        )

        # ---------------------------------------------------------
        # Portfolio backtest.
        #
        # Paper-inspired settings:
        # - top k stocks
        # - equal allocation
        # - 7% trailing stop
        # - 20% take-profit
        # - max holding window = lookahead = 5 trading days
        # ---------------------------------------------------------
        bt_cfg = BacktestConfig(
            initial_nav=1.0,
            top_k=cfg.top_k,
            trailing_stop=0.07,
            take_profit=0.20,
            max_holding_days=cfg.lookahead,
        )

        phase_return, equity = backtest_predictions(
            predictions=preds,
            market=market,
            cfg=bt_cfg,
        )

        # Replace the temporary metric Return.
        summary["Return"] = phase_return
        summary["phase"] = phase.number

        phase_summaries.append(summary)

        # Save phase outputs.
        history.to_csv(
            out_dir / f"phase_{phase.number:02d}_history.csv",
            index=False,
        )

        preds.to_csv(
            out_dir / f"phase_{phase.number:02d}_predictions.csv",
            index=False,
        )

        daily_metrics.to_csv(
            out_dir / f"phase_{phase.number:02d}_daily_metrics.csv",
            index=False,
        )

        equity.to_csv(
            out_dir / f"phase_{phase.number:02d}_equity.csv",
            index=False,
        )

        # Print phase result immediately.
        print("\nPhase metrics:")

        metric_order = [
            "Return",
            "IC",
            "ICIR",
            "Rank_IC",
            "Rank_ICIR",
            "Prec@N",
            "MAE",
            "RMSE",
        ]

        for metric in metric_order:
            if metric in summary:
                print(
                    f"{metric:12s}: "
                    f"{summary[metric]:.6f}"
                )

    # Combine all completed phases.
    summary_df = (
        pd.DataFrame(phase_summaries)
        .sort_values("phase")
        .reset_index(drop=True)
    )

    summary_df.to_csv(
        out_dir / "phase_summary.csv",
        index=False,
    )

    # Mean across phases.
    numeric_cols = [
        col
        for col in summary_df.columns
        if col != "phase"
    ]

    mean_row = summary_df[
        numeric_cols
    ].mean(numeric_only=True)

    print("\n=== Mean across completed phases ===")
    print(mean_row.to_string())

    # Paper reference.
    print("\n=== Paper vanilla LSTM reference ===")

    for metric, value in PAPER_LSTM.items():
        print(
            f"{metric:12s}: "
            f"{value:.6f}"
        )


if __name__ == "__main__":
    main()
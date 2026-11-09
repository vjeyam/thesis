from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    initial_nav: float = 1.0
    top_k: int = 10
    trailing_stop: float = 0.07
    take_profit: float = 0.20
    max_holding_days: int = 5


def _simulate_position(
    df: pd.DataFrame,
    entry_idx: int,
    cfg: BacktestConfig,
):
    """
    Simulate one long position.

    Assumptions:
    - enter at close on prediction day
    - evaluate future daily high/low/close
    - take profit at +20% from entry
    - trailing stop remains 7% below highest price reached
    - otherwise exit after max_holding_days at close

    Returns decimal trade return, e.g. 0.05 = +5%.
    """
    entry_price = float(df.iloc[entry_idx]["close"])

    if not np.isfinite(entry_price) or entry_price <= 0:
        return 0.0

    highest = entry_price
    stop_price = entry_price * (1.0 - cfg.trailing_stop)
    take_profit_price = entry_price * (1.0 + cfg.take_profit)

    last_idx = min(
        entry_idx + cfg.max_holding_days,
        len(df) - 1,
    )

    for i in range(entry_idx + 1, last_idx + 1):
        row = df.iloc[i]

        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Update high-water mark and trailing stop.
        if np.isfinite(high):
            highest = max(highest, high)
            stop_price = highest * (1.0 - cfg.trailing_stop)

        # Conservative same-day ambiguity rule:
        # if both stop and TP could have fired, count stop first.
        if np.isfinite(low) and low <= stop_price:
            return stop_price / entry_price - 1.0

        if np.isfinite(high) and high >= take_profit_price:
            return cfg.take_profit

        # End-of-horizon exit.
        if i == last_idx:
            return close / entry_price - 1.0

    return 0.0


def backtest_predictions(
    predictions: pd.DataFrame,
    market,
    cfg: BacktestConfig,
):
    """
    Backtest one test phase.

    Required prediction columns:
        date
        symbol
        y_pred

    Portfolio rule:
    - rank stocks by predicted return for each signal date
    - select top-k
    - split current NAV equally across those k positions
    - each position is simulated independently for up to 5 trading days
    - portfolio advances in NON-OVERLAPPING 5-day blocks

    This avoids the overlapping-return compounding bug in the previous metric.
    """
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])

    signal_dates = sorted(predictions["date"].unique())

    nav = cfg.initial_nav
    rows = []

    # Prevent overlapping 5-day portfolios.
    last_exit_date = None

    for signal_date in signal_dates:
        if last_exit_date is not None and signal_date <= last_exit_date:
            continue

        day_preds = predictions[
            predictions["date"] == signal_date
        ].nlargest(cfg.top_k, "y_pred")

        trade_returns = []

        exit_dates = []

        for _, row in day_preds.iterrows():
            symbol = row["symbol"]

            if symbol not in market.frames:
                continue

            df = market.frames[symbol]

            if signal_date not in df.index:
                continue

            entry_idx = df.index.get_loc(signal_date)

            if isinstance(entry_idx, slice):
                entry_idx = entry_idx.start

            if entry_idx + 1 >= len(df):
                continue

            trade_return = _simulate_position(
                df=df,
                entry_idx=entry_idx,
                cfg=cfg,
            )

            trade_returns.append(trade_return)

            exit_idx = min(
                entry_idx + cfg.max_holding_days,
                len(df) - 1,
            )
            exit_dates.append(df.index[exit_idx])

        if not trade_returns:
            continue

        # Equal allocation among selected positions.
        portfolio_return = float(np.mean(trade_returns))

        nav_before = nav
        nav *= 1.0 + portfolio_return

        last_exit_date = max(exit_dates)

        rows.append(
            {
                "date": signal_date,
                "nav_before": nav_before,
                "portfolio_return": portfolio_return,
                "nav_after": nav,
                "n_positions": len(trade_returns),
            }
        )

    history = pd.DataFrame(rows)

    total_return = nav / cfg.initial_nav - 1.0

    return total_return, history
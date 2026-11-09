import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def _corr(x, y, rank=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if rank:
        return float(spearmanr(x, y).statistic)
    return float(pearsonr(x, y).statistic)

def daily_cross_sectional_metrics(predictions, top_n=10):
    rows = []

    for date, g in predictions.groupby("date"):
        g = g.dropna(
            subset=["y_true", "y_pred"]
        )

        if len(g) < 2:
            continue

        n = min(top_n, len(g))

        top_pred = g.nlargest(
            n,
            "y_pred",
        )

        rows.append(
            {
                "date": date,

                "IC": _corr(
                    g["y_pred"],
                    g["y_true"],
                ),

                "Rank_IC": _corr(
                    g["y_pred"],
                    g["y_true"],
                    rank=True,
                ),

                # Precision of top-N predicted
                # profitable stocks.
                "Prec@N": float(
                    (
                        top_pred["y_true"]
                        > 0
                    ).mean()
                ),

                # Kept only as a diagnostic.
                "signal_return": float(
                    top_pred["y_true"].mean()
                ),
            }
        )

    daily = pd.DataFrame(rows)

    if daily.empty:
        return daily, {}

    ic = daily["IC"].dropna()
    rank_ic = daily[
        "Rank_IC"
    ].dropna()

    ic_std = ic.std(ddof=0)
    rank_ic_std = rank_ic.std(
        ddof=0
    )

    summary = {
        # Return is overwritten later by
        # the actual portfolio backtester.
        "Return": np.nan,

        "IC": float(
            ic.mean()
        ),

        "ICIR": (
            float(
                ic.mean() / ic_std
            )
            if ic_std != 0
            else np.nan
        ),

        "Rank_IC": float(
            rank_ic.mean()
        ),

        "Rank_ICIR": (
            float(
                rank_ic.mean()
                / rank_ic_std
            )
            if rank_ic_std != 0
            else np.nan
        ),

        "Prec@N": float(
            daily["Prec@N"].mean()
        ),
    }

    return daily, summary

def regression_metrics(predictions):
    err = predictions["y_pred"].to_numpy() - predictions["y_true"].to_numpy()
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
    }

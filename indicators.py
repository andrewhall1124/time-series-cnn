import talib
import polars as pl


def with_labels(full_df):
    return (
        full_df.with_columns(pl.col("close").shift(5).alias("rolling_mid"))
        .with_columns(
            pl.col("close").rolling_max(window_size=11).alias("rolling_max"),
            pl.col("close").rolling_min(window_size=11).alias("rolling_min"),
        )
        .with_columns(
            pl.when(pl.col("rolling_mid").eq(pl.col("rolling_max")))
            .then(pl.lit("SELL"))
            .when(pl.col("rolling_mid").eq(pl.col("rolling_min")))
            .then(pl.lit("BUY"))
            .otherwise(pl.lit("HOLD"))
            .alias("label")
        )
        .drop(["rolling_mid", "rolling_max", "rolling_min"])
    )


indicators = [
    "rsi",
    "william",
    "macd",
    "ppo",
    "roc",
    "cmo",
    "sma",
    "ema",
    "wma",
    "tema",
    "cci",
    "dmi",
    "psar",
]


def transform(full_df):
    to_merge = []
    for ticker in full_df["ticker"].unique():
        df = full_df.filter(pl.col("ticker") == ticker).sort("date")
        for period in range(6, 28):
            new_cols = []
            new_cols.append(
                talib.RSI(df["close"], timeperiod=period).rename(f"rsi_{period}")
            )
            new_cols.append(
                talib.WILLR(
                    df["high"], df["low"], df["close"], timeperiod=period
                ).rename(f"william_{period}")
            )
            if period >= 14:
                # MFI requires at least 14 periods for some reason
                new_cols.append(
                    talib.MFI(
                        df["high"],
                        df["low"],
                        df["close"],
                        df["volume"],
                        timeperiod=period,
                    ).rename(f"mfi_{period}")
                )
            new_cols.append(
                talib.MACD(df["close"], fastperiod=period, slowperiod=2 * period + 2)[
                    0
                ].rename(f"macd_{period}")
            )
            new_cols.append(
                talib.PPO(
                    df["close"], fastperiod=period, slowperiod=2 * period + 2
                ).rename(f"ppo_{period}")
            )
            new_cols.append(
                talib.ROC(df["close"], timeperiod=period).rename(f"roc_{period}")
            )
            # TODO: cmfi
            new_cols.append(
                talib.CMO(df["close"], timeperiod=period).rename(f"cmo_{period}")
            )
            new_cols.append(
                talib.SMA(df["close"], timeperiod=period).rename(f"sma_{period}")
            )
            new_cols.append(
                talib.EMA(df["close"], timeperiod=period).rename(f"ema_{period}")
            )
            new_cols.append(
                talib.WMA(df["close"], timeperiod=period).rename(f"wma_{period}")
            )
            # TODO: hma
            new_cols.append(
                talib.TEMA(df["close"], timeperiod=period).rename(f"tema_{period}")
            )
            new_cols.append(
                talib.CCI(df["high"], df["low"], df["close"], timeperiod=period).rename(
                    f"cci_{period}"
                )
            )
            # TODO: dpo
            # TODO: kst
            # TODO: eom
            # TODO: ibr
            new_cols.append(
                talib.DX(df["high"], df["low"], df["close"], timeperiod=period).rename(
                    f"dmi_{period}"
                )
            )
            new_cols.append(talib.SAR(df["high"], df["low"]).rename(f"psar_{period}"))

            numerical_cols = pl.col(pl.Float64, pl.Int64)
            df = df.with_columns(*new_cols).with_columns(
                (numerical_cols - numerical_cols.min())
                / (numerical_cols.max() - numerical_cols.min())
            )

        to_merge.append(df.drop_nans())
    return pl.concat(to_merge)

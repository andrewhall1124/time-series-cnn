import talib
import polars as pl
import ta


LABEL_WINDOW = 11
MAX_PERIOD = 28
NORM_WINDOW = 56


def with_labels(df):
    """
    Add labels for a df with a single ticker.
    """
    assert df["ticker"].n_unique() == 1
    return (
        df.with_columns(pl.col("close").shift(LABEL_WINDOW // 2).alias("rolling_mid"))
        .with_columns(
            pl.col("close").rolling_max(window_size=LABEL_WINDOW).alias("rolling_max"),
            pl.col("close").rolling_min(window_size=LABEL_WINDOW).alias("rolling_min"),
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


def transform(full_df):
    to_merge = []
    for ticker in full_df["ticker"].unique():
        print(f"Processing {ticker}")
        df = (
            full_df.filter(pl.col("ticker") == ticker)
            .sort("date", descending=False)
            .drop_nans()
        )
        for period in range(6, MAX_PERIOD):
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
            mfv = (
                ((df["close"] - df["low"]) - (df["high"] - df["close"]))
                / (df["high"] - df["low"])
            ) * df["volume"]
            new_cols.append(
                (
                    mfv.rolling_sum(window_size=period)
                    / df["volume"].rolling_sum(window_size=period)
                ).rename(f"cmfi_{period}")
            )
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
            new_cols.append(
                talib.WMA(
                    2 * talib.WMA(df["close"], timeperiod=period // 2)
                    - talib.WMA(df["close"], timeperiod=period),
                    timeperiod=round(period**0.5),
                ).rename(f"hma_{period}")
            )
            new_cols.append(
                talib.TEMA(df["close"], timeperiod=period).rename(f"tema_{period}")
            )
            new_cols.append(
                talib.CCI(df["high"], df["low"], df["close"], timeperiod=period).rename(
                    f"cci_{period}"
                )
            )
            new_cols.append(
                pl.from_pandas(
                    ta.trend.dpo(df["close"].to_pandas(), window=period)
                ).rename(f"dpo_{period}")
            )
            new_cols.append(
                pl.from_pandas(
                    ta.trend.kst(
                        df["close"].to_pandas(),
                        window1=period,
                        window2=period,
                        window3=period,
                        window4=round(period * 1.5),
                    )
                ).rename(f"kst_{period}")
            )
            new_cols.append(
                pl.from_pandas(
                    ta.volume.EaseOfMovementIndicator(
                        df["high"].to_pandas(),
                        df["low"].to_pandas(),
                        df["volume"].to_pandas(),
                        window=period,
                    ).ease_of_movement()
                ).rename(f"eom_{period}")
            )
            # TODO: ibr
            new_cols.append(
                talib.DX(df["high"], df["low"], df["close"], timeperiod=period).rename(
                    f"dmi_{period}"
                )
            )
            new_cols.append(talib.SAR(df["high"], df["low"]).rename(f"psar_{period}"))

            df = df.with_columns(*new_cols)

        numerical_cols = pl.col(pl.Float64, pl.Int64)
        original_close = df["close"]
        df = df.with_columns(
            (numerical_cols - numerical_cols.rolling_min(NORM_WINDOW))
            / (
                numerical_cols.rolling_max(NORM_WINDOW)
                - numerical_cols.rolling_min(NORM_WINDOW)
            )
        )
        # Save original close price for backtesting
        df = df.with_columns(original_close.rename("original_close"))
        df = with_labels(df)

        to_merge.append(df.drop_nans())
    return pl.concat(to_merge).sort("date", descending=False)

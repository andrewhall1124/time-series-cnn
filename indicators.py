import talib
import polars as pl

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
            rsi = talib.RSI(df["close"], timeperiod=period).rename(f"rsi_{period}")
            william = talib.WILLR(
                df["high"], df["low"], df["close"], timeperiod=period
            ).rename(f"william_{period}")
            mfi = talib.MFI(
                df["high"], df["low"], df["close"], df["volume"], timeperiod=period
            ).rename(f"mfi_{period}")
            macd = talib.MACD(
                df["close"], fastperiod=period, slowperiod=2 * period + 2
            )[0].rename(f"macd_{period}")
            ppo = talib.PPO(
                df["close"], fastperiod=period, slowperiod=2 * period + 2
            ).rename(f"ppo_{period}")
            roc = talib.ROC(df["close"], timeperiod=period).rename(f"roc_{period}")
            # TODO: cmfi
            cmo = talib.CMO(df["close"], timeperiod=period).rename(f"cmo_{period}")
            sma = talib.SMA(df["close"], timeperiod=period).rename(f"sma_{period}")
            ema = talib.EMA(df["close"], timeperiod=period).rename(f"ema_{period}")
            wma = talib.WMA(df["close"], timeperiod=period).rename(f"wma_{period}")
            # TODO: hma
            tema = talib.TEMA(df["close"], timeperiod=period).rename(f"tema_{period}")
            cci = talib.CCI(
                df["high"], df["low"], df["close"], timeperiod=period
            ).rename(f"cci_{period}")
            # TODO: dpo
            # TODO: kst
            # TODO: eom
            # TODO: ibr
            dmi = talib.DX(
                df["high"], df["low"], df["close"], timeperiod=period
            ).rename(f"dmi_{period}")
            psar = talib.SAR(df["high"], df["low"]).rename(f"psar_{period}")

            numerical_cols = pl.exclude("date", "ticker")
            df = df.with_columns(
                rsi,
                william,
                mfi,
                macd,
                ppo,
                roc,
                cmo,
                sma,
                ema,
                wma,
                tema,
                cci,
                dmi,
                psar,
            ).with_columns(
                (numerical_cols - numerical_cols.min())
                / (numerical_cols.max() - numerical_cols.min())
            )

        to_merge.append(df.drop_nans())
    return pl.concat(to_merge)

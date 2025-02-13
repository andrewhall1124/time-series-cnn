import talib
import polars as pl


def add_indicators(df: pl.DataFrame) -> pl.DataFrame:
    for period in range(6, 28):
        df = df.with_columns(
            # rsi
            talib.RSI(df["prc"], timeperiod=period).rename(f"rsi_{period}"),
            # william
            # talib.WILLR(df["high"], df["low"], df["prc"], timeperiod=period).rename(
            #     f"william_{period}"
            # ),
            # mfi
            # talib.MFI(
            #     df["high"], df["low"], df["prc"], df["volume"], timeperiod=period
            # ).rename(f"mfi_{period}"),
            # macd
            talib.MACD(df["prc"], fastperiod=period, slowperiod=2 * period + 2)[
                0
            ].rename(f"macd_{period}"),
            # ppo
            talib.PPO(df["prc"], fastperiod=period, slowperiod=2 * period + 2).rename(
                f"ppo_{period}"
            ),
            # roc
            talib.ROC(df["prc"], timeperiod=period).rename(f"roc_{period}"),
            # TODO: cmfi
            # cmo
            talib.CMO(df["prc"], timeperiod=period).rename(f"cmo_{period}"),
            # sma
            talib.SMA(df["prc"], timeperiod=period).rename(f"sma_{period}"),
            # ema
            talib.EMA(df["prc"], timeperiod=period).rename(f"ema_{period}"),
            # wma
            talib.WMA(df["prc"], timeperiod=period).rename(f"wma_{period}"),
            # TODO: hma
            # tema
            talib.TEMA(df["prc"], timeperiod=period).rename(f"tema_{period}"),
            # cci
            # talib.CCI(df["high"], df["low"], df["prc"], timeperiod=period).rename(
            #     f"cci_{period}"
            # ),
            # TODO: dpo
            # TODO: kst
            # TODO: eom
            # TODO: ibr
            # dmi
            # talib.DX(df["high"], df["low"], df["prc"], timeperiod=period).rename(
            #     f"dmi_{period}"
            # ),
            # psar
            # talib.SAR(df["high"], df["low"]).rename(f"psar_{period}"),
        )
    return df

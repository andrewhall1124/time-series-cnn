import talib
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from ta.volume import ChaikinMoneyFlowIndicator, EaseOfMovementIndicator
from ta.trend import dpo, kst
from tqdm import tqdm
import numpy as np
import os

# Configuration constants
MAX_PERIOD = 56
NORM_WINDOW = 365


def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute technical indicators for a single permno partition in one go.
    """
    close = df["close"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)

    feature_dict = {}
    for p in range(6, MAX_PERIOD):
        feature_dict[f"rsi_{p}"] = talib.RSI(close, timeperiod=p)
        feature_dict[f"will_{p}"] = talib.WILLR(high, low, close, timeperiod=p)
        if p >= 14:
            feature_dict[f"mfi_{p}"] = talib.MFI(high, low, close, volume, timeperiod=p)

        macd, _, _ = talib.MACD(close, fastperiod=p, slowperiod=2 * p + 2)
        feature_dict[f"macd_{p}"] = macd

        feature_dict[f"ppo_{p}"] = talib.PPO(close, fastperiod=p, slowperiod=2 * p + 2)
        feature_dict[f"roc_{p}"] = talib.ROC(close, timeperiod=p)

        feature_dict[f"cmfi_{p}"] = (
            ChaikinMoneyFlowIndicator(
                high=df["high"].to_pandas(),
                low=df["low"].to_pandas(),
                close=df["close"].to_pandas(),
                volume=df["volume"].to_pandas(),
                window=p,
            )
            .chaikin_money_flow()
            .to_numpy()
        )

        feature_dict[f"cmo_{p}"] = talib.CMO(close, timeperiod=p)
        feature_dict[f"sma_{p}"] = talib.SMA(close, timeperiod=p)
        feature_dict[f"ema_{p}"] = talib.EMA(close, timeperiod=p)
        feature_dict[f"wma_{p}"] = talib.WMA(close, timeperiod=p)

        half = talib.WMA(close, timeperiod=max(1, p // 2))
        full = talib.WMA(close, timeperiod=p)
        feature_dict[f"hma_{p}"] = talib.WMA(2 * half - full, timeperiod=round(p**0.5))

        feature_dict[f"tema_{p}"] = talib.TEMA(close, timeperiod=p)
        feature_dict[f"cci_{p}"] = talib.CCI(high, low, close, timeperiod=p)

        feature_dict[f"dpo_{p}"] = dpo(
            close=df["close"].to_pandas(), window=p
        ).to_numpy()

        feature_dict[f"kst_{p}"] = kst(
            close=df["close"].to_pandas(),
            window1=p,
            window2=p,
            window3=p,
            window4=round(p * 1.5),
        ).to_numpy()

        feature_dict[f"eom_{p}"] = (
            EaseOfMovementIndicator(
                high=df["high"].to_pandas(),
                low=df["low"].to_pandas(),
                volume=df["volume"].to_pandas(),
                window=p,
            )
            .ease_of_movement()
            .to_numpy()
        )

        # TODO: IBR
        feature_dict[f"dmi_{p}"] = talib.DX(high, low, close, timeperiod=p)
        feature_dict[f"psar_{p}"] = talib.SAR(high, low)

    feature_df = pl.DataFrame(feature_dict)
    return df.with_columns(feature_df)


def normalize_and_label(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize numeric columns over a rolling window and add label columns.
    """
    # identify numeric columns
    num_cols = [
        c
        for c, dt in df.schema.items()
        if dt in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
    ]
    exprs = []
    for c in num_cols:
        rmin = pl.col(c).rolling_min(NORM_WINDOW)
        rmax = pl.col(c).rolling_max(NORM_WINDOW)
        exprs.append(((pl.col(c) - rmin) / (rmax - rmin)).alias(c))

    df = df.with_columns(exprs)
    orig = df["close"]
    df = df.with_columns(
        orig.alias("original_close"), (orig.shift(-1) / orig).alias("label")
    )
    return df


def process_and_save(df: pl.DataFrame, permno: int, out_dir: str) -> None:
    """
    Process one permno group & save result to disk if not empty.
    """
    df = df.sort("date").drop_nulls()
    if len(df) < NORM_WINDOW * 2:
        return
    df = compute_features(df)
    df = normalize_and_label(df).drop_nulls()
    if df.is_empty():
        return
    os.makedirs(out_dir, exist_ok=True)
    df.write_parquet(os.path.join(out_dir, f"{permno}.parquet"))


def transform(full_df: pl.DataFrame) -> pl.DataFrame:
    """
    Transform a full DataFrame (multiple permnos) into a single DataFrame
    with features and labels, using multiprocessing across permno groups.
    """
    # split into per-permno lists
    permnos = full_df["permno"].unique().to_list()
    groups = [(full_df.filter(pl.col("permno") == p), p) for p in permnos]

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    lambda args: process_and_save(*args, "./data/temp"), groups
                ),
                total=len(groups),
            )
        )

    # Load and concat
    results = []
    for permno in permnos:
        file_path = os.path.join("./data/temp", f"{permno}.parquet")
        if os.path.exists(file_path):
            results.append(
                pl.read_parquet(file_path).with_columns(pl.lit(permno).alias("permno"))
            )

    # This line fails: TODO switch to data iterator
    return (
        pl.concat(results).sort("date", descending=False) if results else pl.DataFrame()
    )

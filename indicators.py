import talib
import polars as pl
import sqlite3
from ta.volume import ChaikinMoneyFlowIndicator, EaseOfMovementIndicator
from ta.trend import dpo, kst
from tqdm import tqdm
import numpy as np
import datetime
from data_utils import load_daily_crsp

# Configuration constants
MAX_PERIOD = 56
NORM_WINDOW = 251
DB_PATH = "./data/features.sqlite"


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

    return df.with_columns(pl.DataFrame(feature_dict))


def normalize_and_label(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize numeric columns over a rolling window and add label columns.
    """
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
    close = df["close"]
    df = df.with_columns(
        close.alias("original_close"), (close.shift(-1) / close).alias("label")
    )
    return df


def transform_and_save(full_df: pl.DataFrame, db_path: str = DB_PATH) -> None:
    """
    Transform a full DataFrame into a SQLite DB of features & labels.
    """
    conn = sqlite3.connect(db_path)
    # Speed up writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    cursor = conn.cursor()

    # Drop any existing table
    cursor.execute("DROP TABLE IF EXISTS features;")
    table_created = False

    permnos = full_df["permno"].unique().to_list()
    for permno in tqdm(permnos):
        df = (
            full_df.filter(pl.col("permno") == permno)
            .sort("date")
            .drop_nulls()
            .drop_nans()
        )
        if len(df) < NORM_WINDOW * 2:
            continue

        df_feat = compute_features(df)
        df_norm = normalize_and_label(df_feat).drop_nulls().drop_nans()
        if df_norm.is_empty():
            continue

        # add permno column
        df_out = df_norm.with_columns(pl.lit(permno).alias("permno"))
        cols = df_out.columns

        if not table_created:
            # create table based on schema
            col_defs = []
            for col, dtype in df_out.schema.items():
                if dtype == pl.Utf8:
                    col_type = "TEXT"
                elif dtype in (pl.Int64, pl.Int32):
                    col_type = "INTEGER"
                else:
                    col_type = "REAL"
                col_defs.append(f"{col} {col_type}")
            cursor.execute(f"CREATE TABLE features ({', '.join(col_defs)});")
            table_created = True

        placeholders = ",".join(["?" for _ in cols])
        insert_sql = (
            f"INSERT INTO features ({', '.join(cols)}) VALUES ({placeholders});"
        )
        rows = df_out.select(cols).rows()
        cursor.executemany(insert_sql, rows)
        conn.commit()
    conn.close()


if __name__ == "__main__":
    # Example usage:
    df = load_daily_crsp(
        start_date=datetime.date(2009, 1, 1), end_date=datetime.date(2025, 3, 31)
    )
    transform_and_save(df)

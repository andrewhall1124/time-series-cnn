"""Module for loading data as polars dataframes"""

import polars as pl
from datetime import date
from pathlib import Path
from tqdm import tqdm
import yfinance as yf

CRSP_FOLDER = Path("data/crsp_daily")
CRSP_SCHEMA = {
    "permno": pl.Int64,
    "permco": pl.Int64,
    "date": pl.Date,
    "ncusip": pl.String,
    "ticker": pl.String,
    "shrcd": pl.Int64,
    "exchcd": pl.Int64,
    "siccd": pl.Int64,
    "prc": pl.Float64,
    "ret": pl.Float64,
    "retx": pl.Float64,
    "vol": pl.Float64,
    "shrout": pl.Float64,
    "cfacshr": pl.Float64,
}

YFINANCE_SCHEMA = {
    "date": pl.Date,
    "ticker": pl.Categorical,
    "close": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "open": pl.Float64,
    "volume": pl.Int64
}


def load_daily_crsp_file(
    start_date: date | None = None, end_date: date | None = None
) -> pl.DataFrame:
    start_date = start_date or date(1925, 12, 31)
    end_date = end_date or date(2024, 11, 29)

    # Check
    if end_date < start_date:
        msg = "Error: end_date is before start_date"
        raise ValueError(msg)

    # Join all yearly files
    years = list(range(start_date.year, end_date.year + 1))

    dfs = []
    for year in tqdm(sorted(years), desc="Loading crsp daily data"):
        file_path = f"dsf_{year}.parquet"
        df = pl.read_parquet(CRSP_FOLDER / file_path)

        # Clean
        df = clean_crsp(df)

        # Append
        dfs.append(df)

    # Concatenate
    df: pl.DataFrame = pl.concat(dfs)

    # Sort
    df = df.sort(["permno", "permno"])

    return df


def clean_crsp(df: pl.DataFrame):
    df = df.cast(CRSP_SCHEMA)
    return df


def load_yfinance(
    tickers: list[str], start_date: date | None = None, end_date: date | None = None
) -> pl.DataFrame:
    start_date = start_date or date(1925, 12, 31)
    end_date = end_date or date(2024, 11, 29)

    df = (
        yf.download(tickers=tickers, start=start_date, end=end_date)
        .stack(future_stack=True)
        .reset_index()
    )

    df = pl.from_pandas(df)

    return (
        df
        .rename({col: col.lower() for col in df.columns})
        .cast(YFINANCE_SCHEMA)
    )

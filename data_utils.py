"""Module for loading data as polars dataframes"""

import polars as pl
from datetime import date
from pathlib import Path
from tqdm import tqdm

FOLDER_DIR = Path("data/crsp_daily")
SCHEMA = {
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
        df = pl.read_parquet(FOLDER_DIR / file_path)

        # Clean
        df = clean(df)

        # Append
        dfs.append(df)

    # Concatenate
    df: pl.DataFrame = pl.concat(dfs)

    # Sort
    df = df.sort(["permno", "permno"])

    return df

def clean(df: pl.DataFrame):
    df = df.cast(SCHEMA)
    return df

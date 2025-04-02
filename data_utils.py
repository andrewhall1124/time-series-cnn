"""Module for loading data as polars dataframes"""

import polars as pl
from datetime import date
from pathlib import Path
import yfinance as yf

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
    "volume": pl.Int64,
}


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

    return df.rename({col: col.lower() for col in df.columns}).cast(YFINANCE_SCHEMA)


def generate_annotations(df: pl.DataFrame, ticker_col: str, window: int) -> pl.DataFrame:
    return (
        df
        # Compute returns
        .with_columns(pl.col("close").pct_change().log1p().alias("logret"))
        .sort([ticker_col, "date"])
        # Future window return
        .with_columns(
            pl.col("logret")
            .rolling_sum(window_size=window)
            .shift(-window)
            .alias("ret_20")
        )
        .drop_nulls("ret_20")
        .sort([ticker_col, "date"])
        # Label
        .with_columns(
            pl.when(pl.col("ret_20").ge(0)).then(1).otherwise(0).alias("label")
        )
        .with_columns(
            pl.format(
                "{}_{}_{}.png",
                pl.col(ticker_col),
                pl.col("date").dt.strftime("%Y%d%m"),
                window,
            ).alias("file")
        )
        .select(["file", "label"])
    )


def clean_raw_crsp_file(raw_file_path: str, clean_file_path: str) -> None:
    schema = {
        "NCUSIP": pl.String,
        "CUSIP": pl.String,
        "RET": pl.String,
        "RETX": pl.String,
        "CFACPR": pl.Float64,
        "CFACSHR": pl.Float64,
    }
    # Load
    df = pl.read_csv(raw_file_path, schema_overrides=schema)

    # Data cleaning
    df = df.filter(~pl.col(["RET"]).is_in(["C", "B"]))
    df = df.filter(~pl.col(["RETX"]).is_in(["C", "B"]))
    df = df.cast({"RET": pl.Float64, "RETX": pl.Float64, "date": pl.Date})

    # Lower columns
    df = df.rename({col: col.lower() for col in df.columns})

    # Filters
    df = df.filter(pl.col("shrcd").is_between(10, 11))  # Stocks
    df = df.filter(pl.col("exchcd").is_between(1, 3))  # NYSE, AMEX, NASDAQ

    # Columns
    df = df.select(
        "date",
        "permno",
        # "permco",
        # "nwperm",
        # "cusip",
        # "ncusip",
        # "tsymbol",
        "ticker",
        "comnam",
        # "shrcd",
        # "exchcd",
        "prc",
        "bidlo",
        "askhi",
        "openprc",
        "vol",
        "ret",
        # "retx",
        # "shrout",
        # "facpr",
        # "facshr",
        # "cfacshr",
        # "cfacpr",
    )

    # No negative prices.
    df = df.with_columns(pl.col("prc").abs())

    # Sort
    df = df.sort(["permno", "date"])

    df.write_parquet(clean_file_path)


def load_daily_crsp(start_date: date, end_date: date, look_back: int) -> pl.LazyFrame:
    return (
        pl.read_parquet("data/crsp_daily.parquet")
        .filter(pl.col("date").is_between(start_date, end_date))
        .filter(pl.col('prc').ge(5))
        .with_columns(
            pl.col("ret").shift(1).fill_null(0).add(1).cum_prod().sub(1).over('permno').alias("cumret")
        )
        .with_columns(pl.lit(1).mul(pl.col("cumret").add(1)).alias('close'))
        .with_columns(
            pl.col('openprc').truediv(pl.col('prc')).mul('close').alias('open'),
            pl.col('askhi').truediv(pl.col('prc')).mul('close').alias('high'),
            pl.col('bidlo').truediv(pl.col('prc')).mul('close').alias('low'),
            pl.col('vol').alias('volume'),
            pl.col("close").rolling_mean(window_size=look_back, min_samples=1).over('permno').alias("ma")
        )
        .select('date', 'permno', 'open', 'high', 'low', 'close', 'volume', 'ma')
    )

def load_daily_crsp_annotations(start_date: date, end_date: date) -> pl.LazyFrame:
    return (
        pl.read_parquet("data/crsp_daily.parquet")
        .filter(pl.col("date").is_between(start_date, end_date))
        .filter(pl.col('prc').ge(5))
        .select('date', 'permno', 'ticker', 'prc', 'ret')
    )


if __name__ == "__main__":
    df = load_daily_crsp(start_date=date(2010, 3, 5), end_date=date(2019, 12, 31))

    print(df)

    annotations = generate_annotations(df, 'permno', 20)
    annotations.write_csv("data/annotations_20.csv")

    print(annotations) 

    

# TODO: Get CRSP data
# TODO: Rework to work with CRSP data
# TODO: Generate volume bars
# TODO: Generate moving average price
# TODO: Save image as date_ticker_window.png

import data_utils as du
import polars as pl
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm
import numpy as np

pl.enable_string_cache()

def get_period_pairs(dates: pl.DataFrame, look_back: int) -> list[tuple[date, date]]:
    return (
        dates.with_columns(pl.col("date").shift(look_back - 1).alias("start"))
        .select("start", pl.col("date").alias("end"))
        .drop_nulls()
        .rows()
    )

def generate_ohlc_bars(df: pl.DataFrame, n_days: int, n_bins: int) -> np.array:
    labels = [f"{i}" for i in range(n_bins)]
    index = pl.DataFrame({'bin': [str(i) for i in range(n_bins)]})
    df = (
        df
        # Create index column
        .with_row_index("index", 1)
        # Unpivot OHLC variables
        .unpivot(on=["open", "high", "low", "close"], index="index")
        .sort(["index", "variable"])
        # Put variable values into bins for the whole sample by ticker
        .with_columns(
            pl.col("value")
            .qcut(n_bins, labels=labels, allow_duplicates=True)
            .alias("bin"),
            pl.lit(1).alias("in_bin"),
        )
        .sort("bin")
        # Pivot to bins wide
        .pivot(on="bin", index=["index", "variable"], values="in_bin")
        .sort("index", "variable")
        # Concatenate high and low together
        .with_columns(
            pl.col("variable").replace({"high": "high_low", "low": "high_low"})
        )
        # Create x axis column
        .with_columns(
            pl.concat_str(pl.col("index"), pl.col("variable"), separator="_").alias("x")
        )
        .drop("index", "variable")
        # Unpivot and pivot to x and y axes
        .unpivot(index="x", variable_name="bin", value_name="in_bin")
        .pivot(on="x", index="bin", values="in_bin", aggregate_function="max")
    )
    df = (
        index
        .join(df, on='bin', how='left')
        .cast({"bin": pl.Int32})
        .sort("bin", descending=True)
        # Interpolate vertical missing values
        .interpolate()
        .fill_null(0)
        # Reorder x axis to be by date open, high_low, close
        .select(
            [
                f"{i + 1}_{var}"
                for i in range(n_days)
                for var in ["open", "high_low", "close"]
            ]
        )
        .to_numpy()
    )

    return df

def generate_ma_line(df: pl.DataFrame, n_days: int, n_bins: int) -> np.array:
    labels = [f"{i}" for i in range(n_bins)]
    min_value = df['low'].min()
    max_value = df['high'].max()
    breaks = np.linspace(min_value, max_value, n_bins + 2)[1:n_bins]
    index = pl.DataFrame({'bin': [str(i) for i in range(n_bins)]})
    df = (
        df
        .with_row_index('index', 1)
        .with_columns(pl.lit(None).alias('a'), pl.lit(None).alias('z'))
        .unpivot(index='index', on=['a', 'ma', 'z'])
        .sort('index', 'variable')
        .with_columns(pl.col('value').interpolate())
        .fill_null(strategy='forward')
        .fill_null(strategy='backward')
        .with_columns(
            pl.col("value")
            .cut(breaks=breaks, labels=labels)
            .alias("bin"),
            pl.lit(1).alias("in_bin"),
        )
        .sort("bin")
        # Pivot to bins wide
        .pivot(on="bin", index=["index", "variable"], values="in_bin")
        .sort("index", "variable")
        # Create x axis column
        .with_columns(
            pl.concat_str(pl.col("index"), pl.col("variable"), separator="_").alias("x")
        )
        .drop("index", "variable")
        # Unpivot and pivot to x and y axes
        .unpivot(index="x", variable_name="bin", value_name="in_bin")
        .pivot(on="x", index="bin", values="in_bin", aggregate_function="max")
    )
    df = (
        index.join(df, on='bin', how='left')
        .cast({"bin": pl.Int32})
        .sort("bin", descending=True)
                .interpolate()
        .fill_null(0)
        # Reorder x axis to be by date open, high_low, close
        .select(
            [
                f"{i + 1}_{var}"
                for i in range(n_days)
                for var in ["a", "ma", "z"]
            ]
        )
        .to_numpy()
    )

    return df

def generate_volumes_bars(df: pl.DataFrame) -> np.array:

    df = (
        df.select(['date', 'volume'])
    )

    return df



def generate_images(df: pl.DataFrame, look_back: int = 20, n_bins: int = 40) -> None:
    tickers = df["ticker"].unique().sort()
    dates = df.select("date").unique().sort("date")
    periods = get_period_pairs(dates, look_back)

    for ticker in tqdm(tickers, desc="Generating images...", position=0, disable=True):
        # Subset on ticker
        ticker_df = df.filter(pl.col("ticker").eq(ticker))

        for start, end in tqdm(
            periods[21:22], desc=f"Generating images for {ticker}", leave=False, position=1, disable=True
        ):
            # Subset on period
            period_df = ticker_df.filter(pl.col("date").is_between(start, end))

            # Generate image
            ohlc_bars = generate_ohlc_bars(period_df, look_back, n_bins)
            ma_line = generate_ma_line(period_df, look_back, n_bins)
            volume_bars = generate_volumes_bars(period_df)
            # print(ohlc_bars)
            # print(ma_line)
            print(volume_bars)
            # print(ohlc_bars.shape)
            # print(ma_line.shape)

            image = np.maximum(ohlc_bars, ma_line)
        
            # Save
            # plt.imshow(image, cmap='grey')
            # plt.show()

            plt.imsave(f"images/{ticker}_{end.strftime('%Y%m%d')}.png", image, cmap="grey")
            break


if __name__ == '__main__':
    tickers = ["AAPL"]
    df = (
        du.load_yfinance(tickers)
        .with_columns(pl.col('close').rolling_mean(window_size=20).alias('ma'))
    )
    generate_images(df, look_back=20, n_bins=40) # 60 x 40 image

import data_utils as du
import polars as pl
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm
import numpy as np
import os
import zipfile

pl.enable_string_cache()


def get_period_pairs(dates: pl.DataFrame, window: int) -> list[tuple[date, date]]:
    """Function for getting start and end dates for a given window size."""
    return (
        dates.with_columns(pl.col("date").shift(window - 1).alias("start"))
        .select("start", pl.col("date").alias("end"))
        .drop_nulls()
        .rows()
    )


def generate_ohlc_bars(df: pl.DataFrame, n_days: int, n_bins: int) -> np.array:
    """Function for generating numpy array representing Open High Low Close bars."""
    labels = [f"{i}" for i in range(n_bins)]
    index = pl.DataFrame({"bin": [str(i) for i in range(n_bins)]})

    min_value = min(df["low"].min(), df["ma"].min())
    max_value = max(df["high"].max(), df["ma"].max())
    breaks = np.linspace(min_value, max_value, n_bins + 2)[1:n_bins]

    df = (
        df
        # Create index column
        .with_row_index("index", 1)
        # Unpivot OHLC variables
        .unpivot(on=["open", "high", "low", "close"], index="index")
        .sort(["index", "variable"])
        # Put variable values into bins for the whole sample by ticker
        .with_columns(
            pl.col("value").cut(breaks=breaks, labels=labels).alias("bin"),
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
        # Ensure we have a row for every bin
        .join(df, on="bin", how="left")
        # Sort bins
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
    """Function for generating moving average price line as a numpy array."""
    labels = [f"{i}" for i in range(n_bins)]
    index = pl.DataFrame({"bin": [str(i) for i in range(n_bins)]})

    min_value = df["low"].min()
    max_value = df["high"].max()
    breaks = np.linspace(min_value, max_value, n_bins + 2)[1:n_bins]

    df = (
        df
        # Add index column
        .with_row_index("index", 1)
        # Add arbitrary "open" and "close" columns
        .with_columns(pl.lit(None).alias("a"), pl.lit(None).alias("z"))
        # Put into variable long format
        .unpivot(index="index", on=["a", "ma", "z"])
        .sort("index", "variable")
        # Interpolate the open and close moving average price
        .with_columns(pl.col("value").interpolate())
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
        # Bin moving average price based on min and max value
        .with_columns(
            pl.col("value").cut(breaks=breaks, labels=labels).alias("bin"),
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
        index
        # Ensure every bin has a row
        .join(df, on="bin", how="left")
        # Sort bins
        .cast({"bin": pl.Int32})
        .sort("bin", descending=True)
        # Fill null with zeroes
        .fill_null(0)
        # Reorder x axis to be by date and a, ma, and z
        .select([f"{i + 1}_{var}" for i in range(n_days) for var in ["a", "ma", "z"]])
        .to_numpy()
    )

    return df


def generate_volumes_bars(df: pl.DataFrame, n_days: int, n_bins: int) -> np.array:
    labels = [f"{i}" for i in range(n_bins)]
    index = pl.DataFrame({"bin": [str(i) for i in range(n_bins)]})
    df = (
        df
        # Create index
        .with_row_index("index", 1)
        # Create "open" and "close" columns as null
        .with_columns(pl.lit(None).alias("a"), pl.lit(None).alias("z"))
        # Put into variable long format
        .unpivot(index="index", on=["a", "volume", "z"])
        # Bin volume values
        .with_columns(
            pl.col("value")
            .qcut(n_bins, labels=labels, allow_duplicates=True)
            .alias("bin"),
            pl.lit(1).alias("in_bin"),
        )
        .sort("bin")
        # Pivot to bins wide
        .pivot(on="bin", index=["index", "variable"], values="in_bin")
        .drop("null")
        .sort("index", "variable")
        # Create x axis column
        .with_columns(
            pl.concat_str(pl.col("index"), pl.col("variable"), separator="_").alias("x")
        )
        .drop("index", "variable")
        # Unpivot and pivot to get x and y axes
        .unpivot(index="x", variable_name="bin", value_name="in_bin")
        .pivot(on="x", index="bin", values="in_bin", aggregate_function="max")
    )

    df = (
        index
        # Ensure every bin has a row
        .join(df, on="bin", how="left")
        # Sort bins
        .cast({"bin": pl.Int32})
        .sort("bin", descending=True)
        # Fill bottom of bars
        .fill_null(strategy="forward")
        .fill_null(0)
        # Reorder x axis to be by date open, high_low, close
        .select(
            [f"{i + 1}_{var}" for i in range(n_days) for var in ["a", "volume", "z"]]
        )
        .to_numpy()
    )

    return df


def generate_images(
    df: pl.DataFrame, 
    ticker_col: str, 
    look_back: int = 20, 
    height: int | None = None,
) -> None:
    width = look_back * 3
    height = height or width
    price_height = int(round(height * 0.8))
    volume_height = int(round(height * 0.2))
    
    tickers = df[ticker_col].unique().sort()
    dates = df.select("date").unique().sort("date")

    zip_filename = f"data/images_{look_back}.zip"
    
    # Create a zip file to store all images
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for ticker in tqdm(tickers, desc="Generating images...", position=0):
            # Subset on ticker
            ticker_df = df.filter(pl.col(ticker_col).eq(ticker))
            
            # Get valid periods for ticker
            ticker_start = ticker_df["date"].min()
            ticker_end = ticker_df["date"].max()
            
            ticker_dates = dates.filter(
                pl.col("date").is_between(ticker_start, ticker_end)
            ).sort("date")
            
            ticker_periods = get_period_pairs(ticker_dates, look_back)
            
            # Generate image for each day
            for start, end in tqdm(
                ticker_periods,
                desc=f"Generating images for {ticker}",
                leave=False,
                position=1,
            ):
                # Create a temporary folder for image generation
                temp_image_folder = f"temp_images/{look_back}/{ticker}"
                os.makedirs(temp_image_folder, exist_ok=True)
                
                image_filename = f"{ticker}_{end.strftime('%Y%m%d')}_{look_back}.png"
                image_path = os.path.join(temp_image_folder, image_filename)
                
                # Skip if image already exists in zip
                if any(image_filename in zinfo.filename for zinfo in zipf.filelist):
                    continue
                
                # Subset on period (ensure all dates)
                period_df = ticker_df.filter(pl.col("date").is_between(start, end))
                
                if len(period_df) != look_back:
                    continue
                
                # Generate image components
                ohlc_bars = generate_ohlc_bars(period_df, look_back, price_height)
                ma_line = generate_ma_line(period_df, look_back, price_height)
                volume_bars = generate_volumes_bars(period_df, look_back, volume_height)
                
                # Take max of moving average line and ohlc bars
                image = np.maximum(ohlc_bars, ma_line)
                
                # Stack on top of volume bars
                image = np.vstack([image, volume_bars])
                
                # Put into RGB spaces
                image = image * 255
                assert image.shape == (height, width)
                
                # Save image
                plt.imsave(
                    image_path,
                    image,
                    cmap="grey",
                )
                
                # Add image to zip file
                zipf.write(image_path, arcname=image_filename)
                
                # Remove temporary image file
                os.remove(image_path)
        
        # Remove temporary folders
        os.rmdir(temp_image_folder)
        os.rmdir(os.path.dirname(temp_image_folder))

    print(f"Images have been saved to {zip_filename}")


def generate_yf_images():
    # Parameters
    tickers = ["BRK-A"]
    start_date = date(1993, 1, 1)  # dates are from paper
    end_date = date(2019, 12, 31)
    look_back = 60

    # Data load
    df = (
        du.load_yfinance(tickers, start_date, end_date)
        # Create moving average column
        .with_columns(pl.col("close").rolling_mean(window_size=look_back).alias("ma"))
        # Remove null values
        .drop_nulls("ma")
    )

    # Generate 60 x 60 images
    generate_images(df, look_back=look_back, aspect_ratio=(8, 15))


if __name__ == "__main__":
    # Parameters
    start_date = date(1993, 1, 1)  # dates are from paper
    end_date = date(2019, 12, 31)
    look_back = 20

    df = du.load_daily_crsp(
        start_date=start_date,
        end_date=end_date,
        look_back=look_back
    )

    # df = df.filter(pl.col('permno').eq(14593)).sort('date')
    # df = df.filter(pl.col('permno').ge(86270)).sort(['permno', 'date'])

    generate_images(df, ticker_col='permno', look_back=look_back, height=64)

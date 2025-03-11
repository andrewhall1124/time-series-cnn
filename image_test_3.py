# TODO: Make this work for each day moving forward in dataset.
# TODO: Get CRSP data
# TODO: Rework to work with CRSP data
# TODO: Generate volume bars
# TODO: Generate moving average price
# TODO: Save image as date_ticker_window.png

import data_utils as du
import polars as pl
import matplotlib.pyplot as plt

pl.enable_string_cache()

N_DAYS = 60 # * 3 = width of image
NUM_BINS = N_DAYS * 2 # Height of the image (* num_tickers = height of dataframe)

labels = [f"{i}" for i in range(NUM_BINS)]

tickers = ['AAPL', 'WMT', 'NVDA']
df = du.load_yfinance(tickers)

def get_bulk_images(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df
        # Get last n_days * n_tickers
        .tail(N_DAYS * len(df['ticker'].unique()))
        # Create index column
        .with_columns(
            pl.col('ticker').cum_count().over("ticker").alias("index")     
        )
        # Unpivot OHLC variables
        .unpivot(on=['open', 'high', 'low', 'close'], index=['date', 'index', 'ticker'])
        .sort(['date', 'ticker', 'variable'])
        # Put variable values into bins for the whole sample by ticker
        .with_columns(
            pl.col('value').qcut(NUM_BINS, labels=labels, allow_duplicates=True).over('ticker').alias('bin'),
            pl.lit(1).alias('in_bin')
        )
        .sort('bin')
        # Pivot to bins wide
        .pivot(on='bin', index=['index', 'ticker', 'variable'], values='in_bin')
        .sort('index', 'ticker', 'variable')
        # Concatenate high and low together
        .with_columns(
            pl.col('variable').replace({'high': 'high_low', 'low': 'high_low'})
        )
        # Create x axis column
        .with_columns(
            pl.concat_str(pl.col('index'), pl.col('variable'), separator='_').alias('x')
        )
        .drop('index', 'variable')
        # Unpivot and pivot to x and y axes
        .unpivot(index=['ticker', 'x'], variable_name='bin', value_name='in_bin')
        .pivot(on='x', index=['ticker', 'bin'], values='in_bin', aggregate_function='max')
    )

    return df

def get_image_from_bulk_images(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    df = (
        df
        # Filter to single ticker
        .filter(pl.col('ticker').eq(ticker))
        # Sort y axis by bins descending (high price at top)
        .cast({'bin': pl.Int32})
        .sort('bin', descending=True)
        # Interpolate vertical missing values
        .interpolate()
        .fill_null(0)
        # Reorder x axis to be by date open, high_low, close
        .select(['bin'] + [f"{i + 1}_{var}" for i in range(N_DAYS) for var in ['open', 'high_low', 'close']])
    )

    return df

# Get bulk images
df = get_bulk_images(df)
print(df)

for ticker in tickers:
    # Filter df by ticker and gen image
    sub_df = get_image_from_bulk_images(df, ticker)
    print(sub_df)

    # Plot image 
    plt.imshow(sub_df.drop('bin'), cmap='grey')
    plt.show()
    plt.clf()

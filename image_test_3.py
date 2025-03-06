import data_utils as du
import polars as pl
import numpy as np
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
        .tail(N_DAYS * len(df['ticker'].unique()))
        .with_columns(
            pl.col('ticker').cum_count().over("ticker").alias("index")     
        )
        .unpivot(on=['open', 'high', 'low', 'close'], index=['date', 'index', 'ticker'])
        .sort(['date', 'ticker', 'variable'])
        .with_columns(
            pl.col('value').qcut(NUM_BINS, labels=labels, allow_duplicates=True).over('ticker').alias('bin'),
            pl.lit(1).alias('in_bin')
        )
        .sort('bin')
        .pivot(on='bin', index=['index', 'ticker', 'variable'], values='in_bin')
        .sort('index', 'ticker', 'variable')
        .with_columns(
            pl.col('variable').replace({'high': 'high_low', 'low': 'high_low'})
        )
        .with_columns(
            pl.concat_str(pl.col('index'), pl.col('variable'), separator='_').alias('x')
        )
        .drop('index', 'variable')
        .unpivot(index=['ticker', 'x'], variable_name='bin', value_name='in_bin')
        .pivot(on='x', index=['ticker', 'bin'], values='in_bin', aggregate_function='max')
    )

    return df

def gen_image(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    df = (
        df
        .filter(pl.col('ticker').eq(ticker))
        .cast({'bin': pl.Int32})
        .sort('bin', descending=True)
        .interpolate()
        .fill_null(0)
        .select(['bin'] + [f"{i + 1}_{var}" for i in range(N_DAYS) for var in ['open', 'high_low', 'close']])
    )

    return df

# Get bulk images
df = get_bulk_images(df)
print(df)

for ticker in tickers:
    # Filter df by ticker and gen image
    sub_df = gen_image(df, ticker)
    print(sub_df)

    # Plot image 
    plt.imshow(sub_df.drop('bin'), cmap='grey')
    plt.show()
    plt.clf()

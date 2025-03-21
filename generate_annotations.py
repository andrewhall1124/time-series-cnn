import os
import polars as pl
import data_utils as du
from datetime import date

# Get file paths
file_paths = os.listdir("images/20")
file_paths_df = (
    pl.DataFrame(file_paths, schema={'file_path': pl.String})
    # Seperate columns
    .with_columns(
        pl.col('file_path').str.split("_").alias('parts')
    )
    .with_columns(
        pl.col('parts').list.get(0).cast(pl.Int64).alias('permno'),
        pl.col('parts').list.get(1).str.strptime(pl.Date, "%Y%m%d").alias('date')
    )
    .select('date', 'permno', 'file_path')
    .sort(['permno', 'date'])
)

# Parameters
train_start_date = date(1993, 1, 1)  # dates are from paper
train_end_date = date(2000, 12, 31)
test_start_date = date(2001, 1, 1)
test_end_date = date(2019, 12, 31)

look_back = 20

df = (
    du.load_daily_crsp(train_start_date, test_end_date)
    .with_columns(
        pl.col('close').rolling_mean(window_size=look_back).over('permno').alias('moving_average_price')
    )
    # Future window return
    .with_columns(
        pl.col("return")
        .log1p()
        .rolling_sum(window_size=look_back)
        .over('permno')
        .shift(-look_back)
        .alias(f"return_{look_back}")
    )
    .drop_nulls(subset=[f"return_{look_back}", 'moving_average_price'])
    .sort(['permno', "date"])
    # Label
    .with_columns(
        pl.when(pl.col(f"return_{look_back}").ge(0)).then(1).otherwise(0).alias("label")
    )
    .select(['date', 'permno', 'return_20', 'label'])
)


annotations = (
    df.join(
        file_paths_df,
        on=['date', 'permno'],
        how='inner'
    )
)

# Split annotations
train_annotations = (
    annotations
    # Filter to first 30% of data
    .filter(pl.col('date').is_between(train_start_date, train_end_date))
    .sort(['permno', 'date'])
    .select(
        pl.col('file_path').alias('file'),
        'label'
    )
)
print(train_annotations)
train_annotations.write_csv("data/train_annotations_20.csv")

test_annotations = (
    annotations
    # Filter to second 70% of data
    .filter(pl.col('date').is_between(test_start_date, test_end_date))
    .sort(['permno', 'date'])
    .select(
        pl.col('file_path').alias('file'),
        'label'
    )
)
print(test_annotations)
test_annotations.write_csv("data/test_annotations_20.csv")
import os
import polars as pl
import data_utils as du
from datetime import date

image_folder = "/home/andrew/Data/images/20"

# Get file paths
permnos = os.listdir(image_folder)
file_paths = [file for permno in permnos for file in os.listdir(f"{image_folder}/{permno}")]
file_paths_df = (
    pl.DataFrame(file_paths, schema={"file_path": pl.String})
    # Seperate columns
    .with_columns(pl.col("file_path").str.split("_").alias("parts"))
    .with_columns(
        pl.col("parts").list.get(0).cast(pl.Int64).alias("permno"),
        pl.col("parts").list.get(1).str.strptime(pl.Date, "%Y%m%d").alias("date"),
    )
    .select("date", "permno", "file_path")
    .sort(["permno", "date"])
)

print(file_paths_df)

# Parameters
train_start_date = date(1993, 1, 1)  # dates are from paper
train_end_date = date(2000, 12, 31)
test_start_date = date(2001, 1, 1)
test_end_date = date(2019, 12, 31)

look_back = 20

df = (
    du.load_daily_crsp_annotations(
        start_date=train_start_date, end_date=test_end_date
    )
    .with_columns(
        pl.col("ret")
        .log1p()
        .rolling_sum(window_size=look_back)
        .shift(-look_back)
        .over("permno")
        .alias(f"return_{look_back}")
    )
    .drop_nulls(f"return_{look_back}")
    .with_columns(
        pl.when(pl.col(f"return_{look_back}").ge(0)).then(1).otherwise(0).alias("label")
    )
    .sort(['permno', 'date'])
    .select('date', 'permno', 'ticker', 'prc', 'ret', f'return_{look_back}', 'label')
)

print(df)

annotations = (
    df.join(
        file_paths_df,
        on=['date', 'permno'],
        how='inner'
    )
)

print(annotations)

# Split annotations
train_annotations = (
    annotations
    # Filter to first 30% of data
    .filter(pl.col('date').is_between(train_start_date, train_end_date))
    .sort(['permno', 'date'])
    .select(
        pl.col('permno').alias('folder'),
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
        pl.col('permno').alias('folder'),
        pl.col('file_path').alias('file'),
        'label'
    )
)
print(test_annotations)
test_annotations.write_csv("data/test_annotations_20.csv")

backtest_df = (
    annotations
    .filter(pl.col('date').is_between(test_start_date, test_end_date))
    .sort(['permno', 'date'])
)

print(backtest_df)
backtest_df.write_parquet("data/backtest_starter.parquet")
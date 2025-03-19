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
print(file_paths_df)

# Parameters
start_date = date(2010, 1, 1)  # dates are from paper
end_date = date(2019, 12, 31)
look_back = 20

df = (
    du.load_daily_crsp(start_date, end_date)
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

print(df)

annotations = (
    df.join(
        file_paths_df,
        on=['date', 'permno'],
        how='inner'
    )
    .select(
        pl.col('file_path').alias('file'),
        'label'
    )
)

annotations.write_csv("data/annotations_20.csv")
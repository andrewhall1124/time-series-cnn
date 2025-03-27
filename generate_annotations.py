import os
import zipfile
import polars as pl
import data_utils as du
from datetime import date

def extract_image_info_from_zip(zip_path, look_back):
    """
    Extract image information directly from a zip file without extracting all images.
    """
    file_paths = []
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Get all filenames in the zip
        for filename in zipf.namelist():
            file_paths.append(filename)
    
    file_paths_df = (
        pl.DataFrame(file_paths, schema={"file_path": pl.String})
        # Separate columns
        .with_columns(pl.col("file_path").str.split("_").alias("parts"))
        .with_columns(
            pl.col("parts").list.get(0).cast(pl.Int64).alias("permno"),
            pl.col("parts").list.get(1).str.strptime(pl.Date, "%Y%m%d").alias("date"),
        )
        .select("date", "permno", "file_path")
        .sort(["permno", "date"])
    )
    
    return file_paths_df

# Parameters
train_start_date = date(1993, 1, 1)  # dates are from paper
train_end_date = date(2000, 12, 31)
test_start_date = date(2001, 1, 1)
test_end_date = date(2019, 12, 31)
look_back = 20

# Path to the zipped images
zip_path = "images_20.zip"

# Extract file paths from zip
file_paths_df = extract_image_info_from_zip(zip_path, look_back)
print("File paths from zip:")
print(file_paths_df)

# Load and process data
df = (
    du.load_daily_crsp(
        start_date=train_start_date, end_date=test_end_date, look_back=look_back
    )
    .with_columns(
        pl.col("close")
        .pct_change()
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
    .select('date', 'permno', f'return_{look_back}', 'label')
)
print("\nProcessed data:")
print(df)

# Join annotations
annotations = (
    df.join(
        file_paths_df,
        on=['date', 'permno'],
        how='inner'
    )
)
print("\nAnnotations:")
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
print("\nTrain annotations:")
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
print("\nTest annotations:")
print(test_annotations)
test_annotations.write_csv("data/test_annotations_20.csv")
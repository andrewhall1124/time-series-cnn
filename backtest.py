import polars as pl
import numpy as np
import polars_ds as pds
import polars.selectors as cs

MODEL_NAME = '24_epochs_model_1'

cnn_data = (
    pl.scan_csv("data/test_annotations_20.csv")
    .with_columns(
        pl.col('file').str.split("_").list.get(0).alias('permno'),
        pl.col('file').str.split("_").list.get(1).alias('date')
    )
    .with_columns(
        pl.col('permno').cast(pl.Int32),
        pl.col('date').str.strptime(pl.Date, "%Y%m%d")
    )
    .with_columns(
        pl.read_parquet(f'results/{MODEL_NAME}_test_results.parquet').to_series()
    )
    .with_columns(
        pl.col('probability').round().cast(pl.Int64).alias('y_hat')
    )
    .with_columns(
        pl.col('label').eq(pl.col('y_hat')).alias("correct")
    )
)

baseline_accuracy = cnn_data.collect()['label'].mean()
accuracy = cnn_data.collect()['correct'].mean()

print("-"*10 + "Accuracies" + "-"*10)
print(f"Basline Accuracy: {baseline_accuracy:.2%}")
print(f"Test Accuracy: {accuracy:.2%}\n")

df = (
    pl.scan_parquet("data/backtest_starter.parquet")
    .join(
        cnn_data,
        on=['date', 'permno'],
        how='right'
    )
    .select('date', 'permno', 'ticker', 'prc', 'ret', 'return_20', 'probability')
    .collect()
)

print("DATASET")
print(df)

labels = [str(i) for i in range(10)]
portfolios = (
    df
    .with_columns(
        pl.col('probability').qcut(10, labels=labels).cast(pl.Int8).over('date').alias('bin')
    )
    .group_by(['date', 'bin'])
    .agg(
        pl.col('return_20').mean()
    )
    .sort('bin')
    .pivot(on='bin', index='date', values='return_20')
    .sort('date')
    .with_columns(pl.col('9').sub(pl.col('0')).alias('spread'))
    .gather_every(20)
)

print("PORTFOLIO MONTHLY RETURNS")
print(portfolios)

results = (
    portfolios
    .unpivot(index='date', variable_name='bin', value_name='return')
    .group_by('bin')
    .agg(
        pl.col("return").mean().mul(12).alias('mean'),
        pl.col('return').std().mul(np.sqrt(12)).alias('std'),
        pds.ttest_1samp('return', pop_mean=0, alternative='two-sided').alias('tstat')
    )
    .unnest('tstat')
    .rename({'statistic': 'tstat'})
    .with_columns(pl.col('mean').truediv(pl.col('std')).alias('sharpe'))
    .sort('bin')
    .drop('bin')
    .transpose(include_header=True, column_names=[*labels, 'spread'], header_name='statistic')
    .with_columns(
        cs.numeric().round(4)
    )
)

print(results)
results.write_csv(f"results/{MODEL_NAME}_backtest_table.csv")

cummulative_returns = (
    portfolios
    .with_columns(pl.col([*labels, 'spread']).log1p().cum_sum().mul(100))
    .unpivot(index='date', variable_name='bin', value_name='cummulative_return')
)

sharpe = portfolios['spread'].mean() / portfolios['spread'].std() * np.sqrt(12)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

sns.lineplot(cummulative_returns.to_pandas(), x='date', y='cummulative_return', hue='bin', palette='coolwarm')

plt.title(f"CNN Probabilities Backtest ({sharpe:.2f})")
plt.xlabel(None)
plt.ylabel("Cummulative Sum Return (%)")

plt.legend()

plt.savefig(f"results/{MODEL_NAME}_backtest_plot.png", dpi=300)


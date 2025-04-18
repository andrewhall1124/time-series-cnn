# TODO

- Backtesting code
- Adapt code to work with CRSP data
- Run model on all securities
- Hyperparameter tuning
  - Architecture
  - Dropout
  - Batch size
  - Optimizer
  - Normalization time range
- Add embed for security
- Better loss function
  - https://www.perplexity.ai/search/what-are-best-practices-for-us-zAPeQaGbTLOf5AF8L.Zkxg
- RL?
  - https://spinningup.openai.com/en/latest/algorithms/td3.html

# General Notes

- Walmart
  - Buy-and-Hold gets .82 sharpe
  - Theoretical optimal gets 8.64 sharpe
  - SVC baseline gets .83 sharpe
  - HistGradientBoostingClassifier gets .44 sharpe

# Training Log

## More Tuning

- Norm over 365 days doubled returns
- 3 day label window bumps sharpe up to .62, baseline sharpe up to .82
- 56 max period bumps sharpe up to .94

## Adding additional tickers

- Added both Walmart and Apple starting in the 80s...Sharpe was really bad, negative returns. Testing period was 2016-2024
- Starting in 2010 was better, got .78 sharpe.
- Addded 10 stocks...not great:

```
Balanced accuracy score: 0.8040
              precision    recall  f1-score   support

         BUY       0.43      0.86      0.57       957
        SELL       0.55      0.83      0.66      1151
        HOLD       0.94      0.71      0.81      6578

    accuracy                           0.75      8686
   macro avg       0.64      0.80      0.68      8686
weighted avg       0.83      0.75      0.76      8686

Confusion matrix:
[[ 827   17  113]
 [  14  961  176]
 [1104  785 4689]]
Annualized return: 2.65%
Sharpe ratio: 0.34
Buy+Hold baseline annualized return: 5.65%
Buy+Hold baseline Sharpe ratio: 0.47
True baseline annualized return: 1.71%
True baseline Sharpe ratio: 0.25
```
Model is doing what it can, but I think we're hitting limitations of data/loss function

## Fixed Param Tuning on 53e5ac2d3618fe83f0b516f83a044d8196930976

After the prior param tuning, I realized I failed to set the seed, which makes my results non-deterministic. I fixed that, and re-ran a lot of param tuning.

- Batch Size
  - 256 => .61 sharpe
  - 128 => .85 sharpe
- Warmup
  - 0 => .85 sharpe
  - 10 => .63 sharpe
  - 100 => .43 sharpe
- LR
  - 5e-3 => .85 sharpe
  - 1e-3 => .85 sharpe
  - 5e-4 => .87 sharpe
  - 1e-4 => .94 sharpe
  - 5e-5 => .48 sharpe
- Architecture
  - Padding to keep sizes => .64 sharpe
  - 16x16 images with modifications => .44 sharpe
  - 14x14 => .5 sharpe
  - 20x20 => .45 sharpe
  - Extra Linear Layer => .57 sharpe
- Feature Selection
  - Both => .94 sharpe
  - Only F classif => .94 sharpe
  - Only Mutual Info => .62 sharpe
- Misc
  - AdamW => 0.90 sharpe
  - SeLU => .42 sharpe
  - LeakyRELU => .75 sharpe
  - PReLU => .69 sharpe
  - Basic 2 layer linear => 0.70
  - Linear Hidden Size 5 => .70
  - Double channels + hidden => .67 sharpe
- Label Window
  - 11 => .64 sharpe (nooooo....looks like re-creating the data toasted our sharpe)
  - 5 => .23 sharpe
  - 15 => .23 sharpe

## Some Param Tuning on 5326e53a0653aa1ac07a2d11097dd0faeab3ab29

- Batch Size
  - 128 => .62 sharpe
  - 256 => .64 sharpe
  - 512 => .57 sharpe
- Warmup
  - 0 => .65 sharpe
- Patience
  - 5 => .73 sharpe
  - 1 => .76 sharpe
- LR
  - 1e-2 => .49 sharpe
  - 1e-4 => .74 sharpe
  - 5e-4 => .34 sharpe
  - 5e-3 => .76 sharpe
- Dropout
  - .15 => .76 sharpe
  - .05 => .88 sharpe
  - 0.0 => .73 sharpe
  - .1 => .68 sharpe
  - .03 => .8 sharpe
- BatchNorm instead of Dropout: .92 sharpe

## New Features

- Commit Hash: 5326e53a0653aa1ac07a2d11097dd0faeab3ab29
- Test Acc: 0.7957
- Full Loss

```
              precision    recall  f1-score   support

         BUY       0.28      0.82      0.41       119
        SELL       0.34      0.81      0.48       125
        HOLD       0.97      0.76      0.85      1897

    accuracy                           0.77      2141
   macro avg       0.53      0.80      0.58      2141
weighted avg       0.89      0.77      0.81      2141
```

- Confusion Matrix

```
[[  97    0   22]
 [   0  101   24]
 [ 254  194 1449]]
```

- Annualized return: 5.01%
- Sharpe ratio: 0.40

## First Paper Replication

- Commit Hash: cb6308e04613d0f2af8eff7330879814d2328cd1
- Test Accuracy: 0.6565
- Full Loss

```
              precision    recall  f1-score   support

         BUY       0.14      0.78      0.24       152
        SELL       0.15      0.80      0.25       154
        HOLD       0.94      0.39      0.55      2303

    accuracy                           0.44      2609
   macro avg       0.41      0.66      0.35      2609
weighted avg       0.85      0.44      0.51      2609
```

- Confusion Matrix:

```
[[119   4  29]
 [  7 123  24]
 [711 699 893]]
```

- Annualized return: 4.01%
- Sharpe ratio: 0.35

Faithful replication of the original paper

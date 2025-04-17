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
- Better loss function
  - https://www.perplexity.ai/search/what-are-best-practices-for-us-zAPeQaGbTLOf5AF8L.Zkxg
- RL?
  - https://spinningup.openai.com/en/latest/algorithms/td3.html

# General Notes

- Walmart Buy-and-Hold gets .82 sharpe
- Theoretical optimal Walmart gets 8.64 sharpe

# Training Log

## Some Param Tuning on 5326e53a0653aa1ac07a2d11097dd0faeab3ab29

- Batch Size
  - 128 => .62 sharpe
  - 256 => .64 sharpe
  - 512 => .57 sharpe
- Warmup
  - 0.00 => .65 sharpe
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

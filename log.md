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

# General Notes

- Walmart Buy-and-Hold gets .82 sharpe
- Theoretical optimal Walmart gets 8.64 sharpe

# Training Log

## Some Param Tuning

- Date: April 17, 2025
- .55 Sharpe with 250 day norm window

## First Paper Replication

- Date: April 17, 2025
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

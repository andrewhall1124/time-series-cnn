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

## First Paper Replication

- Date: April 17, 2025
- Commit Hash: cb6308e04613d0f2af8eff7330879814d2328cd1
- Test Accuracy: 0.6602
- Full Loss

```
              precision    recall  f1-score   support

         BUY       0.14      0.78      0.24       152
        SELL       0.16      0.77      0.26       154
        HOLD       0.95      0.43      0.59      2303

    accuracy                           0.47      2609
   macro avg       0.42      0.66      0.36      2609
weighted avg       0.86      0.47      0.55      2609
```

- Confusion Matrix:

```
[[119   4  29]
 [ 11 119  24]
 [692 630 981]]
```

- Sharpe Ratio: .53
- Annualized Return: 7.49%

Faithful replication of the original paper

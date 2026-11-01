# Model Performance Comparison on Untouched Evaluation Data

| Model               | Balancing_Technique     |   Precision |   Recall |   F1_Score |   PR_AUC |   ROC_AUC |   TP |   FP |   FN |    TN |
|:--------------------|:------------------------|------------:|---------:|-----------:|---------:|----------:|-----:|-----:|-----:|------:|
| Random Forest       | None                    |   0.959459  | 0.747368 |  0.840237  | 0.792422 |  0.966882 |   71 |    3 |   24 | 56648 |
| Random Forest       | class_weight='balanced' |   0.886076  | 0.736842 |  0.804598  | 0.778896 |  0.954455 |   70 |    9 |   25 | 56642 |
| Random Forest       | SMOTE                   |   0.675676  | 0.789474 |  0.728155  | 0.790873 |  0.980603 |   75 |   36 |   20 | 56615 |
| XGBoost             | scale_pos_weight        |   0.649573  | 0.8      |  0.716981  | 0.789777 |  0.972419 |   76 |   41 |   19 | 56610 |
| Logistic Regression | None                    |   0.863636  | 0.6      |  0.708075  | 0.700293 |  0.955384 |   57 |    9 |   38 | 56642 |
| Random Forest       | RandomUnderSampler      |   0.075437  | 0.863158 |  0.138748  | 0.701368 |  0.971537 |   82 | 1005 |   13 | 55646 |
| Logistic Regression | class_weight='balanced' |   0.0548579 | 0.873684 |  0.103234  | 0.679321 |  0.962203 |   83 | 1430 |   12 | 55221 |
| Logistic Regression | SMOTE                   |   0.0522341 | 0.873684 |  0.0985748 | 0.681752 |  0.959976 |   83 | 1506 |   12 | 55145 |
| Logistic Regression | RandomUnderSampler      |   0.0519724 | 0.873684 |  0.0981087 | 0.612723 |  0.957459 |   83 | 1514 |   12 | 55137 |

### Metric Interpretations:
- **Precision**: Proportion of flagged transactions that were genuine fraud (TP / (TP + FP)). High precision prevents false alarms.
- **Recall**: Proportion of actual fraudulent transactions detected (TP / (TP + FN)). High recall minimizes financial fraud leak.
- **F1-Score**: Harmonic mean of Precision and Recall. Balances both costs.
- **PR-AUC**: Area under the Precision-Recall curve. The single most honest metric for severely imbalanced fraud detection.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

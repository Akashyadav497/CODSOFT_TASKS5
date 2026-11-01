# Data Audit Report: Credit Card Fraud Detection
**Generated on:** 2026-10-23  
**Project:** CodSoft Data Science Internship - Task 5  

---

## 1. File Inventory and Role Assignment
- **Primary Raw Data File:** `data/creditcard.csv` (Extracted from repository archive `creditcard.csv (1).zip`)
- **Dataset Role Decision:** **Case C**
  - *Justification:* Exactly one comprehensive dataset file exists containing historical credit card transactions and ground truth fraud labels (`Class`). Per task specification Case C, we partition this data into a stratified training set (80%) and test set (20%) using `random_state=42`. All scalers, preprocessors, and balancing techniques are strictly fitted only on the training portion, and the test portion is retained untouched for final uncorrupted evaluation.
- **Original File Immutability:** `data/creditcard.csv` is treated as strictly read-only. Processed clean artifacts are saved to `data/processed/`.

---

## 2. Dataset Dimensions and Structure
- **Total Rows (Raw):** 284,807
- **Total Columns:** 31
- **Duplicate Rows:** 1,081
- **Total Rows (After Deduplication):** 283,726
- **Missing / Null Values:** 0 across all 31 columns.

---

## 3. Detected Target Column and Class Distribution

The target variable is identified with certainty as **`Class`**:
- `0`: Genuine / Normal transaction
- `1`: Fraudulent transaction

### Real Class Counts and Percentages

#### Raw Data (284,807 rows):
| Class | Label | Transaction Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0** | Genuine | 284,315 | 99.8273% |
| **1** | Fraudulent | 492 | 0.1727% |
| **Total** | | 284,807 | 100.0000% |

#### Cleaned / Deduplicated Data (283,726 rows):
| Class | Label | Transaction Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0** | Genuine | 283,253 | 99.8333% |
| **1** | Fraudulent | 473 | 0.1667% |
| **Total** | | 283,726 | 100.0000% |

> **Key Observation:** Severe class imbalance exists with fraud representing approximately ~0.17% (1 in every ~600 transactions). Standard metrics such as simple Accuracy are completely uninformative (a naive null predictor predicting 0 yields ~99.83% accuracy while catching 0% of fraud). Precision, Recall, F1-Score, and PR-AUC are vital.

---

## 4. Column Schema and Data Types

| Column Name | Data Type | Missing Count | Description / Role |
| :--- | :--- | :--- | :--- |
| `Time` | `float64` | 0 | Seconds elapsed between this transaction and the first transaction in the dataset. |
| `V1` - `V28` | `float64` | 0 | 28 principal components obtained via PCA transformation (confidential transaction features). |
| `Amount` | `float64` | 0 | Transaction amount in currency units. Skewed distribution with high variance. |
| `Class` | `int64` | 0 | Ground truth target (0 = Genuine, 1 = Fraud). |

---

## 5. First 5 Rows Sample

```text
   Time        V1        V2        V3  ...       V27       V28  Amount  Class
0   0.0 -1.359807 -0.072781  2.536347  ...  0.133558 -0.021053  149.62      0
1   0.0  1.191857  0.266151  0.166480  ... -0.008983  0.014724    2.69      0
2   1.0 -1.358354 -1.340163  1.773209  ... -0.055353 -0.059752  378.66      0
3   1.0 -0.966272 -0.185226  1.792993  ...  0.062723  0.061458  123.50      0
4   2.0 -1.158233  0.877737  1.548718  ...  0.219422  0.215153   69.99      0
```

---

## 6. Preprocessing Strategy
1. **Deduplication:** Remove 1,081 identical duplicate transaction records to avoid over-weighting repeated observations.
2. **Feature Engineering:**
   - Derive cyclical / temporal features from `Time`: `Hour` (`(Time // 3600) % 24`) to capture diurnal fraud patterns.
   - Robust scaling / log-transformation on `Amount` (heavy right-skew, ranging from 0 to >25,000).
3. **Split-First Protocol:**
   - 80/20 Stratified Split (`random_state=42`).
   - Scalers fit on Training split only.
   - Resampling (SMOTE, RandomUnderSampler, Class Weighting) applied only to Training set.

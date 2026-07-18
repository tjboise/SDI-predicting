# SDI Prediction — Methodology Notes

## Table of Contents
- [Dataset](#dataset)
- [Approach 1: Segment-Aware Longitudinal Prediction](#approach-1-segment-aware-longitudinal-prediction)
  - [Sample Construction](#sample-construction)
  - [Train / Test Split](#train--test-split)
  - [Results](#results-approach-1)
- [Approach 2: Random Split — Full Model Comparison](#approach-2-random-split--full-model-comparison)
  - [Features](#features)
  - [Models](#models)
  - [Results](#results-approach-2)
- [Key Findings](#key-findings)

---

## Dataset

**File:** `modeling_dataset.xlsx`  
**Records:** 1,485 rows · 306 unique road segments · 9 columns  
**Target:** `SDI` (Surface Distress Index, 0–5, higher = better condition)

| Column | Description |
|---|---|
| `SubsectionKey` | Road segment ID |
| `Family` | Pavement structure type (4 categories) |
| `Age` | Years since pavement was laid |
| `SDI` | Surface Distress Index (prediction target) |
| `Existing AC after milling (in)` | Remaining AC thickness after milling |
| `attr_2way_AADTT` | Annual Average Daily Truck Traffic (both directions) |
| `Existing AC material` | Asphalt performance grade (e.g. 64-22, 76-22) |
| `is_PG76` | Binary flag: whether PG76 modified asphalt was used |
| `attr_TotalOverlay_in` | Total overlay thickness (inches) |

Each segment has 4–9 observations recorded at different (non-consecutive) ages.  
`Existing AC material` and `is_PG76` are excluded from all models — Random Forest feature importance confirmed their contribution is negligible (< 0.4% combined).

---

## Approach 1: Segment-Aware Longitudinal Prediction

The goal here is to predict a segment's **next observed SDI** given its known history, without leaking future observations into training.

### Sample Construction

For each segment, observations are sorted by `Age`. The **last observation** is held out as the prediction target. All earlier observations are summarised into a single feature row:

| Feature | Construction |
|---|---|
| `sdi_first` | SDI at the earliest observed age |
| `age_first` | Earliest observed age |
| `sdi_last_known` | SDI at the most recent observation before the target |
| `age_last_known` | Age at the most recent known observation |
| `target_age` | Age of the held-out observation (what we predict) |
| `delta_age` | `target_age − age_last_known` (forecast horizon, mean = 1.7 yrs) |
| `total_delta` | `target_age − age_first` |
| `sdi_slope` | `(sdi_last_known − sdi_first) / (age_last_known − age_first)` |
| `n_obs_known` | Number of historical observations available |
| Static features | `Family`, `AADTT`, `Overlay`, `Milling` |

This yields **306 prediction samples** (one per segment).

### Train / Test Split

Rows are split **by segment** using `GroupKFold(n_splits=5)`. A segment's data never appears in both training and test folds. Random row-level splitting would allow the model to see earlier observations of a test segment during training, artificially inflating R².

### Results (Approach 1)

Naive baseline — predict `sdi_last_known` unchanged: R² = **0.41**, RMSE = **0.81**

| Model | CV R² | RMSE |
|---|---|---|
| Linear Regression | 0.777 | 0.494 |
| Ridge Regression | 0.778 | 0.493 |
| Random Forest | 0.775 | 0.494 |
| Gradient Boosting | 0.753 | 0.518 |

**Feature importance (Random Forest):** `sdi_last_known` accounts for 76.7% of importance. Engineering features contribute < 25% combined, meaning the dominant signal is the segment's most recent known condition.

---

## Approach 2: Random Split — Full Model Comparison

Standard 5-fold random split across all 1,485 rows. Models are compared on a common feature set without using historical SDI values, making this applicable to road segments with no prior observations.

**Script:** `model_comparison.py`

### Features

| Feature | Description |
|---|---|
| `Family_encoded` | Pavement structure type (label-encoded) |
| `attr_2way_AADTT` | Annual Average Daily Truck Traffic |
| `attr_TotalOverlay_in` | Total overlay thickness (in) |
| `Existing AC after milling (in)` | Remaining AC thickness |
| `Age` | Pavement age (years) |

### Models

Eight models are evaluated, ranging from classical ML to physics-informed neural networks.

**Classical ML**
- **Linear Regression / Ridge** — baseline linear models
- **Random Forest** — bagged decision trees, robust to noise
- **Gradient Boosting** — sequential boosting, typically highest performance among tree models
- **SVR** — support vector regression with RBF kernel

**Neural Networks**
- **ANN (pure data)** — standard fully-connected network, MSE loss only
- **ANN → (a, b, c) → formula** — network predicts Weibull decay parameters from static features; SDI computed as `a·exp(-(t/b)^c)`. No direct access to true SDI during forward pass.
- **ANN dual-loss (new)** — two-stage hybrid:
  - Stage 1 (ANN₁): static features → `(a, b, c)`
  - Stage 2 (ANN₂): `(a, b, c, t)` → SDI
  - Loss = `0.5 × MSE(pred, true_SDI) + 0.5 × MSE(pred, a·exp(-(t/b)^c))`

The empirical decay formula used as a physics prior is:

```
SDI(t) = a · exp(-(t / b)^c)
```

### Results (Approach 2)

5-fold CV, random split:

| Model | CV R² | ± std | RMSE |
|---|---|---|---|
| **Gradient Boosting** | **0.6448** | 0.0476 | 0.5842 |
| Random Forest | 0.6265 | 0.0544 | 0.5986 |
| SVR | 0.5873 | 0.0467 | 0.6306 |
| ANN → (a,b,c) → formula | 0.5317 | 0.0294 | 0.6719 |
| ANN dual-loss | 0.5085 | 0.0252 | 0.6886 |
| ANN (pure data) | 0.4746 | 0.0409 | 0.7116 |
| Ridge | 0.3303 | 0.0070 | 0.8040 |
| Linear Regression | 0.3303 | 0.0070 | 0.8040 |

---

## Key Findings

- **Gradient Boosting achieves the best R² (0.645)** among models using only static features + age.
- **Linear models (R² ≈ 0.33)** perform poorly, confirming a nonlinear relationship between features and SDI.
- **ANN → (a,b,c) → formula (R² = 0.53)** outperforms the pure-data ANN (R² = 0.47), suggesting the Weibull structure provides a useful inductive bias.
- **The dual-loss ANN (R² = 0.51)** sits between the two ANN variants; the formula constraint helps regularise but the data signal is still dominant.
- **The R² ceiling (~0.65) reflects the limits of the available features.** Much of the variance in SDI is driven by factors not captured in this dataset (construction quality, subgrade condition, local climate, maintenance history). The segment-aware approach (Approach 1) reaches R² = 0.78 precisely because `sdi_last_known` encodes this latent per-segment information.

# SDI Prediction — Methodology Notes

## Table of Contents
- [Dataset](#dataset)
- [Approach 1: Segment-Level Split](#approach-1-segment-level-split)
  - [Sample Construction](#sample-construction)
  - [Train / Test Split](#train--test-split)
  - [Statistical Feature Models](#statistical-feature-models)
  - [Sequence Models](#sequence-models)
  - [Results](#results-approach-1)
- [Approach 2: Random Split](#approach-2-random-split)
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

## Approach 1: Segment-Level Split

The goal is to predict a segment's **next observed SDI** given its known history. All models in this approach share the same data split: segments are never shared between training and test folds.

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

---

### Statistical Feature Models

Classical ML and ANN trained on the hand-crafted feature row described above.

Naive baseline — predict `sdi_last_known` unchanged: R² = **0.41**, RMSE = **0.81**

| Model | CV R² | ± std | RMSE |
|---|---|---|---|
| **Random Forest** | **0.7754** | 0.0500 | 0.4935 |
| Ridge | 0.7672 | 0.0470 | 0.5033 |
| Linear Regression | 0.7669 | 0.0470 | 0.5037 |
| Gradient Boosting | 0.7518 | 0.0504 | 0.5190 |
| SVR | 0.7392 | 0.0890 | 0.5270 |
| ANN | 0.7131 | 0.0729 | 0.5541 |

**Feature importance (Random Forest):** `sdi_last_known` accounts for 76.7% of importance. Engineering features contribute < 25% combined, meaning the dominant signal is the segment's most recent known condition.

Note: Gradient Boosting ranks first in Approach 2 (random split) but falls behind Random Forest here. With only 306 samples and segment-level splitting, boosting's sequential nature makes it more sensitive to fold composition.

---

### Sequence Models

Instead of summarising history into hand-crafted statistics, sequence models consume the raw ordered observations directly. Each time step is a vector `[age_t / 15, SDI_t / 5, Family, AADTT, Overlay, Milling]`. Sequences have variable length (3–8 steps); shorter ones are zero-padded and packed before the RNN layer.

**Script:** `segment_sequence_models.py`

| Model | Description |
|---|---|
| **RNN** | Vanilla recurrent network, sequence → hidden state → SDI |
| **GRU** | Gated Recurrent Unit, better gradient flow than RNN |
| **LSTM** | Long Short-Term Memory, standard sequence baseline |
| **LSTM → (a,b,c) → formula** | LSTM hidden state predicts Weibull parameters; SDI = `a·exp(-(t/b)^c)` |
| **LSTM + Weibull dual loss** | LSTM predicts SDI directly; loss = `0.5×MSE(pred, true) + 0.5×MSE(pred, weibull(t,a,b,c))` where `a,b,c` are globally fitted on training segments each fold |
| **LSTM + sdi_last_known + delta_age** | LSTM hidden state concatenated with `sdi_last_known` and `delta_age` before the output head, explicitly providing the most informative signal |

---

### Results (Approach 1)

All models, segment-level GroupKFold, 5 folds:

| Model | Type | CV R² | ± std | RMSE |
|---|---|---|---|---|
| **Random Forest** | Stat. features | **0.7754** | 0.0500 | 0.4935 |
| Ridge | Stat. features | 0.7672 | 0.0470 | 0.5033 |
| Linear Regression | Stat. features | 0.7669 | 0.0470 | 0.5037 |
| Gradient Boosting | Stat. features | 0.7518 | 0.0504 | 0.5190 |
| SVR | Stat. features | 0.7392 | 0.0890 | 0.5270 |
| ANN | Stat. features | 0.7131 | 0.0729 | 0.5541 |
| LSTM + sdi_last_known + delta_age | Sequence | 0.6682 | 0.0473 | 0.6017 |
| LSTM → (a,b,c) → formula | Sequence | 0.6051 | 0.0631 | 0.6586 |
| RNN | Sequence | 0.4160 | 0.1430 | 0.7958 |
| GRU | Sequence | 0.4078 | 0.1022 | 0.8028 |
| LSTM | Sequence | 0.3850 | 0.1518 | 0.8151 |
| LSTM + Weibull dual loss | Sequence | 0.0911 | 0.0419 | 1.0003 |

---

## Approach 2: Random Split

Standard 5-fold random split across all 1,485 rows. No historical SDI is used — models receive only static features + age, making this applicable to road segments with no prior observations.

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

**Classical ML**
- **Linear Regression / Ridge** — baseline linear models
- **Random Forest** — bagged decision trees, robust to noise
- **Gradient Boosting** — sequential boosting, typically highest performance among tree models
- **SVR** — support vector regression with RBF kernel

**Neural Networks**
- **ANN (pure data)** — standard fully-connected network, MSE loss only
- **ANN → (a, b, c) → formula** — network predicts Weibull decay parameters from static features; SDI computed as `a·exp(-(t/b)^c)`
- **ANN dual-loss** — two-stage hybrid:
  - Stage 1 (ANN₁): static features → `(a, b, c)`
  - Stage 2 (ANN₂): `(a, b, c, t)` → SDI
  - Loss = `0.5 × MSE(pred, true_SDI) + 0.5 × MSE(pred, a·exp(-(t/b)^c))`

The empirical decay formula used as a physics prior:

```
SDI(t) = a · exp(-(t / b)^c)
```

### Results (Approach 2)

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

- **Random Forest + statistical features + segment split (R² = 0.775)** is the best overall approach. The dominant predictor is `sdi_last_known` (76.7% feature importance) — the most recent known condition is by far the strongest signal.
- **Gradient Boosting (R² = 0.645)** is the best model when no historical SDI is available (Approach 2, random split).
- **Plain sequence models (RNN/GRU/LSTM, R² ≈ 0.39–0.42)** underperform with only 306 segments — too few samples to reliably learn temporal patterns, with high fold-to-fold variance (std up to 0.15).
- **LSTM + sdi_last_known + delta_age (R² = 0.668)** is the best sequence model. Explicitly concatenating the most recent SDI and forecast horizon at the output head confirms the bottleneck was not model capacity but the difficulty of discovering the most informative time step from the sequence alone.
- **LSTM → (a,b,c) → formula (R² = 0.605)** shows the Weibull structure provides a useful inductive bias even with limited data.
- **LSTM + Weibull dual loss (R² = 0.091)** fails — a globally-fitted formula is a poor anchor for individual segment predictions, and the conflicting gradients destabilise training.
- **The R² ceiling for Approach 2 (~0.65)** reflects missing information: construction quality, subgrade condition, local climate, and maintenance history are not in the dataset.

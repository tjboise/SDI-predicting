# SDI Prediction — Methodology Notes

## Table of Contents
- [Dataset](#dataset)
- [Approach 1: Segment-Aware — Statistical Features](#approach-1-segment-aware--statistical-features)
  - [Sample Construction](#sample-construction)
  - [Train / Test Split](#train--test-split)
  - [Results](#results-approach-1)
- [Approach 2: Segment-Aware — Sequence Models](#approach-2-segment-aware--sequence-models)
  - [Sequence Input Structure](#sequence-input-structure)
  - [Models](#models-approach-2)
  - [Results](#results-approach-2)
- [Approach 3: Random Split — Full Model Comparison](#approach-3-random-split--full-model-comparison)
  - [Features](#features)
  - [Models](#models-approach-3)
  - [Results](#results-approach-3)
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

## Approach 1: Segment-Aware — Statistical Features

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

| Model | CV R² | ± std | RMSE |
|---|---|---|---|
| **Random Forest** | **0.7754** | 0.0500 | 0.4935 |
| Ridge | 0.7672 | 0.0470 | 0.5033 |
| Linear Regression | 0.7669 | 0.0470 | 0.5037 |
| Gradient Boosting | 0.7518 | 0.0504 | 0.5190 |
| SVR | 0.7392 | 0.0890 | 0.5270 |
| ANN | 0.7131 | 0.0729 | 0.5541 |

**Feature importance (Random Forest):** `sdi_last_known` accounts for 76.7% of importance. Engineering features contribute < 25% combined, meaning the dominant signal is the segment's most recent known condition.

Note: Gradient Boosting, which ranks first in Approach 3 (random split), falls behind Random Forest here. With only 306 samples and segment-level splitting, boosting's sequential nature makes it more sensitive to fold composition.

---

## Approach 2: Segment-Aware — Sequence Models

Instead of summarising history into hand-crafted statistics, sequence models consume the raw ordered observations directly.  
Each time step is a vector `[age_t / 15, SDI_t / 5, Family, AADTT, Overlay, Milling]`. Sequences have variable length (3–8 steps); shorter ones are zero-padded and packed before the RNN layer.

**Script:** `segment_sequence_models.py`

### Sequence Input Structure

| Dimension | Content |
|---|---|
| `age_t / 15` | Normalised age at step t |
| `SDI_t / 5` | Normalised SDI at step t |
| Static features (4) | Family, AADTT, Overlay, Milling — same value repeated at each step |

Target: SDI at the held-out last observation. Split: `GroupKFold(n_splits=5)` by segment.

### Models (Approach 2)

| Model | Description |
|---|---|
| **RNN** | Vanilla recurrent network, sequence → hidden state → SDI |
| **GRU** | Gated Recurrent Unit, better gradient flow than RNN |
| **LSTM** | Long Short-Term Memory, standard sequence baseline |
| **LSTM → (a,b,c) → formula** | LSTM hidden state predicts Weibull parameters; SDI = `a·exp(-(t/b)^c)` |
| **LSTM + Weibull dual loss** | LSTM predicts SDI directly; loss = `0.5×MSE(pred, true) + 0.5×MSE(pred, weibull(t,a,b,c))` where `a,b,c` are globally fitted on training segments each fold |
| **LSTM + sdi_last_known + delta_age** | LSTM hidden state is concatenated with `sdi_last_known` and `delta_age` before the output head, explicitly providing the most informative signal instead of requiring the model to discover it from the sequence |

### Results (Approach 2)

Segment-level GroupKFold, 5 folds:

| Model | CV R² | ± std | RMSE |
|---|---|---|---|
| Random Forest (stat. features) | 0.7750 | 0.0506 | 0.4939 |
| Linear Regression (stat. features) | 0.7669 | 0.0470 | 0.5037 |
| **LSTM + sdi_last_known + delta_age** | **0.6682** | 0.0473 | 0.6017 |
| LSTM → (a,b,c) → formula | 0.6051 | 0.0631 | 0.6586 |
| RNN | 0.4160 | 0.1430 | 0.7958 |
| GRU | 0.4078 | 0.1022 | 0.8028 |
| LSTM | 0.3850 | 0.1518 | 0.8151 |
| LSTM + Weibull dual loss | 0.0911 | 0.0419 | 1.0003 |

Statistical-feature baselines are included for direct comparison. Plain sequence models underperform with this dataset size (306 segments); the augmented LSTM that explicitly receives `sdi_last_known` and `delta_age` at the output layer closes much of the gap.

---

## Approach 3: Random Split — Full Model Comparison

Standard 5-fold random split across all 1,485 rows. Models are compared on a common feature set without using historical SDI values, making this applicable to road segments with no prior observations.

**Script:** `model_comparison.py`

### Features (Approach 3)

| Feature | Description |
|---|---|
| `Family_encoded` | Pavement structure type (label-encoded) |
| `attr_2way_AADTT` | Annual Average Daily Truck Traffic |
| `attr_TotalOverlay_in` | Total overlay thickness (in) |
| `Existing AC after milling (in)` | Remaining AC thickness |
| `Age` | Pavement age (years) |

### Models (Approach 3)

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

### Results (Approach 3)

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

- **Statistical features + segment split (R² = 0.78)** is the best overall approach. The dominant predictor is `sdi_last_known` (76.7% feature importance), meaning the most recent known condition is the strongest signal.
- **Gradient Boosting (R² = 0.645)** is the best model when no historical SDI is available (random split, static features + age only).
- **Plain sequence models (RNN/GRU/LSTM) underperform** with only 306 segments — too few samples for the models to learn temporal patterns. High variance across folds (std up to 0.15) confirms instability.
- **LSTM + sdi_last_known + delta_age (R² = 0.668)** is the best sequence model. Explicitly concatenating the most recent known SDI and forecast horizon to the output head closes most of the gap to the statistical-feature baseline. This confirms the bottleneck was not the LSTM's capacity but its inability to reliably discover which time step matters most with limited data.
- **LSTM → (a,b,c) → formula (R² = 0.61)** is the second-best sequence model. The Weibull structure provides a useful inductive bias even with limited data.
- **LSTM + Weibull dual loss (R² = 0.09)** fails — the globally-fitted formula is not a good anchor for individual segment predictions, and the conflicting loss signals hurt training.
- **Linear models (R² ≈ 0.33, random split)** confirm a strong nonlinear relationship between static features and SDI when no history is available.
- **The R² ceiling for static-feature models (~0.65)** reflects missing information in the dataset: construction quality, subgrade condition, local climate, and maintenance history are not captured.

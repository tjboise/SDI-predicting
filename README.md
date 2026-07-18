# SDI Prediction — Methodology Notes

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

Each segment has 4–9 observations recorded at different ages (non-consecutive years).

---

## Prediction Task Framing

The goal is to predict a segment's **future SDI** given its known history — not to randomly split rows across train/test, which would leak information from the same segment into both sets.

### Sample construction

For each segment, observations are sorted by `Age`. The **last observation** is held out as the prediction target. All earlier observations are summarized into a single feature row:

| Feature | Construction |
|---|---|
| `sdi_first` | SDI at the earliest observed age |
| `age_first` | Earliest observed age |
| `sdi_last_known` | SDI at the most recent observation before the target |
| `age_last_known` | Age at the most recent known observation |
| `target_age` | Age of the held-out observation (what we predict) |
| `delta_age` | `target_age − age_last_known` (forecast horizon) |
| `total_delta` | `target_age − age_first` |
| `sdi_slope` | `(sdi_last_known − sdi_first) / (age_last_known − age_first)` |
| `n_obs_known` | Number of historical observations available |
| Static features | `Family`, `AADTT`, `is_PG76`, `Overlay`, `Milling`, `AC material` |

This yields **306 prediction samples** (one per segment).

---

## Train / Test Split

Rows are split **by segment** using `GroupKFold(n_splits=5)`. A segment's data never appears in both training and test folds. This is critical: random row-level splitting would allow the model to see earlier observations of a test segment during training, artificially inflating R².

---

## Results

### Naive baseline
Predict `sdi_last_known` unchanged — R² = **0.41**, RMSE = **0.81**

### Model comparison (segment-level 5-fold CV)

| Model | CV R² | RMSE |
|---|---|---|
| Linear Regression | 0.777 | 0.494 |
| Ridge Regression | 0.778 | 0.493 |
| Random Forest | 0.775 | 0.494 |
| Gradient Boosting | 0.753 | 0.518 |

Linear and tree-based models perform similarly, suggesting the relationship between features and target is largely linear given the constructed features.

### Feature importance (Random Forest, trained on all data)

| Feature | Importance |
|---|---|
| `sdi_last_known` | 76.7% |
| `sdi_first` | 9.1% |
| `sdi_slope` | 2.9% |
| `attr_2way_AADTT` | 2.4% |
| `attr_TotalOverlay_in` | 1.2% |
| Other engineering features | < 3% total |

The dominant predictor is the most recent known SDI. Engineering features (pavement family, material, truck traffic) contribute relatively little once recent condition is known.

---

## Limitations and Next Steps

**Current approach only predicts one step ahead** — the average forecast horizon (`delta_age`) is 1.7 years. This may not reflect real operational needs.

Potential directions:

1. **Multi-horizon prediction** — generate a sample for every future observation within a segment (not just the last), to test longer-range forecasting.
2. **Decay curve modeling** — fit a parametric degradation function (e.g. linear or exponential decay) per segment, then use engineering features to predict the decay rate coefficients.
3. **Mixed effects model** — model each segment as having its own intercept (initial condition) with a shared decay slope driven by engineering features.


"""
SDI Prediction - Full Model Comparison
Random split (5-fold CV), dropping AC material and is_PG76.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ── 1. Load & prepare data ────────────────────────────────────────────────────
df = pd.read_excel('modeling_dataset.xlsx')
df['Family_encoded'] = LabelEncoder().fit_transform(df['Family'])
df = df.sort_values(['SubsectionKey', 'Age']).reset_index(drop=True)

FEATURES = ['Family_encoded', 'attr_2way_AADTT', 'attr_TotalOverlay_in',
            'Existing AC after milling (in)', 'Age']
TARGET = 'SDI'

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Features: {FEATURES}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)


# ── Weibull formula ───────────────────────────────────────────────────────────
def weibull_np(t, a, b, c):
    return a * np.exp(-(t / (b + 1e-6)) ** c)

def weibull_torch(t, a, b, c):
    return a * torch.exp(-(t / (b + 1e-6)) ** c)


# ── Helper: evaluate across folds ────────────────────────────────────────────
def cv_sklearn(model, X, y, kf):
    r2s, rmses = [], []
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        model.fit(Xtr, y[tr])
        p = model.predict(Xte)
        r2s.append(r2_score(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
    return np.mean(r2s), np.std(r2s), np.mean(rmses)


# ── Models 1-5: Classical ML ──────────────────────────────────────────────────
results = {}

print("Running classical ML models...")
for name, model in [
    ('Linear Regression', LinearRegression()),
    ('Ridge',             Ridge(alpha=1.0)),
    ('Random Forest',     RandomForestRegressor(n_estimators=200, random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ('SVR',               SVR(kernel='rbf', C=10, epsilon=0.1)),
]:
    r2, r2_std, rmse = cv_sklearn(model, X, y, kf)
    results[name] = (r2, r2_std, rmse)
    print(f"  {name:<25} R2={r2:.4f} ±{r2_std:.4f}  RMSE={rmse:.4f}")


# ── Model 6: Pure ANN ─────────────────────────────────────────────────────────
class PlainANN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 16),     nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_plain_ann(Xtr, ytr, epochs=400, lr=1e-3):
    model = PlainANN(Xtr.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = nn.MSELoss()(model(Xt), yt)
        loss.backward(); opt.step()
    return model

print("\nRunning ANN models...")
r2s, rmses = [], []
for tr, te in kf.split(X):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X[tr]).astype(np.float32)
    Xte_s = sc.transform(X[te]).astype(np.float32)
    model = train_plain_ann(Xtr_s, y[tr])
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(Xte_s)).numpy()
    r2s.append(r2_score(y[te], p))
    rmses.append(np.sqrt(mean_squared_error(y[te], p)))
r2, r2_std, rmse = np.mean(r2s), np.std(r2s), np.mean(rmses)
results['ANN (pure data)'] = (r2, r2_std, rmse)
print(f"  {'ANN (pure data)':<25} R2={r2:.4f} ±{r2_std:.4f}  RMSE={rmse:.4f}")


# ── Model 7: ANN → (a, b, c) → formula ───────────────────────────────────────
# Features WITHOUT Age (Age is t in the formula)
FEAT_STATIC = ['Family_encoded', 'attr_2way_AADTT', 'attr_TotalOverlay_in',
               'Existing AC after milling (in)']
X_static = df[FEAT_STATIC].values.astype(np.float32)
t_all    = df['Age'].values.astype(np.float32)

class ABCNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
        )
        self.ha = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.hb = nn.Sequential(nn.Linear(32, 1), nn.Softplus())
        self.hc = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x):
        h = self.shared(x)
        a = self.ha(h).squeeze(-1) * 5.0
        b = self.hb(h).squeeze(-1) * 15.0
        c = self.hc(h).squeeze(-1) * 2.0
        return a, b, c

def train_abc_net(Xtr, ytr, ttr, epochs=400, lr=1e-3):
    model = ABCNet(Xtr.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    tt = torch.tensor(ttr, dtype=torch.float32)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        a, b, c = model(Xt)
        pred = weibull_torch(tt, a, b, c)
        loss = nn.MSELoss()(pred, yt)
        loss.backward(); opt.step()
    return model

r2s, rmses = [], []
for tr, te in kf.split(X_static):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_static[tr]).astype(np.float32)
    Xte_s = sc.transform(X_static[te]).astype(np.float32)
    model = train_abc_net(Xtr_s, y[tr], t_all[tr])
    model.eval()
    with torch.no_grad():
        a, b, c = model(torch.tensor(Xte_s))
        t_te = torch.tensor(t_all[te])
        p = weibull_torch(t_te, a, b, c).numpy()
    r2s.append(r2_score(y[te], p))
    rmses.append(np.sqrt(mean_squared_error(y[te], p)))
r2, r2_std, rmse = np.mean(r2s), np.std(r2s), np.mean(rmses)
results['ANN → (a,b,c) → formula'] = (r2, r2_std, rmse)
print(f"  {'ANN → (a,b,c) → formula':<25} R2={r2:.4f} ±{r2_std:.4f}  RMSE={rmse:.4f}")


# ── Model 8: ANN₁→(a,b,c) + ANN₂(a,b,c,t)→SDI with dual loss ───────────────
class ANN2(nn.Module):
    """Takes (a, b, c, t) and predicts SDI."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_dual(Xtr, ytr, ttr, lam=0.5, epochs=400, lr=1e-3):
    ann1 = ABCNet(Xtr.shape[1])
    ann2 = ANN2()
    opt  = torch.optim.Adam(
        list(ann1.parameters()) + list(ann2.parameters()), lr=lr, weight_decay=1e-4
    )
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    tt = torch.tensor(ttr, dtype=torch.float32)

    for _ in range(epochs):
        ann1.train(); ann2.train(); opt.zero_grad()
        a, b, c = ann1(Xt)
        inp2     = torch.stack([a, b, c, tt], dim=1)
        sdi_pred = ann2(inp2)
        formula  = weibull_torch(tt, a, b, c)
        loss = (1 - lam) * nn.MSELoss()(sdi_pred, yt) \
             +      lam  * nn.MSELoss()(sdi_pred, formula)
        loss.backward(); opt.step()
    return ann1, ann2

r2s, rmses = [], []
for tr, te in kf.split(X_static):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_static[tr]).astype(np.float32)
    Xte_s = sc.transform(X_static[te]).astype(np.float32)
    ann1, ann2 = train_dual(Xtr_s, y[tr], t_all[tr])
    ann1.eval(); ann2.eval()
    with torch.no_grad():
        a, b, c = ann1(torch.tensor(Xte_s))
        t_te    = torch.tensor(t_all[te])
        inp2    = torch.stack([a, b, c, t_te], dim=1)
        p       = ann2(inp2).numpy()
    r2s.append(r2_score(y[te], p))
    rmses.append(np.sqrt(mean_squared_error(y[te], p)))
r2, r2_std, rmse = np.mean(r2s), np.std(r2s), np.mean(rmses)
results['ANN dual-loss (new)'] = (r2, r2_std, rmse)
print(f"  {'ANN dual-loss (new)':<25} R2={r2:.4f} ±{r2_std:.4f}  RMSE={rmse:.4f}")


# ── Final table ───────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"{'Model':<30} {'R2':>7} {'±':>7} {'RMSE':>7}")
print("-" * 60)
for name, (r2, r2_std, rmse) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"{name:<30} {r2:>7.4f} {r2_std:>7.4f} {rmse:>7.4f}")
print("=" * 60)

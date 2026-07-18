
"""
SDI Prediction - Approach 2: Segment-Aware Sequence Models
Split by segment (GroupKFold), predict last observation from earlier history.
Models: Statistical features baseline, RNN, GRU, LSTM,
        LSTM->(a,b,c)->Weibull, LSTM + Weibull dual loss.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_excel('modeling_dataset.xlsx')
df['Family_encoded'] = LabelEncoder().fit_transform(df['Family'])
df = df.sort_values(['SubsectionKey', 'Age']).reset_index(drop=True)

STATIC = ['Family_encoded', 'attr_2way_AADTT', 'attr_TotalOverlay_in',
          'Existing AC after milling (in)']

# ── 2. Build segment records ───────────────────────────────────────────────────
segments = []
for key, grp in df.groupby('SubsectionKey'):
    grp = grp.sort_values('Age')
    if len(grp) < 2:
        continue
    static_feats = [grp[f].iloc[0] for f in STATIC]
    ages  = grp['Age'].values.astype(np.float32)
    sdis  = grp['SDI'].values.astype(np.float32)
    segments.append({
        'key':          key,
        'static':       np.array(static_feats, dtype=np.float32),
        'ages':         ages,
        'sdis':         sdis,
        # training history (all but last)
        'train_ages':   ages[:-1],
        'train_sdis':   sdis[:-1],
        # prediction target
        'target_age':   float(ages[-1]),
        'target_sdi':   float(sdis[-1]),
    })

print(f"Segments: {len(segments)}")
seq_lens = [len(s['train_ages']) for s in segments]
print(f"Training sequence length: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.1f}")

# ── 3. Statistical feature baseline (Approach 1 re-run) ───────────────────────
def build_stat_features(segs):
    rows = []
    for s in segs:
        ta, ts = s['train_ages'], s['train_sdis']
        slope = (ts[-1] - ts[0]) / max(ta[-1] - ta[0], 1)
        rows.append([
            *s['static'],
            ts[0], ta[0],
            ts[-1], ta[-1],
            s['target_age'],
            s['target_age'] - ta[-1],
            s['target_age'] - ta[0],
            len(ta),
            slope,
        ])
    return np.array(rows, dtype=np.float32)

# ── 4. Sequence tensor builder ─────────────────────────────────────────────────
def build_sequences(segs, static_scaler=None, fit_scaler=False):
    """
    Each time step: [age_t, SDI_t, Family, AADTT, Overlay, Milling]
    Returns padded tensor (N, max_len, features), lengths, targets, target_ages.
    """
    max_len = max(len(s['train_ages']) for s in segs)
    step_dim = 2 + len(STATIC)  # age + SDI + static features

    static_arr = np.array([s['static'] for s in segs], dtype=np.float32)
    if fit_scaler:
        static_scaler = StandardScaler()
        static_scaler.fit(static_arr)
    static_norm = static_scaler.transform(static_arr)

    # Normalise age and SDI globally (0-1 roughly)
    padded   = np.zeros((len(segs), max_len, step_dim), dtype=np.float32)
    lengths  = []
    targets  = []
    t_ages   = []

    for i, s in enumerate(segs):
        ta, ts = s['train_ages'], s['train_sdis']
        L = len(ta)
        lengths.append(L)
        targets.append(s['target_sdi'])
        t_ages.append(s['target_age'])
        for j in range(L):
            padded[i, j, 0] = ta[j] / 15.0          # age normalised
            padded[i, j, 1] = ts[j] / 5.0            # SDI normalised
            padded[i, j, 2:] = static_norm[i]

    return (torch.tensor(padded),
            lengths,
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(t_ages,  dtype=torch.float32),
            static_scaler)

# ── 5. Weibull helpers ─────────────────────────────────────────────────────────
def weibull_np(t, a, b, c):
    return a * np.exp(-(t / (b + 1e-8)) ** c)

def weibull_torch(t, a, b, c):
    return a * torch.exp(-(t / (b + 1e-8)) ** c)

def fit_global_weibull(segs):
    """Fit one global (a,b,c) on all training observations."""
    all_t, all_s = [], []
    for s in segs:
        all_t.extend(s['train_ages'].tolist())
        all_s.extend(s['train_sdis'].tolist())
    try:
        popt, _ = curve_fit(weibull_np,
                            np.array(all_t), np.array(all_s),
                            p0=[5.0, 15.0, 1.0],
                            bounds=([0, 0.1, 0.1], [5.5, 50, 10]),
                            maxfev=10000)
        return popt
    except Exception:
        return np.array([4.5, 15.0, 1.0])

# ── 6. Sequence model definitions ─────────────────────────────────────────────
class SeqModel(nn.Module):
    """Vanilla RNN / GRU / LSTM that predicts scalar SDI from a sequence."""
    def __init__(self, input_dim, hidden=32, num_layers=1, cell='LSTM'):
        super().__init__()
        rnn_cls = {'RNN': nn.RNN, 'GRU': nn.GRU, 'LSTM': nn.LSTM}[cell]
        self.rnn = rnn_cls(input_dim, hidden, num_layers=num_layers,
                           batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))
        self.cell = cell

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                      enforce_sorted=False)
        out, hidden = self.rnn(packed)
        if self.cell == 'LSTM':
            h = hidden[0][-1]   # last layer hidden state
        else:
            h = hidden[-1]
        return self.head(h).squeeze(-1)


class LSTMtoABC(nn.Module):
    """LSTM encodes sequence → predicts (a,b,c) → SDI via Weibull formula."""
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.ha = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.hb = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.hc = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x, lengths, t_target):
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = h[-1]
        a = self.ha(h).squeeze(-1) * 5.0
        b = self.hb(h).squeeze(-1) * 15.0
        c = self.hc(h).squeeze(-1) * 2.0
        sdi = weibull_torch(t_target, a, b, c)
        return sdi, a, b, c


# ── 7. Training helpers ────────────────────────────────────────────────────────
def train_seq(model, X, lengths, y, epochs=300, lr=1e-3, loss_fn=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(X, lengths)
        loss = loss_fn(pred, y)
        loss.backward(); opt.step(); sch.step()
    return model


def train_lstm_abc(model, X, lengths, y, t_ages, epochs=300, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pred, _, _, _ = model(X, lengths, t_ages)
        loss = nn.MSELoss()(pred, y)
        loss.backward(); opt.step(); sch.step()
    return model


def train_lstm_dual(model, X, lengths, y, t_ages, abc_global, lam=0.5,
                    epochs=300, lr=1e-3):
    """LSTM SDI prediction with dual loss: data + Weibull formula."""
    a0, b0, c0 = abc_global
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(X, lengths)
        formula = weibull_torch(t_ages,
                                torch.tensor(a0, dtype=torch.float32),
                                torch.tensor(b0, dtype=torch.float32),
                                torch.tensor(c0, dtype=torch.float32))
        loss = (1 - lam) * nn.MSELoss()(pred, y) \
             +      lam  * nn.MSELoss()(pred, formula)
        loss.backward(); opt.step(); sch.step()
    return model


# ── 8. Cross-validation ────────────────────────────────────────────────────────
seg_keys = np.array([s['key'] for s in segments])
gkf = GroupKFold(n_splits=5)

results = {}

# ── 8a. Statistical feature baseline ──────────────────────────────────────────
print("=" * 55)
print("Running models (segment GroupKFold, 5 folds)...")
print("=" * 55)

X_stat = build_stat_features(segments)
y_stat = np.array([s['target_sdi'] for s in segments], dtype=np.float32)

for name, model in [('Linear Regression', LinearRegression()),
                    ('Random Forest',     RandomForestRegressor(200, random_state=42))]:
    r2s, rmses = [], []
    for tr, te in gkf.split(X_stat, y_stat, seg_keys):
        sc = StandardScaler()
        model.fit(sc.fit_transform(X_stat[tr]), y_stat[tr])
        p = model.predict(sc.transform(X_stat[te]))
        r2s.append(r2_score(y_stat[te], p))
        rmses.append(np.sqrt(mean_squared_error(y_stat[te], p)))
    results[name] = (np.mean(r2s), np.std(r2s), np.mean(rmses))
    print(f"  {name:<30} R2={np.mean(r2s):.4f} ±{np.std(r2s):.4f}  RMSE={np.mean(rmses):.4f}")

# ── 8b. RNN / GRU / LSTM ──────────────────────────────────────────────────────
segs_arr = np.array(segments, dtype=object)
INPUT_DIM = 2 + len(STATIC)

for cell in ['RNN', 'GRU', 'LSTM']:
    r2s, rmses = [], []
    for tr_idx, te_idx in gkf.split(np.zeros(len(segments)), y_stat, seg_keys):
        segs_tr = segs_arr[tr_idx].tolist()
        segs_te = segs_arr[te_idx].tolist()

        X_tr, L_tr, y_tr, t_tr, sc = build_sequences(segs_tr, fit_scaler=True)
        X_te, L_te, y_te, t_te, _  = build_sequences(segs_te, static_scaler=sc)

        model = SeqModel(INPUT_DIM, hidden=32, cell=cell)
        train_seq(model, X_tr, L_tr, y_tr)

        model.eval()
        with torch.no_grad():
            p = model(X_te, L_te).numpy()
        r2s.append(r2_score(y_te.numpy(), p))
        rmses.append(np.sqrt(mean_squared_error(y_te.numpy(), p)))

    results[cell] = (np.mean(r2s), np.std(r2s), np.mean(rmses))
    print(f"  {cell:<30} R2={np.mean(r2s):.4f} ±{np.std(r2s):.4f}  RMSE={np.mean(rmses):.4f}")

# ── 8c. LSTM → (a,b,c) → Weibull ─────────────────────────────────────────────
INPUT_DIM_STATIC = 2 + len(STATIC)   # same; age/SDI in sequence, static appended
r2s, rmses = [], []
for tr_idx, te_idx in gkf.split(np.zeros(len(segments)), y_stat, seg_keys):
    segs_tr = segs_arr[tr_idx].tolist()
    segs_te = segs_arr[te_idx].tolist()

    X_tr, L_tr, y_tr, t_tr, sc = build_sequences(segs_tr, fit_scaler=True)
    X_te, L_te, y_te, t_te, _  = build_sequences(segs_te, static_scaler=sc)

    model = LSTMtoABC(INPUT_DIM, hidden=32)
    train_lstm_abc(model, X_tr, L_tr, y_tr, t_tr)

    model.eval()
    with torch.no_grad():
        p, _, _, _ = model(X_te, L_te, t_te)
        p = p.numpy()
    r2s.append(r2_score(y_te.numpy(), p))
    rmses.append(np.sqrt(mean_squared_error(y_te.numpy(), p)))

results['LSTM → (a,b,c) → formula'] = (np.mean(r2s), np.std(r2s), np.mean(rmses))
print(f"  {'LSTM → (a,b,c) → formula':<30} R2={np.mean(r2s):.4f} ±{np.std(r2s):.4f}  RMSE={np.mean(rmses):.4f}")

# ── 8d. LSTM + Weibull dual loss ──────────────────────────────────────────────
r2s, rmses = [], []
for tr_idx, te_idx in gkf.split(np.zeros(len(segments)), y_stat, seg_keys):
    segs_tr = segs_arr[tr_idx].tolist()
    segs_te = segs_arr[te_idx].tolist()

    X_tr, L_tr, y_tr, t_tr, sc = build_sequences(segs_tr, fit_scaler=True)
    X_te, L_te, y_te, t_te, _  = build_sequences(segs_te, static_scaler=sc)

    # Fit global Weibull on training segments only
    abc = fit_global_weibull(segs_tr)

    model = SeqModel(INPUT_DIM, hidden=32, cell='LSTM')
    train_lstm_dual(model, X_tr, L_tr, y_tr, t_tr, abc)

    model.eval()
    with torch.no_grad():
        p = model(X_te, L_te).numpy()
    r2s.append(r2_score(y_te.numpy(), p))
    rmses.append(np.sqrt(mean_squared_error(y_te.numpy(), p)))

results['LSTM + Weibull dual loss'] = (np.mean(r2s), np.std(r2s), np.mean(rmses))
print(f"  {'LSTM + Weibull dual loss':<30} R2={np.mean(r2s):.4f} ±{np.std(r2s):.4f}  RMSE={np.mean(rmses):.4f}")

# ── 9. Final table ─────────────────────────────────────────────────────────────
print()
print("=" * 58)
print(f"{'Model':<32} {'R2':>7} {'± std':>7} {'RMSE':>7}")
print("-" * 58)
for name, (r2, r2s, rmse) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"{name:<32} {r2:>7.4f} {r2s:>7.4f} {rmse:>7.4f}")
print("=" * 58)

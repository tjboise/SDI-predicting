
"""
Generate prediction visualization plots for Approach 1.
For 4 selected test segments, plots the actual SDI history (line)
and model predictions at the held-out last age (colored dots).
Models: Random Forest, LSTM + sdi_last_known + delta_age, plain LSTM.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn
import warnings, os
warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42)

os.makedirs('figures', exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_excel('modeling_dataset.xlsx')
df['Family_encoded'] = LabelEncoder().fit_transform(df['Family'])
df = df.sort_values(['SubsectionKey', 'Age']).reset_index(drop=True)
STATIC = ['Family_encoded', 'attr_2way_AADTT', 'attr_TotalOverlay_in',
          'Existing AC after milling (in)']

segments = []
for key, grp in df.groupby('SubsectionKey'):
    grp = grp.sort_values('Age')
    if len(grp) < 2: continue
    ta = grp['Age'].values[:-1].astype(float)
    ts = grp['SDI'].values[:-1].astype(float)
    slope = (ts[-1] - ts[0]) / max(ta[-1] - ta[0], 1)
    segments.append({
        'key': key,
        'static': np.array([grp[f].iloc[0] for f in STATIC], dtype=np.float32),
        'all_ages': grp['Age'].values.astype(float),
        'all_sdis': grp['SDI'].values.astype(float),
        'train_ages': ta, 'train_sdis': ts,
        'target_age': float(grp['Age'].iloc[-1]),
        'target_sdi': float(grp['SDI'].iloc[-1]),
        'stat_feats': np.array([
            grp['Family_encoded'].iloc[0],
            grp['attr_2way_AADTT'].iloc[0],
            grp['attr_TotalOverlay_in'].iloc[0],
            grp['Existing AC after milling (in)'].iloc[0],
            ts[0], ta[0], ts[-1], ta[-1],
            float(grp['Age'].iloc[-1]),
            float(grp['Age'].iloc[-1]) - ta[-1],
            float(grp['Age'].iloc[-1]) - ta[0],
            len(ta), slope,
        ], dtype=np.float32),
        'family': grp['Family'].iloc[0],
    })

seg_keys = np.array([s['key'] for s in segments])
segs_arr = np.array(segments, dtype=object)
y_all    = np.array([s['target_sdi'] for s in segments])

# ── LSTM models ───────────────────────────────────────────────────────────────
INPUT_DIM = 2 + len(STATIC)

def build_sequences(segs, static_scaler=None, fit_scaler=False):
    max_len = max(len(s['train_ages']) for s in segs)
    static_arr = np.array([s['static'] for s in segs], dtype=np.float32)
    if fit_scaler:
        static_scaler = StandardScaler(); static_scaler.fit(static_arr)
    static_norm = static_scaler.transform(static_arr)
    padded = np.zeros((len(segs), max_len, INPUT_DIM), dtype=np.float32)
    lengths, targets, sdi_lasts, delta_ages = [], [], [], []
    for i, s in enumerate(segs):
        ta, ts = s['train_ages'], s['train_sdis']
        L = len(ta); lengths.append(L)
        targets.append(s['target_sdi'])
        sdi_lasts.append(ts[-1])
        delta_ages.append(s['target_age'] - ta[-1])
        for j in range(L):
            padded[i,j,0] = ta[j]/15.0; padded[i,j,1] = ts[j]/5.0
            padded[i,j,2:] = static_norm[i]
    return (torch.tensor(padded), lengths,
            torch.tensor(targets,    dtype=torch.float32),
            torch.tensor(sdi_lasts,  dtype=torch.float32),
            torch.tensor(delta_ages, dtype=torch.float32),
            static_scaler)

class PlainLSTM(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h,_) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)

class AugLSTM(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h+2,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x, lengths, sdi_last, delta_age):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h,_) = self.lstm(packed)
        aug = torch.stack([sdi_last, delta_age], dim=1)
        return self.head(torch.cat([h[-1], aug], dim=1)).squeeze(-1)

def train_lstm(model, X, L, y, sl=None, da=None, epochs=300, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=150, gamma=0.5)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(X, L, sl, da) if sl is not None else model(X, L)
        nn.MSELoss()(pred, y).backward(); opt.step(); sch.step()

# ── Train on fold 0 train, collect test predictions ───────────────────────────
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(np.zeros(len(segments)), y_all, seg_keys))
tr_idx, te_idx = splits[0]

segs_tr = segs_arr[tr_idx].tolist()
segs_te = segs_arr[te_idx].tolist()

# Random Forest
STAT_FEATS = ['Family_encoded','attr_2way_AADTT','attr_TotalOverlay_in',
              'Existing AC after milling (in)','sdi_first','age_first',
              'sdi_last_known','age_last_known','target_age','delta_age',
              'total_delta','n_obs_known','sdi_slope']
X_tr_stat = np.array([s['stat_feats'] for s in segs_tr])
X_te_stat = np.array([s['stat_feats'] for s in segs_te])
y_tr = np.array([s['target_sdi'] for s in segs_tr])
y_te = np.array([s['target_sdi'] for s in segs_te])
sc_stat = StandardScaler()
rf = RandomForestRegressor(200, random_state=42)
rf.fit(sc_stat.fit_transform(X_tr_stat), y_tr)
rf_preds = rf.predict(sc_stat.transform(X_te_stat))

# Plain LSTM
X_tr, L_tr, y_tr_t, sl_tr, da_tr, sc_seq = build_sequences(segs_tr, fit_scaler=True)
X_te, L_te, y_te_t, sl_te, da_te, _      = build_sequences(segs_te, static_scaler=sc_seq)
plain_lstm = PlainLSTM(INPUT_DIM)
train_lstm(plain_lstm, X_tr, L_tr, y_tr_t)
plain_lstm.eval()
with torch.no_grad():
    lstm_preds = plain_lstm(X_te, L_te).numpy()

# Augmented LSTM
aug_lstm = AugLSTM(INPUT_DIM)
train_lstm(aug_lstm, X_tr, L_tr, y_tr_t, sl_tr, da_tr)
aug_lstm.eval()
with torch.no_grad():
    aug_preds = aug_lstm(X_te, L_te, sl_te, da_te).numpy()

print(f"Test fold size: {len(segs_te)} segments")
print(f"RF   R2: {r2_score(y_te, rf_preds):.4f}")
print(f"LSTM R2: {r2_score(y_te, lstm_preds):.4f}")
print(f"AugLSTM R2: {r2_score(y_te, aug_preds):.4f}")

# ── Pick 4 representative segments ────────────────────────────────────────────
# Select segments that span a range of true SDI values and error patterns
errors_rf = np.abs(rf_preds - y_te)
# Pick: 2 with small error (good predictions), 2 with moderate error, varied true SDI
quartiles = np.percentile(y_te, [20, 40, 60, 80])
chosen = []
for q in quartiles:
    idx = np.argmin(np.abs(y_te - q))
    if idx not in chosen:
        chosen.append(idx)
chosen = chosen[:1]

# ── Plot ───────────────────────────────────────────────────────────────────────
COLORS = {
    'RF':   ('#2196F3', 'Random Forest'),
    'AUG':  ('#4CAF50', 'LSTM + last SDI + Δage'),
    'LSTM': ('#F44336', 'Plain LSTM'),
}

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#0f1923')
ax.set_facecolor('#162032')

seg_i = chosen[0]
s = segs_te[seg_i]
ages = s['all_ages']
sdis = s['all_sdis']
target_age = s['target_age']
target_sdi = s['target_sdi']

ax.plot(ages[:-1], sdis[:-1], color='#a8d4f5', linewidth=2,
        marker='o', markersize=5, zorder=3, label='Actual (known)')
ax.plot([ages[-2], ages[-1]], [sdis[-2], target_sdi],
        color='#a8d4f5', linewidth=1.5, linestyle='--', zorder=2)
ax.scatter([target_age], [target_sdi], color='white', s=120, zorder=5,
           marker='*', label='True (held-out)', edgecolors='#a8d4f5', linewidth=0.8)
ax.scatter([target_age], [rf_preds[seg_i]],
           color=COLORS['RF'][0], s=120, zorder=6, marker='D',
           label=COLORS['RF'][1], edgecolors='white', linewidth=0.6)
ax.scatter([target_age], [aug_preds[seg_i]],
           color=COLORS['AUG'][0], s=120, zorder=6, marker='^',
           label=COLORS['AUG'][1], edgecolors='white', linewidth=0.6)
ax.scatter([target_age], [lstm_preds[seg_i]],
           color=COLORS['LSTM'][0], s=120, zorder=6, marker='s',
           label=COLORS['LSTM'][1], edgecolors='white', linewidth=0.6)

ax.set_xlim(max(0, ages[0]-1), ages[-1]+1.5)
ax.set_ylim(0, 5.4)
ax.set_xlabel('Age (years)', color='#6b88a4', fontsize=11)
ax.set_ylabel('SDI', color='#6b88a4', fontsize=11)
ax.tick_params(colors='#6b88a4')
for spine in ax.spines.values():
    spine.set_edgecolor('#1e3048')
ax.grid(True, color='#1e3048', linewidth=0.6, linestyle='--')

family_short = s['family'].replace('AC over ', '').replace(' | ', '/')
ax.set_title(f"{s['key']}  [{family_short}]", color='#d8e4f0', fontsize=11, pad=8)

err_rf  = rf_preds[seg_i]  - target_sdi
err_aug = aug_preds[seg_i] - target_sdi
err_lst = lstm_preds[seg_i] - target_sdi
ax.text(0.02, 0.04,
        f"RF err={err_rf:+.2f}   AugLSTM err={err_aug:+.2f}   LSTM err={err_lst:+.2f}",
        transform=ax.transAxes, color='#6b88a4', fontsize=9,
        verticalalignment='bottom')

ax.legend(frameon=True, facecolor='#162032', edgecolor='#1e3048',
          labelcolor='#d8e4f0', fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('figures/approach1_predictions.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1923')
print("Saved: figures/approach1_predictions.png")

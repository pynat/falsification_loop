# held_out.py, true out-of-sample validation on unseen held-out data
# requires: results/eth_dollar_bars_held_out.csv  (written by fetch_data.py)
#           results/eth_dollar_bars.csv            (train bars, for vol warm-up only)
#           results/final_model.pkl               (written by tuning.py)
# outputs:  results/held_out_report.txt

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

os.makedirs("results", exist_ok=True)

from config import (
    FINAL_FEATURES, EDGE_THRESHOLD, MIN_PROB, B,
    REGIME_GATE_ENABLED, PT_SL, CAPITAL, FEE_BPS, SLIPPAGE_BPS,
    BARS_PER_YEAR, KELLY_FRACS, MAX_HOLD
)
from cpcv import build_features, make_labels, fit_hmm_regimes

ROUND_TRIP = 2 * (FEE_BPS + SLIPPAGE_BPS) / 10_000

def kelly_f(p, b=B):
    return max((p * b - (1 - p)) / b, 0.0)

# ── load ──────────────────────────────────────────────────────────────────────
print("loading held-out bars...")
df_raw  = pd.read_csv("results/eth_dollar_bars_held_out.csv", parse_dates=True, index_col=0)
print(f"  bars: {len(df_raw)}  ({df_raw.index[0].date()} to {df_raw.index[-1].date()})")

# prepend last 20 train bars so vol_20b window has no NaN at held-out start
train_raw  = pd.read_csv("results/eth_dollar_bars.csv", parse_dates=True, index_col=0)
warmup = train_raw.tail(max(20, MAX_HOLD) + 10)
df_raw_ext = pd.concat([warmup, df_raw])

# build features on extended df, slice back to held-out timestamps only
df_ext = build_features(df_raw_ext)
df     = df_ext.loc[df_raw.index].copy()
labels = make_labels(df["close"])
df     = df.join(labels)

feats = [f for f in FINAL_FEATURES if f in df.columns]
df    = df.dropna(subset=["label"] + feats)
df["label"]  = df["label"].astype(int)
df["regime"] = fit_hmm_regimes(df["volatility_7b"].values)

print(f"  usable bars: {len(df)}")
print(f"  regime gate: {'on' if REGIME_GATE_ENABLED else 'off'}")
print("─" * 70)

if len(df) < 20:
    raise RuntimeError(f"held-out too small ({len(df)} bars)")

# ── predict ───────────────────────────────────────────────────────────────────
clf  = joblib.load("results/final_model.pkl")
prob = clf.predict_proba(df[feats])[:, 1]
auc  = roc_auc_score(df["label"], prob)

verdict = "edge" if auc >= EDGE_THRESHOLD else ("marginal" if auc >= 0.52 else "no edge")
print(f"held-out auc:  {auc:.4f}  ({verdict})")
print(f"label balance: {df['label'].mean():.3f}")
print("─" * 70)

# ── auc by regime ─────────────────────────────────────────────────────────────
print("auc by regime:")
for r, name in [(0, "low"), (1, "mid"), (2, "high")]:
    mask = df["regime"].values == r
    if mask.sum() >= 10 and len(np.unique(df["label"].values[mask])) == 2:
        r_auc = roc_auc_score(df["label"].values[mask], prob[mask])
        print(f"  {name}_vol:  n={mask.sum():4d}  auc={r_auc:.4f}")
    else:
        print(f"  {name}_vol:  n={mask.sum():4d}  auc=n/a")
print("─" * 70)

# ── barrier widths: asymmetric PT_SL[0] for wins, PT_SL[1] for losses ─────────
df_raw_ext["log_ret"] = np.log(df_raw_ext["close"] / df_raw_ext["close"].shift(1))
df_raw_ext["vol_20b"] = df_raw_ext["log_ret"].rolling(20).std()
df_raw_ext["bw_pt"]   = PT_SL[0] * df_raw_ext["vol_20b"]
df_raw_ext["bw_sl"]   = PT_SL[1] * df_raw_ext["vol_20b"]

bw_pt = df_raw_ext.loc[df.index, "bw_pt"].values
bw_sl = df_raw_ext.loc[df.index, "bw_sl"].values

# effective barrier per trade: pt width for winners, sl width for losers
barrier_w = np.where(df["label"].values == 1, bw_pt, bw_sl)

if np.isnan(barrier_w).any():
    n_nan = np.isnan(barrier_w).sum()
    print(f"  warning: {n_nan} NaN barrier_w -> filling with mean")
    barrier_w = np.where(np.isnan(barrier_w), np.nanmean(barrier_w), barrier_w)

print(f"barrier_width: mean={barrier_w.mean()*100:.3f}%  median={np.median(barrier_w)*100:.3f}%")
print("─" * 70)

# ── bet mask ──────────────────────────────────────────────────────────────────
if REGIME_GATE_ENABLED:
    bet_mask = (prob > MIN_PROB) & (df["regime"].values != 2)
else:
    bet_mask = prob > MIN_PROB

# asymmetric payoff: +PT_SL[0] for wins, -PT_SL[1] for losses
payoff     = np.where(df["label"].values == 1, PT_SL[0], -PT_SL[1])
f_full_raw = np.vectorize(kelly_f)(prob).clip(0)

# ── pnl per kelly fraction ────────────────────────────────────────────────────
report_blocks = []
n_years = max((df.index[-1] - df.index[0]).days / 365.25, 0.25)

for frac in KELLY_FRACS:
    f_vals = np.where(bet_mask, f_full_raw * frac, 0.0)

    # asymmetric USD pnl: win trades use bw_pt, loss trades use bw_sl
    pnl_gross = CAPITAL * f_vals * np.where(
        df["label"].values == 1, bw_pt, -bw_sl
    )
    cost    = np.where(bet_mask, CAPITAL * f_vals * ROUND_TRIP, 0.0)
    pnl_net = pnl_gross - cost

    # f-unit pnl for sizing.py comparison
    pnl_fstar = f_vals * payoff

    cum_net    = pd.Series(pnl_net).cumsum()
    max_dd_usd = (cum_net - cum_net.cummax()).min()
    max_dd_pct = max_dd_usd / CAPITAL * 100

    n_bets     = bet_mask.sum()
    annual_net = pnl_net.sum() / max(n_years, 1e-6)
    annual_ret = annual_net / CAPITAL * 100
    calmar     = annual_ret / abs(max_dd_pct) if max_dd_pct != 0 else float("nan")

    # fix: sharpe on all bars with sqrt(BARS_PER_YEAR)
    pnl_net_s = pd.Series(pnl_net)
    sharpe    = (pnl_net_s.mean() / pnl_net_s.std() * np.sqrt(BARS_PER_YEAR)
                 if pnl_net_s.std() > 0 else float("nan"))

    avg_pos  = (CAPITAL * f_vals[bet_mask] * barrier_w[bet_mask]).mean() if n_bets > 0 else 0.0
    avg_pnl  = pnl_net[bet_mask].mean() if n_bets > 0 else 0.0

    cum_f    = pd.Series(pnl_fstar).cumsum()
    max_dd_f = (cum_f - cum_f.cummax()).min()

    # vol-correlation diagnostic
    bet_bw       = barrier_w[bet_mask]
    bet_fstar    = pnl_fstar[bet_mask]
    vol_pnl_corr = np.corrcoef(bet_fstar, bet_bw)[0, 1] if n_bets > 1 else float("nan")

    label = f"kelly {int(frac*100)}%"
    print(f"{label}:")
    print(f"  n_bets:        {n_bets} / {len(df)}")
    print(f"  total_pnl:     USD {pnl_net.sum():,.2f}")
    print(f"  annual_pnl:    USD {annual_net:,.0f}  ({annual_ret:.2f}%/yr)")
    print(f"  sharpe (net):  {sharpe:.3f}")
    print(f"  max_dd:        USD {max_dd_usd:,.2f}  ({max_dd_pct:.2f}% of capital)")
    print(f"  calmar:        {calmar:.3f}")
    print(f"  avg position:  USD {avg_pos:,.0f}")
    print(f"  avg pnl/bet:   USD {avg_pnl:,.2f}")
    print(f"  total cost:    USD {cost.sum():,.2f}")
    print(f"  vol-pnl corr:  {vol_pnl_corr:.3f}  (negative = losses in high-vol bars)")
    print(f"  f* total_pnl:  {pnl_fstar.sum():.4f}  (sizing.py comparison)")
    print(f"  f* max_dd:     {max_dd_f:.4f}")
    print("─" * 70)

    report_blocks.append([
        f"{label}:",
        f"  total_pnl:     USD {pnl_net.sum():,.2f}",
        f"  annual_pnl:    USD {annual_net:,.0f}  ({annual_ret:.2f}%/yr)",
        f"  sharpe (net):  {sharpe:.3f}",
        f"  max_dd:        USD {max_dd_usd:,.2f}  ({max_dd_pct:.2f}% of capital)",
        f"  calmar:        {calmar:.3f}",
        f"  avg position:  USD {avg_pos:,.0f}",
        f"  avg pnl/bet:   USD {avg_pnl:,.2f}",
        f"  total cost:    USD {cost.sum():,.2f}",
        f"  vol-pnl corr:  {vol_pnl_corr:.3f}",
        f"  f* total_pnl:  {pnl_fstar.sum():.4f}",
        f"  f* max_dd:     {max_dd_f:.4f}",
        "",
    ])

# ── plots ─────────────────────────────────────────────────────────────────────
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size']   = 11
rcParams['axes.titlesize'] = 11
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8

# recompute pnl for primary kelly fraction for plots
frac_plot  = KELLY_FRACS[0]
f_vals_plt = np.where(bet_mask, f_full_raw * frac_plot, 0.0)

pnl_gross_plt = CAPITAL * f_vals_plt * np.where(
    df["label"].values == 1, bw_pt, -bw_sl
)
cost_plt    = np.where(bet_mask, CAPITAL * f_vals_plt * ROUND_TRIP, 0.0)
pnl_net_plt = pd.Series(pnl_gross_plt - cost_plt, index=df.index)
cum_net_plt = pnl_net_plt.cumsum()
dd_plt      = cum_net_plt - cum_net_plt.cummax()

C_LINE   = '#5588cc'
C_POS    = '#c9a0dc'
C_NEG    = '#FFB6C1'
C_ACCENT = '#b0c4de'
C_GRID   = '#cccccc'
C_TICK   = '#808079'

fig = plt.figure(figsize=(16, 14), facecolor='#fafafa')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

def style_ax(ax):
    ax.set_facecolor('#fafafa')
    ax.grid(True, color='#cccccc', linestyle='--', linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=C_TICK)

# equity curve
ax1.plot(cum_net_plt.index, cum_net_plt.values, color=C_LINE, linewidth=1.4)
ax1.axhline(0, color=C_TICK, linewidth=0.8, linestyle=':')
ax1.fill_between(cum_net_plt.index, 0, cum_net_plt.values, alpha=0.10, color=C_LINE)
ax1.set_title(
    f"held-out equity curve  |  kelly {int(frac_plot*100)}%  |  capital USD {CAPITAL:,}",
    fontsize=12, pad=12
)
ax1.set_ylabel("cumulative PnL (USD)", fontsize=9, color=C_TICK)
ax1.tick_params(labelsize=9)
style_ax(ax1)

# drawdown
ax2.fill_between(dd_plt.index, dd_plt.values, 0, alpha=0.5, color=C_NEG)
ax2.plot(dd_plt.index, dd_plt.values, color=C_NEG, linewidth=0.9)
ax2.set_title("drawdown (USD)", fontsize=12, pad=12)
ax2.tick_params(labelsize=9)
style_ax(ax2)

# prob distribution
ax3.hist(prob[~bet_mask], bins=30, color=C_ACCENT, alpha=0.7,
         label="no bet", density=True, edgecolor='none')
ax3.hist(prob[bet_mask],  bins=30, color=C_LINE,   alpha=0.85,
         label="bet",    density=True, edgecolor='none')
ax3.axvline(MIN_PROB, color=C_NEG, linewidth=1.2, linestyle='--', label='min_prob')
ax3.set_title("predicted probability distribution", fontsize=12, pad=12)
ax3.legend(fontsize=9, framealpha=0.8)
ax3.tick_params(labelsize=9, colors=C_TICK)
style_ax(ax3)

# pnl per bet scatter by regime
regime_colors = {0: C_LINE, 1: C_POS, 2: C_NEG}
bet_idx   = np.where(bet_mask)[0]
bet_times = df.index[bet_idx]
bet_pnl   = pnl_net_plt.values[bet_idx]
bet_reg   = df["regime"].values[bet_idx]

for r in [0, 1, 2]:
    m = bet_reg == r
    ax4.scatter(
        bet_times[m], bet_pnl[m],
        s=6, alpha=0.65, color=regime_colors[r], edgecolors='none'
    )

ax4.axhline(0, color=C_TICK, linewidth=0.8, linestyle=':')
ax4.set_title("PnL per bet by regime", fontsize=12, pad=12)
ax4.legend(handles=[
    mpatches.Patch(color=C_LINE, label='low vol'),
    mpatches.Patch(color=C_POS,  label='mid vol'),
    mpatches.Patch(color=C_NEG,  label='high vol'),
], fontsize=9, framealpha=0.8)
ax4.tick_params(labelsize=9, colors=C_TICK)
style_ax(ax4)

# cumulative bets over time
cum_bets = pd.Series(bet_mask.astype(int), index=df.index).cumsum()
ax5.plot(cum_bets.index, cum_bets.values, color=C_LINE, linewidth=1.2)
ax5.fill_between(cum_bets.index, 0, cum_bets.values, alpha=0.12, color=C_LINE)
ax5.set_title("cumulative bets over time", fontsize=12, pad=12)
ax5.set_ylabel("n bets", fontsize=9, color=C_TICK)
ax5.tick_params(labelsize=9, colors=C_TICK)
style_ax(ax5)

# fix: sharpe_plot on all bars with sqrt(BARS_PER_YEAR)
sharpe_plot = (pnl_net_plt.mean() / pnl_net_plt.std() * np.sqrt(BARS_PER_YEAR)
               if pnl_net_plt.std() > 0 else 0)

plt.suptitle(
    f"falsification_loop  |  held-out validation  |  "
    f"{df.index[0].date()} – {df.index[-1].date()}  |  "
    f"AUC {auc:.3f}  |  Sharpe {sharpe_plot:.2f}",
    fontsize=12, fontweight='semibold', y=0.98, color='#2a2a2a'
)

plot_path = "results/held_out_plots.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"saved -> {plot_path}")

# ── save ──────────────────────────────────────────────────────────────────────
lines = [
    "held-out validation report (barrier-adjusted USD)",
    f"generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    "─" * 70,
    f"bars:           {len(df)}  ({df.index[0].date()} to {df.index[-1].date()})",
    f"features:       {feats}",
    f"regime gate:    {'on' if REGIME_GATE_ENABLED else 'off'}",
    f"capital:        USD {CAPITAL:,}",
    f"round-trip:     {ROUND_TRIP*10_000:.0f} bps",
    f"barrier_w mean: {barrier_w.mean()*100:.3f}%",
    "─" * 70,
    f"auc:            {auc:.4f}  ({verdict})",
    f"label balance:  {df['label'].mean():.3f}",
    "─" * 70,
]
for block in report_blocks:
    lines.extend(block)
lines += [
    "─" * 70,
    "note: single test split. do not optimize on this result.",
    "      pnl = f_frac * barrier_w * capital * payoff_sign - f_frac * capital * round_trip",
    "      payoff: +PT_SL[0]*vol for wins, -PT_SL[1]*vol for losses (asymmetric).",
    "      vol-pnl corr < 0 means losses concentrate in high-volatility bars.",
]

with open("results/held_out_report.txt", "w") as fh:
    fh.write("\n".join(lines))
print("saved -> results/held_out_report.txt")
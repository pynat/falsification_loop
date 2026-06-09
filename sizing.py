# sizing.py, fractional kelly bet sizing on cpcv probabilities
# requires: results/cpcv_probs_*.csv  (written by cpcv.py)
# outputs:  results/sizing_report.csv, results/sizing_report.txt,
#           results/sizing_plots.png

import warnings
warnings.filterwarnings("ignore")

import glob
import numpy as np
import pandas as pd
from cpcv import build_features, fit_hmm_regimes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from config import MIN_PROB, BARS_PER_YEAR, KELLY_FRACS, B, REGIME_GATE_ENABLED, PT_SL

os.makedirs("results", exist_ok=True)

# ── load all cpcv probability files ───────────────────────────────────────────
files = sorted(glob.glob("results/cpcv_probs_*.csv"))
if not files:
    raise FileNotFoundError("no cpcv_probs_*.csv found - run cpcv.py first")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.groupby("bar_idx").agg(
    timestamp=("timestamp", "first"),
    prob=("prob", "median"),
    label=("label", "first"),
).reset_index().sort_values("bar_idx")

print(f"loaded {len(df)} unique bars across {len(files)} folds")
print(f"prob range: {df['prob'].min():.3f} - {df['prob'].max():.3f}")
print(f"label balance: {df['label'].mean():.3f}")
print("─" * 70)

# ── regime gate ───────────────────────────────────────────────────────────────
df_bars = pd.read_csv("results/dollar_bars.csv", parse_dates=True, index_col=0)
df_feat = build_features(df_bars)
df_feat = df_feat.dropna(subset=["volatility_7b"])
regimes = fit_hmm_regimes(df_feat["volatility_7b"].values)
reg_ser = pd.Series(regimes, index=df_feat.index, name="regime")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.merge(
    reg_ser.reset_index().rename(columns={"index": "timestamp"}),
    on="timestamp", how="left"
)
df["regime"] = df["regime"].fillna(1)

n_gated = (df["regime"] == 2).sum()
if REGIME_GATE_ENABLED:
    print(f"regime gate: ON  ->  {n_gated} high-vol bars blocked")
else:
    print(f"regime gate: OFF ->  {n_gated} high-vol bars included (no blocking)")
print(f"regime distribution: low={(df['regime']==0).sum()}  mid={(df['regime']==1).sum()}  high={n_gated}")
print("─" * 70)

# ── kelly calculation ─────────────────────────────────────────────────────────
def kelly_f(p, b=B):
    # raw kelly fraction: clip at 0 (no shorts), no upper cap
    return max((p * b - (1 - p)) / b, 0.0)

df["f_full"] = df["prob"].apply(kelly_f)  # raw kelly, lower-bounded at 0

if REGIME_GATE_ENABLED:
    df["bet"] = (df["prob"] > MIN_PROB) & (df["regime"] != 2)
else:
    df["bet"] = df["prob"] > MIN_PROB

# final fraction = f_full * kelly_frac, zeroed out where no bet
for frac in KELLY_FRACS:
    col = f"f_{int(frac * 100)}pct"
    df[col] = np.where(df["bet"], df["f_full"] * frac, 0.0)

print(f"bars with bet signal (prob > {MIN_PROB}): {df['bet'].sum()} / {len(df)}  "
      f"({df['bet'].mean():.1%})")
print(f"mean f_full (when betting): {df.loc[df['bet'], 'f_full'].mean():.4f}")
print(f"max f_full (when betting):  {df.loc[df['bet'], 'f_full'].max():.4f}")
print("─" * 70)

# ── simulated pnl per kelly fraction ─────────────────────────────────────────
results  = {}
pnl_data = {}

payoff = np.where(df["label"] == 1, PT_SL[0], -PT_SL[1])

for frac in KELLY_FRACS:
    col     = f"f_{int(frac * 100)}pct"
    pnl     = df[col] * payoff
    cum_pnl = pnl.cumsum()

    n_bets    = df["bet"].sum()
    total_pnl = pnl.sum()
    mean_pnl  = pnl[df["bet"]].mean()

    sharpe = (pnl.mean() / pnl.std() * np.sqrt(BARS_PER_YEAR)
              if pnl.std() > 0 else 0)

    roll_max = cum_pnl.cummax()
    drawdown = cum_pnl - roll_max
    max_dd   = drawdown.min()

    results[frac] = {
        "kelly_fraction":    frac,
        "n_bets":            int(n_bets),
        "total_pnl":         round(total_pnl, 4),
        "mean_pnl_per_bet":  round(mean_pnl, 6),
        "annualized_sharpe": round(sharpe, 4),
        "max_drawdown":      round(max_dd, 4),
    }
    pnl_data[frac] = {"pnl": pnl, "cum_pnl": cum_pnl, "drawdown": drawdown}

    print(f"kelly {int(frac*100)}%:  "
          f"n_bets={n_bets}  "
          f"total_pnl={total_pnl:.4f}  "
          f"sharpe={sharpe:.3f}  "
          f"max_dd={max_dd:.4f}")

print("─" * 70)

# ── plots ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor='#fafafa')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

C_LINE   = '#5588cc'
C_POS    = '#c9a0dc'
C_NEG    = '#FFB6C1'
C_ACCENT = '#b0c4de'
C_GRID   = '#cccccc'
C_TICK   = '#808079'

# maps frac -> color; supports any two fracs in KELLY_FRACS
color_list = [C_LINE, C_POS, C_NEG, C_ACCENT]
colors = {frac: color_list[i] for i, frac in enumerate(KELLY_FRACS)}

def style_ax(ax):
    ax.set_facecolor('#fafafa')
    ax.grid(True, color=C_GRID, linestyle='--', linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=C_TICK)

ax1 = fig.add_subplot(gs[0, :])
for frac, d in pnl_data.items():
    ax1.plot(d["cum_pnl"].values, label=f"kelly {int(frac*100)}%",
             color=colors[frac], linewidth=1.2)
ax1.axhline(0, color=C_TICK, linewidth=0.8, linestyle=':')
gate_label = "regime-gated" if REGIME_GATE_ENABLED else "no regime gate"
ax1.set_title(f"cumulative pnl (gross, symmetric triple-barrier units, {gate_label})",
              fontsize=12, pad=12)
ax1.set_xlabel("bar index", fontsize=9, color=C_TICK)
ax1.set_ylabel("cumulative pnl (f units)", fontsize=9, color=C_TICK)
ax1.legend(fontsize=9, framealpha=0.8)
style_ax(ax1)

ax2 = fig.add_subplot(gs[1, 0])
for frac, d in pnl_data.items():
    ax2.fill_between(range(len(d["drawdown"])), d["drawdown"].values,
                     alpha=0.35, label=f"kelly {int(frac*100)}%", color=colors[frac])
ax2.set_title("drawdown over time", fontsize=12, pad=12)
ax2.set_xlabel("bar index", fontsize=9, color=C_TICK)
ax2.set_ylabel("drawdown (f units)", fontsize=9, color=C_TICK)
ax2.legend(fontsize=9, framealpha=0.8)
style_ax(ax2)

ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(df["prob"], bins=40, color=C_ACCENT, alpha=0.80, edgecolor='none')
ax3.axvline(MIN_PROB, color=C_NEG, linestyle='--', linewidth=1.2,
            label=f"bet threshold ({MIN_PROB})")
ax3.axvline(0.5, color=C_LINE, linestyle='--', linewidth=1.0, label="0.5")
ax3.set_title("model probability distribution (median across folds)",
              fontsize=12, pad=12)
ax3.set_xlabel("predicted probability", fontsize=9, color=C_TICK)
ax3.set_ylabel("bar count", fontsize=9, color=C_TICK)
ax3.legend(fontsize=9, framealpha=0.8)
style_ax(ax3)

plt.suptitle("sizing analysis - fractional kelly", fontsize=12,
             fontweight='semibold', y=0.98, color='#2a2a2a')

plt.savefig("results/sizing_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved -> results/sizing_plots.png")
print("─" * 70)

# ── save ──────────────────────────────────────────────────────────────────────
df.to_csv("results/sizing_report.csv", index=False)

report_df    = pd.DataFrame(results.values())
report_lines = [
    "sizing report (fractional kelly)",
    f"generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    "─" * 70,
    f"bet threshold: prob > {MIN_PROB}",
    f"win/loss ratio b: {B}  (symmetric triple barrier)",
    f"regime gate: {'on' if REGIME_GATE_ENABLED else 'off'}",
    "",
    report_df.to_string(index=False),
    "",
    "─" * 70,
    "note: pnl is in units of f*. does not include transaction costs.",
    "      apply sizing to real returns only after cost-adjusted sharpe check.",
]

with open("results/sizing_report.txt", "w") as f:
    f.write("\n".join(report_lines))

print("saved -> results/sizing_report.csv")
print("saved -> results/sizing_report.txt")
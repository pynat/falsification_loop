# cost_analysis.py  -  post-cost sharpe estimation
# requires: results/cpcv_performance.csv, results/sizing_report.csv
# outputs:  appended section in results/analysis_report.txt,
#           results/cost_plots.png

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from config import FEE_BPS, SLIPPAGE_BPS, BARS_PER_YEAR, KELLY_FRACS

os.makedirs("results", exist_ok=True)

ROUND_TRIP = 2 * (FEE_BPS + SLIPPAGE_BPS) / 10_000

print("cost-adjusted sharpe analysis")
print("─" * 70)
print(f"  taker fee:  {FEE_BPS} bps/side")
print(f"  slippage:   {SLIPPAGE_BPS} bps/side")
print(f"  round-trip: {ROUND_TRIP * 10_000:.0f} bps = {ROUND_TRIP:.4f}")
print("─" * 70)

try:
    sz = pd.read_csv("results/sizing_report.csv")
except FileNotFoundError:
    raise FileNotFoundError("run cpcv.py and sizing.py first")

n_bars  = len(sz)
n_bets  = sz["bet"].sum()

sz["timestamp"] = pd.to_datetime(sz["timestamp"])
n_years         = (sz["timestamp"].max() - sz["timestamp"].min()).days / 365.25
trades_per_year = n_bets / n_years if n_years > 0 else n_bets
n_bets_per_year = sz["bet"].mean() * BARS_PER_YEAR

print(f"  total bars:        {n_bars}")
print(f"  bets placed:       {n_bets}")
print(f"  period:            {n_years:.2f} years")
print(f"  trades/year:       {trades_per_year:.1f}")
print("─" * 70)

# ── cost-adjusted sharpe + collect plot data ───────────────────────────────────

plot_data   = {}
cost_block  = f"""
─────────────────────────────────────────────────────────────────────
7. cost-adjusted sharpe
   cost model: {FEE_BPS} bps taker fee + {SLIPPAGE_BPS} bps slippage per side
   round-trip: {ROUND_TRIP * 10_000:.0f} bps
   trades/year (estimated): {trades_per_year:.1f}
"""

for frac in KELLY_FRACS:
    col = f"f_{int(frac * 100)}pct"
    if col not in sz.columns:
        continue

    direction    = np.where(sz["label"] == 1, 1.0, -1.0)
    pnl_gross    = sz[col] * direction
    cost_per_bar = np.where(sz["bet"], sz[col] * ROUND_TRIP, 0.0)
    pnl_net      = pnl_gross - cost_per_bar
    bet_mask     = sz["bet"].astype(bool)

    gross_sharpe = (pnl_gross[bet_mask].mean() / pnl_gross[bet_mask].std()
                    * np.sqrt(n_bets_per_year)) if pnl_gross[bet_mask].std() > 0 else 0
    net_sharpe   = (pnl_net[bet_mask].mean() / pnl_net[bet_mask].std()
                    * np.sqrt(n_bets_per_year)) if pnl_net[bet_mask].std() > 0 else 0
    cum_gross    = pnl_gross.cumsum()
    cum_net      = pnl_net.cumsum()
    max_dd_net   = (cum_net - cum_net.cummax()).min()
    dd_net       = cum_net - cum_net.cummax()

    plot_data[frac] = {
        "cum_gross":   cum_gross,
        "cum_net":     cum_net,
        "dd_net":      dd_net,
        "gross_sharpe": gross_sharpe,
        "net_sharpe":   net_sharpe,
    }

    print(f"kelly {int(frac*100)}%:")
    print(f"  gross sharpe:     {gross_sharpe:.3f}")
    print(f"  net sharpe:       {net_sharpe:.3f}  (after {ROUND_TRIP*10_000:.0f} bps round-trip)")
    print(f"  sharpe decay:     {gross_sharpe - net_sharpe:.3f}")
    print(f"  max drawdown net: {max_dd_net:.4f}")
    print(f"  verdict:          {'survives costs' if net_sharpe > 0.5 else 'marginal after costs' if net_sharpe > 0 else 'does not survive costs'}")
    print()

    cost_block += (
        f"   kelly {int(frac*100)}%:  "
        f"gross sharpe={gross_sharpe:.3f}  "
        f"net sharpe={net_sharpe:.3f}\n"
    )

cost_block += "─" * 70

# ── plots ─────────────────────────────────────────────────────────────────────
# 3 panels: gross vs net equity per kelly fraction + sharpe comparison bar chart

n_fracs = len(plot_data)
fig     = plt.figure(figsize=(14, 4 + 4 * n_fracs))
gs      = gridspec.GridSpec(n_fracs + 1, 1, figure=fig, hspace=0.45)

colors = {0.25: "#2196F3", 0.50: "#FF5722"}

for i, (frac, d) in enumerate(plot_data.items()):
    ax = fig.add_subplot(gs[i])
    ax.plot(d["cum_gross"].values, color=colors[frac], linewidth=1.0,
            alpha=0.5, label="gross")
    ax.plot(d["cum_net"].values,   color=colors[frac], linewidth=1.4,
            label="net (after costs)")
    ax.fill_between(range(len(d["dd_net"])), d["dd_net"].values,
                    alpha=0.15, color=colors[frac])
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(
        f"kelly {int(frac*100)}%  |  "
        f"gross sharpe={d['gross_sharpe']:.3f}  "
        f"net sharpe={d['net_sharpe']:.3f}  "
        f"(cost drag={d['gross_sharpe']-d['net_sharpe']:.3f})"
    )
    ax.set_xlabel("bar index")
    ax.set_ylabel("cumulative pnl (f units)")
    ax.legend(loc="upper left")

# sharpe comparison bar chart
ax_bar = fig.add_subplot(gs[n_fracs])
fracs       = list(plot_data.keys())
gross_vals  = [plot_data[f]["gross_sharpe"] for f in fracs]
net_vals    = [plot_data[f]["net_sharpe"]   for f in fracs]
x           = np.arange(len(fracs))
width       = 0.35

ax_bar.bar(x - width/2, gross_vals, width, label="gross sharpe",
           color=[colors[f] for f in fracs], alpha=0.5)
ax_bar.bar(x + width/2, net_vals,   width, label="net sharpe",
           color=[colors[f] for f in fracs], alpha=1.0)
ax_bar.axhline(0.5, color="red", linestyle="--", linewidth=1.0,
               label="sharpe=0.5 (minimum viable)")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f"kelly {int(f*100)}%" for f in fracs])
ax_bar.set_title("gross vs net sharpe by kelly fraction")
ax_bar.set_ylabel("annualized sharpe")
ax_bar.legend()

plt.suptitle(
    f"cost analysis  |  round-trip={ROUND_TRIP*10_000:.0f} bps  |  "
    f"{trades_per_year:.0f} trades/year",
    fontsize=12, y=1.01
)
plt.savefig("results/cost_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved -> results/cost_plots.png")
print("─" * 70)

# ── append to analysis_report.txt ─────────────────────────────────────────────
if os.path.exists("results/analysis_report.txt"):
    with open("results/analysis_report.txt", "a") as f:
        f.write(cost_block)
    print("appended cost section -> results/analysis_report.txt")
else:
    with open("results/cost_analysis.txt", "w") as f:
        f.write(cost_block)
    print("saved -> results/cost_analysis.txt")
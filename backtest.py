# backtest.py, realistic pnl simulation with actual barrier widths
# requires: results/sizing_report.csv, results/dollar_bars.csv
# outputs:  results/backtest_report.csv, results/backtest_report.txt,
#           results/backtest_plots.png
#
# key distinction vs sizing.py:
#   sizing.py:   pnl = f * payoff         (f-units, asymmetric PT/SL)
#   backtest.py: pnl = f * barrier_w * payoff_mult  (vol-scaled, USD)
#
# barrier_w for PT trades  = PT_SL[0] * vol_20b
# barrier_w for SL trades  = PT_SL[1] * vol_20b
# sharpe computed on all bars with sqrt(BARS_PER_YEAR) — no bet-filter inflation

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from config import (
    FEE_BPS, SLIPPAGE_BPS, BARS_PER_YEAR, KELLY_FRACS, PT_SL, CAPITAL
)

os.makedirs("results", exist_ok=True)

ROUND_TRIP = 2 * (FEE_BPS + SLIPPAGE_BPS) / 10_000

print("realistic backtest (barrier-width adjusted)")
print("─" * 70)
print(f"  capital:       USD {CAPITAL:,}")
print(f"  PT_SL:         {PT_SL}")
print(f"  round-trip:    {ROUND_TRIP * 10_000:.0f} bps")
print("─" * 70)

# ── load data ─────────────────────────────────────────────────────────────────

sz   = pd.read_csv("results/sizing_report.csv")
bars = pd.read_csv("results/dollar_bars.csv", parse_dates=True, index_col=0)

sz["timestamp"] = pd.to_datetime(sz["timestamp"])
n_years         = (sz["timestamp"].max() - sz["timestamp"].min()).days / 365.25
n_bets_per_year = sz["bet"].mean() * BARS_PER_YEAR  # info only
bet_mask        = sz["bet"].astype(bool)

# ── compute barrier widths from actual bar data ───────────────────────────────
# pt trades exit at PT_SL[0] * vol, sl trades exit at PT_SL[1] * vol

bars["log_ret"] = np.log(bars["close"] / bars["close"].shift(1))
bars["vol_20b"] = bars["log_ret"].rolling(20).std()
bars["bw_pt"]   = PT_SL[0] * bars["vol_20b"]   # barrier width when pt hit
bars["bw_sl"]   = PT_SL[1] * bars["vol_20b"]   # barrier width when sl hit

merged = sz.merge(
    bars[["close", "vol_20b", "bw_pt", "bw_sl"]].reset_index().rename(
        columns={"timestamp": "timestamp"}),
    on="timestamp", how="left"
)

# effective barrier width per trade: PT_SL[0]*vol for winners, PT_SL[1]*vol for losers
merged["barrier_w"] = np.where(
    merged["label"] == 1,
    merged["bw_pt"],
    merged["bw_sl"]
)

print(f"  barrier_width mean:   {merged['barrier_w'].mean()*100:.3f}%")
print(f"  barrier_width median: {merged['barrier_w'].median()*100:.3f}%")
print(f"  barrier_width p25:    {merged['barrier_w'].quantile(0.25)*100:.3f}%")
print(f"  barrier_width p75:    {merged['barrier_w'].quantile(0.75)*100:.3f}%")
print("─" * 70)

# ── simulate pnl ──────────────────────────────────────────────────────────────

results  = {}
pnl_data = {}

for frac in KELLY_FRACS:
    col = f"f_{int(frac * 100)}pct"

    # asymmetric payoff: +PT_SL[0]*vol for wins, -PT_SL[1]*vol for losses
    payoff_mult = np.where(merged["label"] == 1, merged["bw_pt"], -merged["bw_sl"])

    pnl_gross = CAPITAL * merged[col] * payoff_mult
    cost      = np.where(sz["bet"], CAPITAL * merged[col] * ROUND_TRIP, 0.0)
    pnl_net   = pnl_gross - cost

    cum_net    = pnl_net.cumsum()
    dd         = (cum_net - cum_net.cummax()) / CAPITAL * 100
    max_dd_pct = dd.min()

    # fix: sharpe on all bars, sqrt(BARS_PER_YEAR)
    sharpe_net_usd = (pnl_net.mean() / pnl_net.std() * np.sqrt(BARS_PER_YEAR)
                      if pnl_net.std() > 0 else 0)

    annual_net  = pnl_net.sum() / n_years
    annual_ret  = annual_net / CAPITAL * 100
    calmar      = annual_ret / abs(max_dd_pct) if max_dd_pct != 0 else np.nan
    avg_pos_usd = (CAPITAL * merged.loc[bet_mask, col]).mean()
    avg_pnl_bet = pnl_net[bet_mask].mean()

    # monthly returns
    monthly       = pnl_net.copy()
    monthly.index = pd.to_datetime(sz["timestamp"])
    monthly_pnl   = monthly.resample("ME").sum()
    monthly_ret   = (monthly_pnl / CAPITAL * 100).round(2)
    win_months    = (monthly_ret > 0).sum()
    total_months  = len(monthly_ret)

    # f-unit gross vs net for cost drag analysis (asymmetric payoff in f-units)
    payoff_f    = np.where(merged["label"] == 1, PT_SL[0], -PT_SL[1])
    pnl_gross_f = merged[col] * payoff_f
    cost_f      = np.where(sz["bet"], merged[col] * ROUND_TRIP, 0.0)
    pnl_net_f   = pnl_gross_f - cost_f

    gross_sharpe = (pnl_gross_f.mean() / pnl_gross_f.std() * np.sqrt(BARS_PER_YEAR)
                    if pnl_gross_f.std() > 0 else 0)
    net_sharpe_f = (pnl_net_f.mean() / pnl_net_f.std() * np.sqrt(BARS_PER_YEAR)
                    if pnl_net_f.std() > 0 else 0)
    sharpe_decay = gross_sharpe - net_sharpe_f
    cum_gross_f  = pd.Series(pnl_gross_f).cumsum()
    cum_net_f    = pd.Series(pnl_net_f).cumsum()

    cost_verdict = (
        "survives costs"       if net_sharpe_f > 0.5
        else "marginal"        if net_sharpe_f > 0
        else "does not survive costs"
    )

    results[frac] = {
        "kelly_fraction":  frac,
        "capital":         CAPITAL,
        "annual_net_usd":  round(annual_net, 0),
        "annual_ret_pct":  round(annual_ret, 2),
        "sharpe_net_usd":  round(sharpe_net_usd, 3),
        "sharpe_net_f":    round(net_sharpe_f, 3),
        "gross_sharpe":    round(gross_sharpe, 3),
        "sharpe_decay":    round(sharpe_decay, 3),
        "cost_verdict":    cost_verdict,
        "max_dd_pct":      round(max_dd_pct, 2),
        "calmar":          round(calmar, 3),
        "avg_pos_usd":     round(avg_pos_usd, 0),
        "avg_pnl_per_bet": round(avg_pnl_bet, 2),
        "win_months":      f"{win_months}/{total_months}",
    }
    pnl_data[frac] = {
        "pnl_net":     pnl_net,
        "cum_net":     cum_net,
        "dd":          dd,
        "monthly":     monthly_ret,
        "cum_gross_f": cum_gross_f,
        "cum_net_f":   cum_net_f,
    }

    print(f"kelly {int(frac*100)}%:")
    print(f"  avg position size:  USD {avg_pos_usd:,.0f}  ({avg_pos_usd/CAPITAL:.1%} of capital)")
    print(f"  avg pnl per bet:    USD {avg_pnl_bet:,.2f}")
    print(f"  annual net pnl:     USD {annual_net:,.0f}")
    print(f"  annual return:      {annual_ret:.2f}%")
    print(f"  sharpe (net, usd):  {sharpe_net_usd:.3f}")
    print(f"  sharpe (net, f):    {net_sharpe_f:.3f}")
    print(f"  gross sharpe (f):   {gross_sharpe:.3f}")
    print(f"  sharpe decay:       {sharpe_decay:.3f}")
    print(f"  cost verdict:       {cost_verdict}")
    print(f"  max drawdown:       {max_dd_pct:.2f}%")
    print(f"  calmar ratio:       {calmar:.3f}")
    print(f"  winning months:     {win_months}/{total_months}")
    print()

# ── plots ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 18), facecolor='#fafafa')
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

C_LINE   = '#5588cc'
C_POS    = '#c9a0dc'
C_NEG    = '#FFB6C1'
C_ACCENT = '#b0c4de'
C_GRID   = '#cccccc'
C_TICK   = '#808079'

color_list = [C_LINE, C_POS, C_NEG, C_ACCENT]
colors = {frac: color_list[i] for i, frac in enumerate(KELLY_FRACS)}

def style_ax(ax):
    ax.set_facecolor('#fafafa')
    ax.grid(True, color=C_GRID, linestyle='--', linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=C_TICK)

# 1. equity curve (USD)
ax1 = fig.add_subplot(gs[0, :])
for frac, d in pnl_data.items():
    r = results[frac]
    ax1.plot(
        d["cum_net"].values,
        color=colors[frac],
        linewidth=1.3,
        label=f"kelly {int(frac*100)}%  |  SR={r['sharpe_net_f']}  |  {r['annual_ret_pct']:.1f}%/yr"
    )
ax1.axhline(0, color=C_TICK, linewidth=0.8, linestyle=':')
ax1.set_title(f"equity curve  (capital USD {CAPITAL:,}, no compounding)", fontsize=12, pad=12)
ax1.set_xlabel("bar index", fontsize=9, color=C_TICK)
ax1.set_ylabel("cumulative pnl (USD)", fontsize=9, color=C_TICK)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(fontsize=9, framealpha=0.8)
style_ax(ax1)

# 2. drawdown (%)
ax2 = fig.add_subplot(gs[1, 0])
for frac, d in pnl_data.items():
    ax2.fill_between(
        range(len(d["dd"])), d["dd"].values,
        alpha=0.35, color=colors[frac], label=f"kelly {int(frac*100)}%"
    )
ax2.set_title("drawdown (% of capital)", fontsize=12, pad=12)
ax2.set_xlabel("bar index", fontsize=9, color=C_TICK)
ax2.set_ylabel("drawdown %", fontsize=9, color=C_TICK)
ax2.legend(fontsize=9, framealpha=0.8)
style_ax(ax2)

# 3. monthly returns (kelly 25%)
ax3 = fig.add_subplot(gs[1, 1])
frac_main   = KELLY_FRACS[0]
monthly_ret = pnl_data[frac_main]["monthly"]
ax3.bar(
    range(len(monthly_ret)), monthly_ret.values,
    color=[C_LINE if v > 0 else C_NEG for v in monthly_ret.values]
)
ax3.axhline(0, color=C_TICK, linewidth=0.8, linestyle=':')
ax3.set_title(f"monthly returns %  (kelly {int(frac_main*100)}%)", fontsize=12, pad=12)
ax3.set_ylabel("return % of capital", fontsize=9, color=C_TICK)
ax3.set_xlabel("month", fontsize=9, color=C_TICK)
style_ax(ax3)

# 4. barrier width distribution
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(merged["barrier_w"].dropna() * 100, bins=40, color=C_ACCENT, alpha=0.8, edgecolor='none')
ax4.axvline(
    merged["barrier_w"].mean() * 100, color=C_NEG, linestyle='--', linewidth=1.2,
    label=f"mean={merged['barrier_w'].mean()*100:.2f}%"
)
ax4.set_title("barrier width distribution (% of price)", fontsize=12, pad=12)
ax4.set_xlabel("barrier width %", fontsize=9, color=C_TICK)
ax4.set_ylabel("bar count", fontsize=9, color=C_TICK)
ax4.legend(fontsize=9, framealpha=0.8)
style_ax(ax4)

# 5. pnl per bet distribution (kelly 25%)
ax5 = fig.add_subplot(gs[2, 1])
pnl_bets = pnl_data[KELLY_FRACS[0]]["pnl_net"][bet_mask]
ax5.hist(pnl_bets.values, bins=40, color=C_LINE, alpha=0.85, edgecolor='none')
ax5.axvline(0, color=C_NEG, linestyle='--', linewidth=1.0)
ax5.axvline(
    pnl_bets.mean(), color=C_LINE, linestyle='--', linewidth=1.2,
    label=f"mean=${pnl_bets.mean():.2f}"
)
ax5.set_title(f"pnl per bet (USD)  (kelly {int(KELLY_FRACS[0]*100)}%)", fontsize=12, pad=12)
ax5.set_xlabel("pnl per bet (USD)", fontsize=9, color=C_TICK)
ax5.set_ylabel("count", fontsize=9, color=C_TICK)
ax5.legend(fontsize=9, framealpha=0.8)
style_ax(ax5)

# 6. gross vs net in f-units
ax6 = fig.add_subplot(gs[3, :])
for frac, d in pnl_data.items():
    r = results[frac]
    ax6.plot(
        d["cum_gross_f"].values, color=colors[frac], linewidth=1.0, alpha=0.4,
        linestyle='--', label=f"kelly {int(frac*100)}% gross (SR={r['gross_sharpe']})"
    )
    ax6.plot(
        d["cum_net_f"].values, color=colors[frac], linewidth=1.3,
        label=f"kelly {int(frac*100)}% net (decay={r['sharpe_decay']:+.3f})"
    )
ax6.axhline(0, color=C_TICK, linewidth=0.8, linestyle=':')
ax6.set_title(
    f"gross vs net pnl (f-units)  |  round-trip={ROUND_TRIP*10_000:.0f} bps  |  cost drag",
    fontsize=12, pad=12
)
ax6.set_xlabel("bar index", fontsize=9, color=C_TICK)
ax6.set_ylabel("cumulative pnl (f units)", fontsize=9, color=C_TICK)
ax6.legend(fontsize=9, framealpha=0.8)
style_ax(ax6)

plt.suptitle(
    f"realistic backtest  |  capital USD {CAPITAL:,}  |  "
    f"PT_SL={PT_SL}  |  {n_years:.1f} years",
    fontsize=12, fontweight='semibold', y=0.98, color='#2a2a2a'
)
plt.savefig("results/backtest_plots.png", dpi=150, bbox_inches="tight")
plt.close()

print("saved -> results/backtest_plots.png")
print("─" * 70)

# ── save ──────────────────────────────────────────────────────────────────────

pd.DataFrame(results.values()).to_csv("results/backtest_report.csv", index=False)

lines = [
    "backtest report (barrier-width adjusted, no compounding)",
    f"generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
    f"capital: USD {CAPITAL:,}  |  PT_SL: {PT_SL}  |  round-trip: {ROUND_TRIP*10_000:.0f} bps",
    "─" * 70,
    f"barrier_width: mean={merged['barrier_w'].mean()*100:.3f}%  "
    f"median={merged['barrier_w'].median()*100:.3f}%",
    "",
]
for frac, r in results.items():
    lines += [
        f"kelly {int(frac*100)}%:",
        f"  annual return:      {r['annual_ret_pct']:.2f}%  (USD {r['annual_net_usd']:,.0f}/year)",
        f"  sharpe (net, usd):  {r['sharpe_net_usd']}",
        f"  sharpe (net, f):    {r['sharpe_net_f']}",
        f"  gross sharpe (f):   {r['gross_sharpe']}",
        f"  sharpe decay:       {r['sharpe_decay']}  (cost drag)",
        f"  cost verdict:       {r['cost_verdict']}",
        f"  max drawdown:       {r['max_dd_pct']:.2f}%",
        f"  calmar:             {r['calmar']}",
        f"  avg position:       USD {r['avg_pos_usd']:,.0f}",
        f"  avg pnl/bet:        USD {r['avg_pnl_per_bet']:.2f}",
        f"  winning months:     {r['win_months']}",
        "",
    ]
lines.append("─" * 70)

with open("results/backtest_report.txt", "w") as f:
    f.write("\n".join(lines))

print("saved -> results/backtest_report.csv")
print("saved -> results/backtest_report.txt")
print("─" * 70)


# label balance on bet bars
lb_all  = merged["label"].mean()
lb_bets = merged.loc[bet_mask, "label"].mean()
print(f"  label balance (all):      {lb_all:.3f}")
print(f"  label balance (bet bars): {lb_bets:.3f}")
print(f"  breakeven hit rate:       {PT_SL[1]/(PT_SL[0]+PT_SL[1]):.3f}")
print(f"  edge margin:              {lb_bets - PT_SL[1]/(PT_SL[0]+PT_SL[1]):.3f}")
print("─" * 70)

# hit rate on bet-bars
hit_rate = (merged.loc[bet_mask, "label"] == 1).mean()
print(f"hit rate (bet bars): {hit_rate:.3f}")
print(f"breakeven hit rate:  {PT_SL[1] / (PT_SL[0] + PT_SL[1]):.3f}")
print(f"edge margin:         {hit_rate - PT_SL[1]/(PT_SL[0]+PT_SL[1]):.3f}")

# drawdown-periods
cum = pnl_data[KELLY_FRACS[0]]["cum_net"]
in_dd = cum < cum.cummax()
dd_lengths = in_dd.astype(int).groupby((~in_dd).cumsum()).sum()
print(f"\nlongest drawdown period: {dd_lengths.max()} bars")
print(f"avg drawdown period:     {dd_lengths[dd_lengths>0].mean():.0f} bars")
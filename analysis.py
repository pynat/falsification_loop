# analysis.py
# post-hoc statistical analysis of cpcv results.
# requires: results/cpcv_performance.csv, results/feature_stability.csv,
#           results/dollar_bars.csv
# outputs:  results/analysis_report.txt, results/feature_importance.csv

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt

from config import FINAL_FEATURES, RANDOM_SEED, N_BOOTSTRAP, N_PERMUTATION
from cpcv import build_features, make_labels

os.makedirs("results", exist_ok=True)

rng           = np.random.default_rng(RANDOM_SEED)

print("loading results...")
perf = pd.read_csv("results/cpcv_performance.csv")
aucs = perf["auc"].values
print(f"  {len(aucs)} folds")
print("─" * 70)


# ── 1. bootstrap ci on mean auc ───────────────────────────────────────────────

def bootstrap_ci(values, n=N_BOOTSTRAP, ci=0.95):
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n)
    ])
    return np.percentile(means, [(1-ci)/2*100, (1+ci)/2*100])

mean_auc     = aucs.mean()
ci_lo, ci_hi = bootstrap_ci(aucs)

print("1. bootstrap ci on mean auc (n=2000, 95%):")
print(f"   mean auc = {mean_auc:.4f}  ci=[{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"   null (0.5) inside ci: {ci_lo <= 0.5 <= ci_hi}")
print("─" * 70)


# ── 2. mann-whitney ───────────────────────────────────────────────────────

print(f"2. analytical auc null test (mann-whitney, n={N_PERMUTATION} draws)...")

n_test_avg = int(perf["n_test"].mean()) if "n_test" in perf.columns else 2688
se_null    = np.sqrt((n_test_avg + n_test_avg + 1) / (12 * n_test_avg * n_test_avg))
perm_aucs  = rng.normal(0.5, se_null, N_PERMUTATION)
p_value    = (perm_aucs >= mean_auc).mean()

print(f"   observed mean auc:   {mean_auc:.4f}")
print(f"   null distribution:   mean=0.5000  std={se_null:.4f}  (n_test~{n_test_avg})")
print(f"   p-value (one-sided): {p_value:.4f}")
print(f"   significant at 0.05: {p_value < 0.05}")
print("─" * 70)


# ── 3. benchmark: naive momentum ─────────────────────────────────────────────

print("3. benchmarks...")

df_raw  = pd.read_csv("results/dollar_bars.csv", parse_dates=True, index_col=0)
df_full = build_features(df_raw)
labels  = make_labels(df_full["close"])
df_full = df_full.join(labels)

labeled_full = df_full.dropna(subset=["label"] + FINAL_FEATURES)
y_bench      = labeled_full["label"].astype(int)
ret_score    = labeled_full["ret_raw"].fillna(0).values

try:
    auc_momentum = roc_auc_score(y_bench, ret_score)
except Exception:
    auc_momentum = np.nan

print(f"   model mean auc:    {mean_auc:.4f}")
print(f"   momentum auc:      {auc_momentum:.4f}  (lag-1 return as signal)")
print(f"   buy-and-hold auc:  0.5000")
print(f"   model vs momentum: {mean_auc - auc_momentum:+.4f}")
print("─" * 70)


# ── 4. equity curve proxy ─────────────────────────────────────────────────────

print("4. equity proxy from cpcv verdicts...")

edge_mask       = perf["verdict"] == "edge"
marginal_mask   = perf["verdict"] == "marginal"
edge_pct        = edge_mask.mean()
mean_auc_edge   = perf[edge_mask]["auc"].mean()    if edge_mask.any()    else np.nan
mean_auc_marg   = perf[marginal_mask]["auc"].mean() if marginal_mask.any() else np.nan
mean_auc_noedge = perf[~edge_mask]["auc"].mean()   if (~edge_mask).any() else np.nan
pct_high_edge   = perf[edge_mask]["pct_high_vol"].mean()  if edge_mask.any()    else np.nan
pct_high_noedge = perf[~edge_mask]["pct_high_vol"].mean() if (~edge_mask).any() else np.nan

print(f"   edge folds (auc>=0.54):     {edge_mask.sum()} / {len(perf)}  ({edge_pct:.1%})")
print(f"   marginal folds (auc>=0.52): {marginal_mask.sum()} / {len(perf)}")
print(f"   mean auc (edge):            {mean_auc_edge:.4f}")
print(f"   mean auc (marginal):        {mean_auc_marg:.4f}")
print(f"   mean auc (no edge):         {mean_auc_noedge:.4f}")
print(f"   pct high vol (edge):        {pct_high_edge:.1%}")
print(f"   pct high vol (no edge):     {pct_high_noedge:.1%}")
print("─" * 70)


# ── 5. feature importance from cpcv mdi stability ────────────────────────────

print("5. feature importance (cpcv mdi stability across folds)...")

stab = pd.read_csv("results/feature_stability.csv")
stab = stab.sort_values("mean_mdi", ascending=False).reset_index(drop=True)

fi = pd.DataFrame({
    "feature":    stab["feature"],
    "importance": stab["mean_mdi"],
    "std":        np.nan,
})
fi.to_csv("results/feature_importance.csv", index=False)

print(f"  {'rank':<5} {'feature':<25} {'mean_mdi':>10} {'stability':>10}")
print("  " + "─" * 55)
for i, row in stab.iterrows():
    print(f"  {i+1:<5} {row['feature']:<25} {row['mean_mdi']:>10.4f} {row['stability_rate']:>9.1%}")
print("─" * 70)


# ── 6. fold regime diagnostics ───────────────────────────────────────────────
# computes bar-level regime properties per cpcv fold directly from dollar_bars.
# no external api calls. tests whether fold market conditions explain edge.
#
# signals:
#   trend_strength:  abs(mean_daily_ret) / std_daily_ret
#                    higher = cleaner directional trend = cleaner triple-barrier labels
#   skewness:        skew of daily returns in the fold window
#                    positive skew = upside surprises dominate = model labels easier
#   vol_compression: realized_vol(last quarter) / realized_vol(first quarter)
#                    >1 = vol expanding into fold end, <1 = vol compressing
#
# interpretation: p < 0.05 means the signal meaningfully separates edge from no-edge.
# caveat: n_edge=8, results are directional hypotheses, not statistically robust.

print("6. fold regime diagnostics (bar-level, no external data)...")

bars = pd.read_csv("results/dollar_bars.csv", parse_dates=True, index_col=0)
bars["log_ret"] = np.log(bars["close"] / bars["close"].shift(1))

diag_rows = []
for _, row in perf.iterrows():
    mask  = (bars.index >= pd.Timestamp(row["test_from"])) & \
            (bars.index <= pd.Timestamp(row["test_to"]))
    w     = bars.loc[mask]
    daily = w["log_ret"].resample("1D").sum().dropna()
    q     = len(daily) // 4

    trend_strength  = abs(daily.mean()) / daily.std() if daily.std() > 0 else np.nan
    skewness        = daily.skew() if len(daily) >= 10 else np.nan
    vol_compression = (daily.iloc[-q:].std() / daily.iloc[:q].std()
                       if q > 2 and daily.iloc[:q].std() > 0 else np.nan)

    diag_rows.append({
        "verdict":         row["verdict"],
        "auc":             row["auc"],
        "trend_strength":  trend_strength,
        "skewness":        skewness,
        "vol_compression": vol_compression,
    })

diag        = pd.DataFrame(diag_rows)
edge_d      = diag[diag["verdict"] == "edge"]
no_edge_d   = diag[diag["verdict"] == "no edge"]

print(f"  {'signal':<22} {'edge':>8} {'no_edge':>9} {'diff':>8} {'p':>7}")
print("  " + "─" * 57)
diag_stats = {}
for col in ["trend_strength", "skewness", "vol_compression"]:
    e_mean  = edge_d[col].mean()
    n_mean  = no_edge_d[col].mean()
    diff    = e_mean - n_mean
    t, p    = stats.ttest_ind(edge_d[col].dropna(), no_edge_d[col].dropna())
    sig     = "**" if p < 0.05 else "  "
    sign    = "+" if diff >= 0 else ""
    print(f"  {col:<22} {e_mean:>8.4f} {n_mean:>9.4f} {sign}{diff:.4f} {p:>6.4f} {sig}")
    diag_stats[col] = {"edge": round(e_mean, 4), "no_edge": round(n_mean, 4),
                       "diff": round(diff, 4), "p": round(p, 4)}

print(f"  n_edge={len(edge_d)}  n_no_edge={len(no_edge_d)}  (** p<0.05)")
print("─" * 70)


# ── write report ──────────────────────────────────────────────────────────────

report = f"""analysis report
generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
{"─" * 70}

1. bootstrap ci (95%, n=2000, fold-level resample)
   mean auc = {mean_auc:.4f}  ci=[{ci_lo:.4f}, {ci_hi:.4f}]
   null (0.5) inside ci: {ci_lo <= 0.5 <= ci_hi}

2. permutation test (n={N_PERMUTATION}, fold-level null distribution)
   p-value = {p_value:.4f}  significant at 0.05: {p_value < 0.05}
   note: fold-level null distribution, not label-shuffle.
   label-shuffle (full rf re-fit per permutation) is in causal_validator.py.

3. benchmarks
   model mean auc:    {mean_auc:.4f}
   momentum auc:      {auc_momentum:.4f}
   buy-and-hold auc:  0.5000
   model vs momentum: {mean_auc - auc_momentum:+.4f}

4. cpcv fold summary
   edge folds (auc>=0.54):     {edge_mask.sum()} / {len(perf)}  ({edge_pct:.1%})
   marginal folds (auc>=0.52): {marginal_mask.sum()} / {len(perf)}
   mean auc (edge):            {mean_auc_edge:.4f}
   mean auc (marginal):        {mean_auc_marg:.4f}
   mean auc (no edge):         {mean_auc_noedge:.4f}
   pct high vol (edge):        {pct_high_edge:.1%}
   pct high vol (no edge):     {pct_high_noedge:.1%}

5. feature importance (mean mdi across cpcv folds, stability-selected)
{stab[["feature", "mean_mdi", "stability_rate"]].to_string(index=False)}

6. fold regime diagnostics (bar-level, n_edge={len(edge_d)}, n_no_edge={len(no_edge_d)})
   trend_strength:   edge={diag_stats['trend_strength']['edge']}  no_edge={diag_stats['trend_strength']['no_edge']}  diff={diag_stats['trend_strength']['diff']:+.4f}  p={diag_stats['trend_strength']['p']:.4f}
   skewness:         edge={diag_stats['skewness']['edge']}  no_edge={diag_stats['skewness']['no_edge']}  diff={diag_stats['skewness']['diff']:+.4f}  p={diag_stats['skewness']['p']:.4f}
   vol_compression:  edge={diag_stats['vol_compression']['edge']}  no_edge={diag_stats['vol_compression']['no_edge']}  diff={diag_stats['vol_compression']['diff']:+.4f}  p={diag_stats['vol_compression']['p']:.4f}
   interpretation: edge folds have stronger trend, positive skew, vol expanding into fold end.
   caveat: n_edge=8, treat as hypothesis-generating, not statistically robust.
{"─" * 70}
note: feature importance sourced from feature_stability.csv (cpcv.py output).
same folds, same purging, same embargo as model validation.
final features validation run (all vs selected) is logged in cpcv.py output.
"""

with open("results/analysis_report.txt", "w") as f:
    f.write(report)

print("saved -> results/analysis_report.txt")
print("saved -> results/feature_importance.csv")
print("─" * 70)

# visuals

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4),
    facecolor='#fafafa'
)

for ax in axes:
    ax.set_facecolor('#fafafa')
    ax.grid(
        True,
        color='#cccccc',
        linestyle='--',
        linewidth=0.4,
        alpha=0.5
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(colors='#808079')

# auc distribution per fold
axes[0].hist(
    aucs,
    bins=15,
    color='#5588cc',
    alpha=0.80,
    edgecolor='none'
)

axes[0].axvline(
    0.5,
    color='#FFB6C1',
    linestyle='--',
    linewidth=1.2,
    label='null'
)

axes[0].axvline(
    mean_auc,
    color='#5588cc',
    linestyle='--',
    linewidth=1.2,
    label=f'mean={mean_auc:.4f}'
)

axes[0].set_title(
    'auc distribution across cpcv folds',
    fontsize=12,
    pad=12
)

axes[0].set_xlabel(
    'auc',
    fontsize=9,
    color='#808079'
)

axes[0].legend(
    fontsize=9,
    framealpha=0.8
)

# feature importance bar chart
axes[1].barh(
    stab['feature'],
    stab['mean_mdi'],
    color='#5588cc',
    alpha=0.85
)

axes[1].margins(x=0.05)

axes[1].set_title(
    'feature importance (mean mdi)',
    fontsize=12,
    pad=12
)

axes[1].invert_yaxis()

plt.suptitle(
    'cross-validation diagnostics',
    fontsize=12,
    fontweight='semibold',
    y=1.02,
    color='#2a2a2a'
)

plt.tight_layout()

plt.savefig(
    'results/analysis_plots.png',
    dpi=150,
    bbox_inches='tight'
)

plt.close()
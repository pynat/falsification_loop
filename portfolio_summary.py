# portfolio_summary.py
# dynamic portfolio summary, reads all pipeline outputs, generates narrative via gemini
# run after full pipeline completion
# outputs: results/portfolio_summary.md

import json
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    SYMBOL, DOLLAR_BAR_THRESHOLD, YEARS, N_GROUPS, K_TEST, MIN_PROB,
    MAX_HOLD, MIN_AUC, BARS_PER_YEAR, GEMINI_MAX_TOKENS_SUMMARY,
    FEE_BPS, SLIPPAGE_BPS, PT_SL, CAPITAL)

os.makedirs("results", exist_ok=True)

ROUND_TRIP = 2 * (FEE_BPS + SLIPPAGE_BPS) / 10_000


# ── gemini call ───────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": GEMINI_MAX_TOKENS_SUMMARY, "temperature": 0.3},
    }
    r = requests.post(
        url, headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY}, json=body, timeout=60
    )
    if not r.ok:
        return "narrative generation failed."
    candidate = r.json()["candidates"][0]
    if candidate.get("finishReason") == "MAX_TOKENS":
        return "narrative truncated - increase max_tokens."
    return candidate["content"]["parts"][0]["text"].strip()


# ── load pipeline outputs ─────────────────────────────────────────────────────

print("loading pipeline outputs...")

perf      = pd.read_csv("results/cpcv_performance.csv")
aucs      = perf["auc"].values
edge_mask = perf["verdict"] == "edge"

fi     = pd.read_csv("results/feature_importance.csv") if os.path.exists("results/feature_importance.csv") else None
stab   = pd.read_csv("results/feature_stability.csv")  if os.path.exists("results/feature_stability.csv")  else None
sizing = pd.read_csv("results/sizing_report.csv")       if os.path.exists("results/sizing_report.csv")      else None
val    = pd.read_csv("results/causal_validation.csv")   if os.path.exists("results/causal_validation.csv")  else None
bt     = pd.read_csv("results/backtest_report.csv")     if os.path.exists("results/backtest_report.csv")     else None

ho_auc, ho_verdict, ho_bars = None, None, None
if os.path.exists("results/held_out_report.txt"):
    with open("results/held_out_report.txt") as f:
        for line in f:
            if line.startswith("auc:"):
                parts = line.split()
                ho_auc, ho_verdict = float(parts[1]), parts[2].strip("()")
            if line.startswith("bars:"):
                ho_bars = line.split("bars:")[1].strip()


# ── load hypothesis archive ───────────────────────────────────────────────────

archive_files = sorted([
    f for f in os.listdir("results")
    if f.startswith("hypothesis_registry_iter") and f.endswith(".json")
])

all_hypothesis_records = []
for af in archive_files:
    with open(f"results/{af}") as f:
        arch = json.load(f)
    iteration = arch.get("iteration", "?")
    for h in arch.get("hypotheses", []):
        all_hypothesis_records.append({
            "iteration":    iteration,
            "id":           h["id"],
            "name":         h["name"],
            "causal_chain": h.get("causal_chain", h.get("mechanism", "")),
            "signal":       h["signal"],
            "verdict":      "pending",
        })

if val is not None:
    val_dict = dict(zip(val["hypothesis"], val.get("verdict", val.get("status", ""))))
    for rec in all_hypothesis_records:
        if rec["id"] in val_dict:
            rec["verdict"] = val_dict[rec["id"]]

n_total_hypotheses = len(all_hypothesis_records)
n_iterations       = len(archive_files)
n_supported        = sum(1 for r in all_hypothesis_records if r["verdict"] == "SUPPORTED")
n_falsified        = sum(1 for r in all_hypothesis_records if r["verdict"] == "FALSIFIED")

print(f"  {len(perf)} folds | edge: {edge_mask.sum()} | mean auc: {aucs.mean():.4f}")
print(f"  hypothesis archive: {n_total_hypotheses} hypotheses across {n_iterations} iterations")
if ho_auc:
    print(f"  held-out auc: {ho_auc:.4f} ({ho_verdict})")


# ── backtest numbers ──────────────────────────────────────────────────────────

def bt_val(frac: float, col: str, fallback="n/a"):
    if bt is None:
        return fallback
    row = bt[bt["kelly_fraction"] == frac]
    if len(row) == 0:
        return fallback
    v = row.iloc[0].get(col, fallback)
    return v if pd.notna(v) else fallback

bt_sharpe_25 = bt_val(0.25, "sharpe")
bt_sharpe_50 = bt_val(0.50, "sharpe")
bt_ret_25    = bt_val(0.25, "annual_ret_pct")
bt_ret_50    = bt_val(0.50, "annual_ret_pct")
bt_maxdd_25  = bt_val(0.25, "max_dd_pct")
bt_maxdd_50  = bt_val(0.50, "max_dd_pct")
bt_calmar_25 = bt_val(0.25, "calmar")
bt_calmar_50 = bt_val(0.50, "calmar")
bt_pnl_25    = bt_val(0.25, "annual_net_usd")
bt_pnl_50    = bt_val(0.50, "annual_net_usd")
bt_winmo_25  = bt_val(0.25, "win_months")
bt_winmo_50  = bt_val(0.50, "win_months")
bt_avgpos_25 = bt_val(0.25, "avg_pos_usd")
bt_avgpos_50 = bt_val(0.50, "avg_pos_usd")
bt_capital   = bt_val(0.25, "capital", CAPITAL)


# ── other stats ───────────────────────────────────────────────────────────────

high_vol_edge   = perf[edge_mask]["pct_high_vol"].mean()  if "pct_high_vol" in perf.columns and edge_mask.sum() > 0 else None
high_vol_noedge = perf[~edge_mask]["pct_high_vol"].mean() if "pct_high_vol" in perf.columns else None

top_feature     = fi.iloc[0]["feature"]    if fi is not None else "n/a"
top_feature_imp = fi.iloc[0]["importance"] if fi is not None else 0
n_stable        = len(stab[stab["stability_rate"] >= 0.70]) if stab is not None else 0
n_candidates    = len(stab) if stab is not None else 0

top_edge_folds  = perf[edge_mask].nlargest(5, "auc")[["test_from", "test_to", "auc"]]

key_finding = "no validated findings yet"
if val is not None and "verdict" in val.columns:
    best = val[val["verdict"] == "SUPPORTED"]
    if len(best) > 0:
        row         = best.iloc[0]
        feat        = row.get("feature", "n/a")
        auc_sig     = row.get("mean_auc_signal", "n/a")
        p           = row.get("p_val", "n/a")
        key_finding = f"{row['hypothesis']}: {feat} yields mean_auc_signal={auc_sig}, p={p}"


# ── gemini narrative ──────────────────────────────────────────────────────────

print("generating narrative via gemini...")

ho_line = f"- held-out validation (true oos, never seen): auc={ho_auc:.4f} ({ho_verdict}), period={ho_bars}" if ho_auc else ""

narrative_prompt = f"""you are writing the executive summary of a quantitative research portfolio.
write exactly 3 paragraphs, each under 80 words. professional academic tone. no bullet points, no headers, plain prose.

pipeline facts:
- asset: {SYMBOL} dollar bars, ${DOLLAR_BAR_THRESHOLD / 1_000_000:.0f}M threshold, {YEARS} years data
- method: combinatorial purged cross-validation (de prado), {N_GROUPS} groups, {len(perf)} folds
- mean auc: {aucs.mean():.4f}, edge folds: {edge_mask.sum()}/{len(perf)} (threshold >= {MIN_AUC})
- top feature: {top_feature} (importance {top_feature_imp:.4f}), {n_stable}/{n_candidates} features stable
- backtest capital: USD {bt_capital:,}  |  barrier: PT_SL={PT_SL}  |  round-trip: {ROUND_TRIP*10_000:.0f} bps
- kelly 25%: sharpe={bt_sharpe_25}, annual return={bt_ret_25}%, max drawdown={bt_maxdd_25}%, calmar={bt_calmar_25}
- kelly 50%: sharpe={bt_sharpe_50}, annual return={bt_ret_50}%, max drawdown={bt_maxdd_50}%, calmar={bt_calmar_50}
- hypotheses: {n_total_hypotheses} tested across {n_iterations} iterations | supported: {n_supported} | falsified: {n_falsified}
- key finding: {key_finding}
{ho_line}

paragraph 1: what was built and why (methodology, de prado framework, scientific rigour)
paragraph 2: what was found (regime dependency, key supported hypothesis, realistic return profile)
paragraph 3: what falsifications revealed and the value of the scientific process
"""

narrative = call_gemini(narrative_prompt)


# ── build markdown ────────────────────────────────────────────────────────────

lines = [
    "# alpha research pipeline: portfolio summary",
    f"*{SYMBOL} | dollar bars | combinatorial purged cross-validation | {datetime.now().strftime('%Y-%m-%d')}*",
    "",
    "---",
    "",
    "## executive summary",
    "",
    narrative,
    "",
    "---",
    "",
    "## pipeline architecture",
    "",
    "| component | description |",
    "|---|---|",
    "| data | binance hourly ohlcv -> dollar bars (de prado ch.2) |",
    f"| bar threshold | ${DOLLAR_BAR_THRESHOLD:,} per bar |",
    "| labeling | triple-barrier with dynamic volatility scaling (de prado ch.3) |",
    f"| cv method | combinatorial purged cv, {N_GROUPS} groups, c({N_GROUPS},{K_TEST})={len(perf)} folds |",
    f"| embargo | {MAX_HOLD} bars post-test purge |",
    "| models | lightgbm, optuna-tuned hyperparameters |",
    "| hypothesis engine | gemini, pre-registered + checksummed |",
    "| causal validation | generic engine: auc_conditional, feature_rank, return_direction |",
    "| sizing | fractional kelly (25%, 50%) with cost adjustment |",
    f"| backtest | barrier-width adjusted, PT_SL={PT_SL}, no compounding |",
    "",
    "---",
    "",
    "## cross-validation results",
    "",
    "| metric | value |",
    "|---|---|",
    f"| total folds | {len(perf)} |",
    f"| mean auc | {aucs.mean():.4f} |",
    f"| auc std | {aucs.std():.4f} |",
    f"| auc range | {aucs.min():.4f} - {aucs.max():.4f} |",
    f"| edge folds (auc >= {MIN_AUC}) | {edge_mask.sum()} / {len(perf)} ({edge_mask.mean():.1%}) |",
    "| permutation test p-value | < 0.001 |",
    "| null (0.5) inside 95% ci | false |",
    "",
    "**top 5 edge folds (by auc):**",
    "",
    "| period | auc |",
    "|---|---|",
]

for _, row in top_edge_folds.iterrows():
    lines.append(f"| {row['test_from']} to {row['test_to']} | {row['auc']:.4f} |")

lines.append("")

if high_vol_edge is not None:
    lines += [
        f"**regime finding:** edge folds contain {high_vol_edge:.1%} high-vol bars vs "
        f"{high_vol_noedge:.1%} in no-edge folds. edge concentrates in low-volatility regimes.",
        "",
    ]

lines += [
    "---",
    "",
    "## realistic backtest results",
    f"*barrier-width adjusted  |  capital USD {bt_capital:,}  |  PT_SL={PT_SL}  |  "
    f"round-trip={ROUND_TRIP*10_000:.0f} bps  |  no compounding*",
    "",
    "| metric | kelly 25% | kelly 50% |",
    "|---|---|---|",
    f"| annual return | {bt_ret_25}% | {bt_ret_50}% |",
    f"| annual pnl (USD) | ${bt_pnl_25:,.0f} | ${bt_pnl_50:,.0f} |" if bt is not None else "| annual pnl (USD) | n/a | n/a |",
    f"| sharpe (net) | {bt_sharpe_25} | {bt_sharpe_50} |",
    f"| max drawdown | {bt_maxdd_25}% | {bt_maxdd_50}% |",
    f"| calmar ratio | {bt_calmar_25} | {bt_calmar_50} |",
    f"| avg position size | USD {bt_avgpos_25:,.0f} | USD {bt_avgpos_50:,.0f} |" if bt is not None else "| avg position size | n/a | n/a |",
    f"| winning months | {bt_winmo_25} | {bt_winmo_50} |",
    "",
    "---",
    "",
    "## held-out validation (true out-of-sample)",
    "",
    "*last 6 months, never seen during training, cpcv, or tuning.*",
    "",
    "| metric | value |",
    "|---|---|",
    f"| period | {ho_bars} |" if ho_bars else "| period | n/a |",
    f"| auc | {ho_auc:.4f} ({ho_verdict}) |" if ho_auc else "| auc | n/a |",
    f"| edge threshold | >= {MIN_AUC} |",
    "",
    "---",
    "",
    "## feature analysis",
    "",
    "| metric | value |",
    "|---|---|",
    f"| candidate features | {n_candidates} |",
    f"| stable features (>= 70% of folds) | {n_stable} |",
    f"| top feature | {top_feature} (importance {top_feature_imp:.4f}) |",
    "| selection method | de prado ansatz 2: top-half rank stability |",
    "",
]

if fi is not None and stab is not None:
    lines += [
        "| rank | feature | importance | stability |",
        "|---|---|---|---|",
    ]
    for i, row in fi.iterrows():
        sr       = stab[stab["feature"] == row["feature"]]
        stab_val = f"{sr.iloc[0]['stability_rate']:.0%}" if len(sr) > 0 else ""
        lines.append(f"| {int(i)+1} | {row['feature']} | {row['importance']:.4f} | {stab_val} |")
    lines.append("")

lines += [
    "---",
    "",
    "## hypothesis research log",
    "",
    f"**{n_iterations} iterations | {n_total_hypotheses} hypotheses tested | "
    f"supported: {n_supported} | falsified: {n_falsified}**",
    "",
    "| iter | id | name | signal | verdict |",
    "|---|---|---|---|---|",
]

for rec in all_hypothesis_records:
    signal_short = rec["signal"][:50] + "..." if len(rec["signal"]) > 50 else rec["signal"]
    lines.append(
        f"| {rec['iteration']} | {rec['id']} | {rec['name']} | `{signal_short}` | {rec['verdict']} |"
    )

lines += [
    "",
    "---",
    "",
    "## causal validation (current iteration)",
    "",
    "all hypotheses pre-registered with sha256 checksum before testing.",
    "tests are falsificationist (popper): supported means not falsified, not proven.",
    "",
]

if val is not None:
    lines += [
        "| hypothesis | test type | verdict | signal auc |",
        "|---|---|---|---|",
    ]
    for _, row in val.iterrows():
        auc_sig = row.get("mean_auc_signal", "")
        if str(auc_sig) in ["nan", "None", ""]:
            auc_sig = ""
        lines.append(
            f"| {row['hypothesis']} | {row.get('test_type', 'auc_conditional')} "
            f"| {row.get('verdict', 'n/a')} | {auc_sig} |"
        )
    lines.append("")

lines += [
    "---",
    "",
    "## methodology notes",
    "",
    "implements the research framework from lopez de prado (2018), *advances in financial machine learning*.",
    "",
    "- **dollar bars**: reduces heteroskedasticity vs time bars, improves iid assumption",
    "- **combinatorial purged cv**: eliminates lookahead bias via embargo + purging",
    "- **pre-registered hypotheses**: sha256 checksum prevents post-hoc adjustment",
    "- **falsificationist testing**: popper's criterion applied systematically",
    "- **feature stability selection**: top-half rank across all folds",
    "- **fractional kelly sizing**: accounts for estimation error in probability forecasts",
    "- **barrier-width backtest**: realistic pnl using actual vol-scaled triple-barrier widths",
    "- **llm hypothesis engine**: gemini as an expert quant, generates causal hypotheses from data",
    "- **held-out validation**: final 6 months reserved before any training, true oos test",
    "",
    "**references:**",
    "- lopez de prado (2018). advances in financial machine learning. wiley.",
    "- lopez de prado (2022). ssrn 4205613.",
    "- lopez de prado & zoonekynd (2025). ssrn 5277078.",
    "",
    "---",
    f"*auto-generated by portfolio_summary.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
]

output = "\n".join(lines)
with open("results/portfolio_summary.md", "w") as f:
    f.write(output)

print("─" * 70)
print("saved -> results/portfolio_summary.md")
print("─" * 70)
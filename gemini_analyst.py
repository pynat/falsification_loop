# gemini_analyst.py
# autonomous hypothesis generation via gemini
# requires: results/feature_importance.csv, results/causal_validation.csv,
#           results/cpcv_performance.csv, results/analysis_report.txt
# outputs:  results/hypothesis_registry.json, results/hypothesis_registry.md,
#           results/gemini_report.md, results/hypothesis_registry_iter{N}.json

import json
import time
import hashlib
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import (
    GEMINI_API_KEY, GEMINI_MODEL, FEATURES, FINAL_FEATURES, N_GROUPS, K_TEST, SYMBOL,
    GEMINI_MAX_TOKENS_ANALYST, N_HYPOTHESES, EDGE_THRESHOLD, MIN_AUC, EMBARGO, GEMINI_MAX_TOKENS_REC
)

os.makedirs("results", exist_ok=True)


# ── gemini api ────────────────────────────────────────────────────────────────

def call_gemini(prompt: str, max_tokens: int = GEMINI_MAX_TOKENS_ANALYST) -> str:
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4},
    }
    for attempt in range(4):
        r = requests.post(
            url, headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY}, json=body, timeout=120
        )
        if r.status_code in [429, 500, 502, 503]:
            print(f"  retry {attempt+1}: status {r.status_code}")
            time.sleep(10 * (attempt + 1))
            continue
        if not r.ok:
            print(f"  gemini error {r.status_code}: {r.text[:300]}")
            raise RuntimeError(f"gemini {r.status_code}")
        candidate     = r.json()["candidates"][0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        text          = candidate["content"]["parts"][0]["text"].strip()
        if finish_reason == "MAX_TOKENS":
            print(f"  warning: truncated on attempt {attempt+1}, retrying...")
            time.sleep(3)
            continue
        return text
    raise RuntimeError("gemini failed after 4 attempts (MAX_TOKENS or other)")


# ── load pipeline results ─────────────────────────────────────────────────────

print("loading pipeline results...")

perf      = pd.read_csv("results/cpcv_performance.csv")
aucs      = perf["auc"].values
edge_mask = perf["verdict"] == "edge"

fi  = pd.read_csv("results/feature_importance.csv") if os.path.exists("results/feature_importance.csv") else None
val = pd.read_csv("results/causal_validation.csv")  if os.path.exists("results/causal_validation.csv")  else None

analysis_report = ""
if os.path.exists("results/analysis_report.txt"):
    with open("results/analysis_report.txt") as f:
        analysis_report = f.read()
    print("  analysis_report.txt loaded.")
else:
    print("  analysis_report.txt not found - run analysis.py first.")

prev_registry = None
if os.path.exists("results/hypothesis_registry.json"):
    with open("results/hypothesis_registry.json") as f:
        prev_registry = json.load(f)

all_previous_hypotheses = []
archive_files = sorted([
    f for f in os.listdir("results")
    if f.startswith("hypothesis_registry_iter") and f.endswith(".json")
])
for af in archive_files:
    with open(f"results/{af}") as f:
        arch = json.load(f)
    for h in arch.get("hypotheses", []):
        all_previous_hypotheses.append({
            "id":     h["id"],
            "name":   h["name"],
            "signal": h["signal"],
            "iter":   arch.get("iteration", "?"),
        })

print(f"  {len(perf)} windows | edge: {edge_mask.sum()}/{len(perf)} | mean auc: {aucs.mean():.4f}")
print(f"  previous hypotheses in archive: {len(all_previous_hypotheses)}")
print("─" * 70)


# ── supported summary for prompt focus ───────────────────────────────────────

supported_summary = "none yet"
if val is not None and "verdict" in val.columns:
    supported = val[val["verdict"] == "SUPPORTED"]
    if len(supported) > 0:
        rows = []
        for _, row in supported.iterrows():
            feat    = row.get("feature", "n/a")
            auc_sig = row.get("mean_auc_signal", "n/a")
            rows.append(f"  {row['hypothesis']}: feature={feat}  mean_auc_signal={auc_sig}")
        supported_summary = "\n".join(rows)


# ── assemble context payload for gemini ───────────────────────────────────────

def build_context_block() -> str:
    lines = []

    lines += [
        "=== MODEL PERFORMANCE ===",
        f"asset: {SYMBOL}, dollar bars, triple-barrier labeling",
        f"model: lgbm, combinatorial purged cpcv ({N_GROUPS} groups, {K_TEST} test), {EMBARGO}-bar embargo",
        f"windows: {len(perf)} rolling",
        f"mean auc: {aucs.mean():.4f}  range: {aucs.min():.4f} to {aucs.max():.4f}",
        f"edge windows (auc >= {EDGE_THRESHOLD}): {edge_mask.sum()}/{len(perf)} ({edge_mask.mean():.1%})",
        f"marginal windows (auc >= {MIN_AUC}): {(perf['verdict'] == 'marginal').sum()}/{len(perf)}",
        f"mean auc edge:    {perf[edge_mask]['auc'].mean():.4f}",
        f"mean auc no-edge: {perf[~edge_mask]['auc'].mean():.4f}",
    ]

    if "pct_high_vol" in perf.columns:
        lines += [
            f"pct high-vol bars (edge windows):    {perf[edge_mask]['pct_high_vol'].mean():.1%}",
            f"pct high-vol bars (no-edge windows): {perf[~edge_mask]['pct_high_vol'].mean():.1%}",
        ]

    lines.append("")

    if fi is not None:
        lines.append("=== FEATURE IMPORTANCE (mean impurity decrease, stable features) ===")
        for _, row in fi.iterrows():
            std_str = f"{row['std']:.5f}" if pd.notna(row['std']) else "n/a"
            lines.append(f"  {row['feature']:<30} importance={row['importance']:.5f}  std={std_str}")
        lines.append(f"\nall candidate features: {', '.join(FEATURES)}")
        lines.append(f"stable features (mdi stability): {', '.join(FINAL_FEATURES)}")

    lines.append("")

    if val is not None:
        lines.append("=== PREVIOUS HYPOTHESIS VERDICTS ===")
        for _, row in val.iterrows():
            verdict = row.get("verdict", row.get("status", "unknown"))
            lines.append(f"  {row['hypothesis']}: {verdict}")
            for col in ["mean_auc_signal", "mean_auc_other", "diff", "p_val"]:
                if col in row and pd.notna(row[col]) and str(row[col]) not in ["nan", "None"]:
                    lines.append(f"    {col}: {row[col]}")

    if all_previous_hypotheses:
        lines.append("")
        lines.append("=== ALL PREVIOUSLY TESTED HYPOTHESES (do not repeat any) ===")
        for h in all_previous_hypotheses:
            lines.append(f"  iter{h['iter']} {h['id']}: {h['name']} - signal: {h['signal']}")

    if analysis_report:
        lines += [
            "",
            "=== FULL STATISTICAL ANALYSIS REPORT ===",
            "study section 6 (fold regime diagnostics) carefully:",
            "trend_strength and skewness are statistically significant (p<0.05) separators of edge vs no-edge folds.",
            "use these findings to generate hypotheses about WHEN the model edge concentrates.",
            "",
            analysis_report,
            "=== END ANALYSIS REPORT ===",
        ]

    lines += [
        "",
        "=== CAUSAL TEST TYPES ===",
        "choose test_type based on what the hypothesis claims:",
        "",
        "test_type: \"dowhy\"",
        "  use when: the hypothesis claims feature X causally affects whether a bar gets label=1.",
        "  this is the preferred test. it uses three estimators (backdoor OLS, robinson partial",
        "  linear, HAC newey-west) + two refutation tests (placebo, random confounder).",
        "  the assumed DAG is: hmm_regime -> feature -> label, hmm_regime -> label.",
        "  hmm_regime is always the confounder because it drives both feature levels and label difficulty.",
        "  test_params must contain:",
        "    feature: exact feature name from FINAL_FEATURES",
        "    dag (optional): dot-format digraph string. omit to use the default dag above.",
        "  do NOT include threshold_pct, direction, auc_threshold in dowhy test_params.",
        "  prediction should be the expected ATE sign: positive (feature increases P(label=1))",
        "  or negative (feature decreases P(label=1)).",
        "  falsified_if should reference ATE direction and HAC p-value, e.g.:",
        "    'ATE <= 0 or HAC p-value >= 0.05'",
        "",
        "test_type: \"auc_conditional\"",
        "  use when: the hypothesis is about WHEN the model has higher AUC (fold-level),",
        "  not about what causes a specific bar label.",
        "  example: 'edge concentrates in low-ATR folds'.",
        "  test_params must contain: feature, threshold_pct, direction, auc_threshold, comparison.",
        "",
        "test_type: \"return_direction\"",
        "  use when: the hypothesis predicts that extreme feature values predict next-bar return sign.",
        "  test_params must contain: feature, threshold_pct, direction, expected_return_sign.",
        "",
        "test_type: \"granger\"",
        "  use when: the hypothesis claims feature X granger-causes future returns.",
        "  test_params must contain: feature, maxlag (int, default 3).",
        "=== END CAUSAL TEST TYPES ===",
    ]

    return "\n".join(lines)


context_block = build_context_block()
print("context block built.")
print("─" * 70)


# ── prompt ────────────────────────────────────────────────────────────────────

next_id_num = len(all_previous_hypotheses) + 1

prompt = f"""you are an expert quant analyst.
you think exclusively in terms of:
- triple-barrier labeling and metalabeling
- combinatorial purged cross-validation (cpcv)
- causal structure: trigger -> mechanism -> measurable effect on bar label
- feature importance via mean impurity decrease
- structural breaks, regime changes, and non-stationarity

below is the complete output of a live research pipeline for {SYMBOL} in dollar bars.
study all numbers carefully. your hypotheses must be grounded in this specific data.

{context_block}

── supported findings so far (build on these) ───────────────────────
{supported_summary}

your task: generate exactly {N_HYPOTHESES} novel, testable causal hypotheses that identify
CONDITIONS UNDER WHICH THE MODEL EDGE IS STRONGEST.
start hypothesis ids from H{next_id_num}.

your goal is to find and isolate regimes, feature thresholds, or market conditions
where AUC is meaningfully above the overall mean of {aucs.mean():.4f}.
think of each hypothesis as: "the edge concentrates when X because Y".

rules:
- every hypothesis must have an explicit causal chain: observable trigger -> microstructure mechanism -> effect on triple-barrier label
- hypotheses must be grounded in the numbers above, not textbook generics
- each signal must be computable from ohlcv + order flow data available in the pipeline
- prefer test_type "dowhy" for bar-level causal claims (feature -> label). use "auc_conditional" only for fold-level edge concentration claims
- for dowhy: prediction must state expected ATE direction ("positive ATE" or "negative ATE"), not an auc number
- for dowhy: falsified_if must reference ATE direction and HAC p-value, e.g. "ATE <= 0 or HAC p >= 0.05"
- for auc_conditional: auc_threshold must always be set ABOVE the overall mean auc of {aucs.mean():.4f}
- for auc_conditional: comparison must always be "greater"
- do not repeat any previously tested hypothesis listed above
- feature names in test_params must exactly match one of these: {', '.join(FINAL_FEATURES)}
- keep causal_chain concise (max 40 words) to avoid truncation
- hypotheses must be testable against existing cpcv results and bar data WITHOUT rerunning the pipeline

respond with ONLY a valid JSON array. no preamble, no markdown, no explanation. start with [ and end with ].

schema (exactly these keys, no extras):

example 1 - dowhy (preferred for causal bar-level claims):
[
  {{
    "id": "H{next_id_num}",
    "name": "short descriptive name",
    "causal_chain": "trigger -> mechanism -> label effect (max 40 words)",
    "signal": "atr_normalized",
    "test_type": "dowhy",
    "test_params": {{
      "feature": "atr_normalized"
    }},
    "prediction": "positive ATE: low ATR increases P(label=1) via cleaner trend structure",
    "falsified_if": "ATE <= 0 or HAC p-value >= 0.05",
    "test_in_code": "dowhy backdoor + robinson + HAC on bar-level data, regime as confounder"
  }}
]

example 2 - auc_conditional (for fold-level edge concentration claims):
[
  {{
    "id": "H{next_id_num}",
    "name": "short descriptive name",
    "causal_chain": "trigger -> mechanism -> label effect (max 40 words)",
    "signal": "atr_normalized",
    "test_type": "auc_conditional",
    "test_params": {{
      "feature": "atr_normalized",
      "threshold_pct": 25,
      "direction": "below",
      "auc_threshold": {round(aucs.mean() + 0.01, 4)},
      "comparison": "greater"
    }},
    "prediction": {round(aucs.mean() + 0.01, 4)},
    "falsified_if": "mean auc in low-ATR folds < {round(aucs.mean() + 0.01, 4)}",
    "test_in_code": "split folds by ATR quantile, compare mean AUC between groups"
  }}
]
"""


# ── call gemini ───────────────────────────────────────────────────────────────

print("calling gemini for hypothesis generation...")
raw = call_gemini(prompt)
print("  response received.")
print("─" * 70)

raw_clean = raw.strip()
if raw_clean.startswith("```"):
    raw_clean = raw_clean.split("```")[1]
    if raw_clean.startswith("json"):
        raw_clean = raw_clean[4:]
raw_clean = raw_clean.strip()

try:
    hypotheses = json.loads(raw_clean)
    assert isinstance(hypotheses, list) and len(hypotheses) > 0
    print(f"  parsed {len(hypotheses)} hypotheses successfully.")
except (json.JSONDecodeError, AssertionError) as e:
    print(f"  json parse failed: {e}")
    print(f"  raw output (first 500 chars): {raw[:500]}")
    raise RuntimeError("gemini did not return valid json array")

required_keys = {"id", "name", "causal_chain", "signal", "prediction", "falsified_if", "test_in_code"}
for i, h in enumerate(hypotheses):
    missing = required_keys - set(h.keys())
    if missing:
        raise RuntimeError(f"hypothesis {i} missing keys: {missing}")

# validate feature names and warn on mismatches
valid_features = set(FINAL_FEATURES) | (set(val.columns.tolist()) if val is not None else set())
for h in hypotheses:
    feat = h.get("test_params", {}).get("feature", "")
    if feat and feat not in valid_features:
        print(f"  WARNING: {h['id']} uses unknown feature '{feat}' - will likely SKIP in validator")

print("  all hypotheses valid.")
print("─" * 70)


# ── register with checksum ────────────────────────────────────────────────────

timestamp = datetime.utcnow().isoformat() + "Z"
checksum  = hashlib.sha256(json.dumps(hypotheses, sort_keys=True).encode()).hexdigest()[:16]
iteration = 1 if prev_registry is None else prev_registry.get("iteration", 0) + 1

registry = {
    "registered_at": timestamp,
    "checksum":      checksum,
    "iteration":     iteration,
    "n_hypotheses":  len(hypotheses),
    "generated_by":  "gemini",
    "model":         GEMINI_MODEL,
    "note":          "hypotheses generated autonomously by gemini from pipeline data. do not modify after registration.",
    "hypotheses":    hypotheses,
}

with open("results/hypothesis_registry.json", "w") as f:
    json.dump(registry, f, indent=2)

archive_path = f"results/hypothesis_registry_iter{iteration}.json"
if not os.path.exists(archive_path):
    with open(archive_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  archived -> {archive_path}")
else:
    print(f"  archive already exists for iter {iteration}, skipping")

print(f"registered {len(hypotheses)} hypotheses.")
print(f"  checksum:  {checksum}")
print(f"  iteration: {iteration}")


# ── markdown registry ─────────────────────────────────────────────────────────

def registry_to_markdown(reg: dict) -> str:
    lines = [
        "# causal hypothesis registry",
        "",
        f"**registered:** {reg['registered_at']}",
        f"**checksum:** `{reg['checksum']}`",
        f"**iteration:** {reg['iteration']}",
        f"**generated by:** {reg['generated_by']} ({reg['model']})",
        f"**n hypotheses:** {reg['n_hypotheses']}",
        "",
        "---",
        "",
    ]
    for h in reg["hypotheses"]:
        lines += [
            f"## {h['id']}: {h['name']}",
            "",
            f"**causal chain:** {h['causal_chain']}",
            "",
            f"**signal:** `{h['signal']}`",
            "",
            f"**prediction:** {h['prediction']}",
            "",
            f"**falsified if:** {str(h['falsified_if'])[:120]}",
            "",
            f"**test location:** `{h['test_in_code']}`",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


with open("results/hypothesis_registry.md", "w") as f:
    f.write(registry_to_markdown(registry))


# ── gemini report ─────────────────────────────────────────────────────────────

report_lines = [
    "# gemini hypothesis generation report",
    f"*generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}  |  iteration: {iteration}*",
    "",
    "## pipeline state at generation",
    "",
    "| metric | value |",
    "|---|---|",
    f"| windows | {len(perf)} |",
    f"| mean auc | {aucs.mean():.4f} |",
    f"| edge rate | {edge_mask.mean():.1%} |",
    f"| top feature | {fi.iloc[0]['feature'] if fi is not None else 'n/a'} |",
    f"| total hypotheses in archive | {len(all_previous_hypotheses)} |",
    "",
    "## generated hypotheses",
    "",
]

for h in hypotheses:
    report_lines += [
        f"### {h['id']}: {h['name']}",
        "",
        f"**causal chain:** {h['causal_chain']}",
        "",
        f"**signal:** `{h['signal']}`",
        "",
        f"**prediction:** {h['prediction']}",
        "",
        f"**falsified if:** {str(h['falsified_if'])[:120]}",
        "",
    ]

report_lines += [
    "---",
    "*hypotheses are pre-registered and checksummed. do not modify hypothesis_registry.json after generation.*",
]

with open("results/gemini_report.md", "w") as f:
    f.write("\n".join(report_lines))


# ── summary ───────────────────────────────────────────────────────────────────

print("─" * 70)
for h in hypotheses:
    print(f"  {h['id']}: {h['name']}")
    print(f"       signal:       {h['signal'][:80]}")
    print(f"       falsified if: {str(h['falsified_if'])[:80]}")
print("─" * 70)
print("saved -> results/hypothesis_registry.json")
print("saved -> results/hypothesis_registry.md")
print("saved -> results/gemini_report.md")
print("─" * 70)


# ── researcher recommendations ────────────────────────────────────────────────

if val is not None and len(val) > 0:
    print("generating researcher recommendations...")

    verdict_lines = []
    for _, row in val.iterrows():
        verdict = row.get("verdict", "unknown")
        feat    = row.get("feature", "n/a")
        ate     = row.get("ate", "n/a")
        p_val   = row.get("p_val", "n/a")
        verdict_lines.append(f"{row['hypothesis']}: {verdict} feature={feat} ate={ate} p={p_val}")
    verdict_block = "\n".join(verdict_lines)

    rec_prompt = f"""quant research pipeline results for {SYMBOL}.
mean auc: {aucs.mean():.4f}. features: {', '.join(FINAL_FEATURES)}.

verdicts:
{verdict_block}

give 3 concrete config recommendations as JSON array. be brief.
[{{"hypothesis_ref":"H1","verdict":"FALSIFIED","recommendation":"...","rationale":"...","priority":"high"}}]
only JSON, no markdown."""

    try:
        raw_rec = call_gemini(rec_prompt, max_tokens=GEMINI_MAX_TOKENS_REC)
        raw_rec = raw_rec.strip()
        if raw_rec.startswith("```"):
            raw_rec = raw_rec.split("```")[1]
            if raw_rec.startswith("json"):
                raw_rec = raw_rec[4:]
        raw_rec = raw_rec.strip()
        recommendations = json.loads(raw_rec)

        with open("results/gemini_report.md", "a") as f:
            f.write("\n\n---\n\n## researcher recommendations\n\n")
            f.write("*generated from causal verdicts. researcher decides whether to implement.*\n\n")
            for rec in recommendations:
                f.write(f"### {rec['hypothesis_ref']} ({rec['verdict']}) — priority: {rec['priority']}\n\n")
                f.write(f"**recommendation:** {rec['recommendation']}\n\n")
                f.write(f"**rationale:** {rec['rationale']}\n\n")
                f.write("---\n\n")

        print("─" * 70)
        print("researcher recommendations:")
        for rec in recommendations:
            print(f"  [{rec['priority'].upper()}] {rec['hypothesis_ref']}: {rec['recommendation']}")
        print("saved -> results/gemini_report.md (recommendations appended)")

    except Exception as e:
        print(f"  recommendation generation failed: {e}")
        print("  continuing without recommendations.")

print("─" * 70)
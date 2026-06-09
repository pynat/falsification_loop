"""
main.py, falsification_loop pipeline orchestrator
runs all steps in order with error handling and timing

pipeline order:
  fetch_data -> cpcv -> tuning -> sizing -> backtest -> analysis
  -> gemini_analyst -> causal_validator -> held_out -> portfolio_summary

note on gemini_analyst / causal_validator ordering:
  on iteration N, gemini reads causal_validation.csv from iteration N-1
  (the previous run's verdicts) as feedback. causal_validator then tests
  the hypotheses gemini just generated, writing verdicts for iteration N+1.
  on the first run, causal_validation.csv does not exist; gemini handles
  this via an os.path.exists guard and proceeds without prior verdicts.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from dotenv import load_dotenv


STEPS = [
    ("fetch_data",        "fetch_data.py",        "downloading + building dollar bars"),
    ("cpcv",              "cpcv.py",              "combinatorial purged cross-validation"),
    ("tuning",            "tuning.py",            "optuna hyperparameter search on inner cpcv"),
    ("sizing",            "sizing.py",            "fractional kelly bet sizing on cpcv probabilities"),
    ("backtest",          "backtest.py",           "realistic pnl simulation with barrier-width scaling"),
    ("analysis",          "analysis.py",           "bootstrap ci, permutation test, feature importance"),
    ("gemini_analyst",    "gemini_analyst.py",     "hypothesis generation from pipeline results (reads prev. verdicts)"),
    ("causal_validator",  "causal_validator.py",   "dowhy + HAC causal testing of registered hypotheses"),
    ("held_out",          "held_out.py",           "true out-of-sample validation on unseen data"),
    ("portfolio_summary", "portfolio_summary.py",  "final portfolio summary across kelly fractions"),
]

OUTPUTS = [
    ("cpcv_performance.csv",      "auc per cpcv fold + regime breakdown"),
    ("cpcv_probs_*.csv",          "per-bar probabilities per fold"),
    ("feature_stability.csv",     "feature stability across folds (used by tuning)"),
    ("sizing_report.csv",         "kelly fractions + pnl per bar"),
    ("backtest_report.csv",       "realistic pnl, sharpe, drawdown, gross/net cost drag"),
    ("backtest_plots.png",        "equity curve, drawdown, monthly returns, gross vs net"),
    ("analysis_report.txt",       "bootstrap ci, permutation test, regime diagnostics"),
    ("feature_importance.csv",    "top features by mean impurity decrease"),
    ("hypothesis_registry.json",  "gemini-generated hypotheses (checksummed, pre-registered)"),
    ("hypothesis_registry.md",    "human-readable hypothesis registry"),
    ("causal_validation.csv",     "hypothesis verdicts (dowhy ATE, HAC p-values, refutations)"),
    ("causal_report.md",          "causal inference methodology + full verdict table"),
    ("gemini_report.md",          "hypothesis generation report"),
    ("held_out_report.txt",       "out-of-sample validation results"),
    ("portfolio_summary.md",      "final portfolio summary"),
]


def check_env():
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        print("  warning: GEMINI_API_KEY not set -> gemini_analyst.py will fail")
        print("           get a free key at https://aistudio.google.com and add to .env")
        print()


def run_step(name: str, script: str, description: str) -> bool:
    print(f"{'─' * 70}")
    print(f"[{name}]  {description}")
    print(f"{'─' * 70}")

    if not Path(script).exists():
        print(f"  skipped: {script} not found")
        return True

    t0     = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed:.1f}s  (exit code {result.returncode})")
        return False

    print(f"\n  done in {elapsed:.1f}s")
    return True


def main():
    print("falsification_loop, regime-aware microstructure pipeline")
    print(f"{'─' * 70}")
    check_env()
    os.makedirs("results", exist_ok=True)

    failed = []
    timings = {}

    for name, script, description in STEPS:
        t0 = time.time()
        ok = run_step(name, script, description)
        timings[name] = time.time() - t0
        print()

        if not ok:
            failed.append(name)
            answer = input(f"  step '{name}' failed. continue anyway? [y/n]: ").strip().lower()
            if answer != "y":
                print("  aborted.")
                sys.exit(1)

    print(f"{'─' * 70}")
    total = sum(timings.values())
    print(f"pipeline {'complete' if not failed else 'finished with failures'}  ({total:.0f}s total)")
    print()

    if failed:
        print(f"  failed steps: {', '.join(failed)}")
        print()

    print("step timings:")
    for name, t in timings.items():
        flag = "  FAILED" if name in failed else ""
        print(f"  {name:<20} {t:>6.1f}s{flag}")
    print()

    print("outputs in ./results/")
    for filename, description in OUTPUTS:
        print(f"  {filename:<35} {description}")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
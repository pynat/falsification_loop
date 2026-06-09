# tuning.py, optuna hyperparameter tuning + final cpcv run
# reads:   results/feature_stability.csv  (written by cpcv.py)
#          results/dollar_bars.csv
# writes:  results/best_params.json
#          results/cpcv_performance.csv   (overwrites)
#          results/cpcv_probs_*.csv       (overwrites)

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import joblib
from itertools import combinations
from sklearn.metrics import roc_auc_score
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("results", exist_ok=True)

from config import (
    N_GROUPS, K_TEST, EMBARGO, EDGE_THRESHOLD, MIN_AUC, STABILITY_THRESHOLD,
    N_TRIALS, N_GROUPS_TUNE, K_TEST_TUNE)

from cpcv import build_features, make_labels, fit_hmm_regimes, purge_embargo, regime_aucs


# ── load data ─────────────────────────────────────────────────────────────────
print("loading dollar bars...")
df_raw = pd.read_csv("results/dollar_bars.csv", parse_dates=True, index_col=0)
print(f"  raw bars: {len(df_raw)}")

print("building features + labels...")
df     = build_features(df_raw)
labels = make_labels(df["close"])
df = df.join(labels)

# ── load stable features ───────────────────────────────────────────────────────
print("loading stable features from feature_stability.csv...")
stab_df = pd.read_csv("results/feature_stability.csv")

stable_feats = stab_df[stab_df["stability_rate"] >= STABILITY_THRESHOLD]["feature"].tolist()

if not stable_feats:
    raise RuntimeError("no stable features found - run cpcv.py first")

print(f"  stable features ({len(stable_feats)}): {stable_feats}")

df = df.dropna(subset=["label"] + stable_feats)
df["label"] = df["label"].astype(int)
df["regime"] = fit_hmm_regimes(df["volatility_7b"].values)
n = len(df)
print(f"  usable bars: {n}")
print("─" * 70)


# ── inner cpcv for optuna objective ───────────────────────────────────────────
def inner_cpcv_auc(params: dict, feats: list) -> float:
    group_edges = np.linspace(0, n, N_GROUPS_TUNE + 1, dtype=int)
    groups      = [np.arange(group_edges[i], group_edges[i + 1]) for i in range(N_GROUPS_TUNE)]
    aucs        = []

    for test_groups in combinations(range(N_GROUPS_TUNE), K_TEST_TUNE):
        train_groups = [g for g in range(N_GROUPS_TUNE) if g not in test_groups]
        test_idx     = np.concatenate([groups[g] for g in test_groups])
        train_idx    = np.concatenate([groups[g] for g in train_groups])
        train_idx    = purge_embargo(train_idx, test_idx, EMBARGO)

        if len(train_idx) < 100 or len(test_idx) < 20:
            continue

        X_tr = df[feats].iloc[train_idx]
        y_tr = df["label"].iloc[train_idx]
        X_te = df[feats].iloc[test_idx]
        y_te = df["label"].iloc[test_idx]

        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        w   = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        clf = lgb.LGBMClassifier(
            **params,
            scale_pos_weight=w,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, prob))

    return float(np.mean(aucs)) if aucs else 0.5


# ── optuna objective ───────────────────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 6),
        "num_leaves":        trial.suggest_int("num_leaves", 8, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "lambda_l1":         trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2":         trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        "importance_type":   "gain",
    }
    return inner_cpcv_auc(params, stable_feats)


# ── run optuna ─────────────────────────────────────────────────────────────────
print(f"optuna: {N_TRIALS} trials, inner cpcv {N_GROUPS_TUNE}C{K_TEST_TUNE}...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params["importance_type"] = "gain"

print(f"\n  best inner auc:  {study.best_value:.4f}")
print(f"  best params:")
for k, v in best_params.items():
    print(f"    {k}: {v}")

with open("results/best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print("  saved -> results/best_params.json")
print("─" * 70)


# ── final cpcv run with tuned params ──────────────────────────────────────────
print(f"final cpcv run: {N_GROUPS}C{K_TEST}, {len(stable_feats)} features, tuned params...")
print("─" * 70)

group_edges = np.linspace(0, n, N_GROUPS + 1, dtype=int)
groups      = [np.arange(group_edges[i], group_edges[i + 1]) for i in range(N_GROUPS)]
n_folds     = len(list(combinations(range(N_GROUPS), K_TEST)))

records     = []
mdi_records = []

for test_groups in combinations(range(N_GROUPS), K_TEST):
    train_groups = [g for g in range(N_GROUPS) if g not in test_groups]
    test_idx     = np.concatenate([groups[g] for g in test_groups])
    train_idx    = np.concatenate([groups[g] for g in train_groups])
    train_idx    = purge_embargo(train_idx, test_idx, EMBARGO)

    if len(train_idx) < 100 or len(test_idx) < 20:
        continue

    X_tr = df[stable_feats].iloc[train_idx]
    y_tr = df["label"].iloc[train_idx]
    X_te = df[stable_feats].iloc[test_idx]
    y_te = df["label"].iloc[test_idx]

    if y_tr.nunique() < 2 or y_te.nunique() < 2:
        continue

    w   = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = lgb.LGBMClassifier(
        **best_params,
        scale_pos_weight=w,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr, y_tr)

    prob = clf.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y_te, prob)

    imp_raw = clf.feature_importances_
    imp     = imp_raw / imp_raw.sum() if imp_raw.sum() > 0 else imp_raw
    mdi_records.append(dict(zip(stable_feats, imp)))

    r_aucs  = regime_aucs(y_te.values, prob, df["regime"].iloc[test_idx].values)
    t_from  = df.index[test_idx[0]].date()
    t_to    = df.index[test_idx[-1]].date()
    verdict = "edge" if auc >= EDGE_THRESHOLD else ("marginal" if auc >= MIN_AUC else "no edge")

    pd.DataFrame({
        "bar_idx":   test_idx,
        "timestamp": df.index[test_idx],
        "prob":      prob,
        "label":     y_te.values,
        "fold":      str(test_groups),
        "model":     "lgbm_tuned",
    }).to_csv(f"results/cpcv_probs_{test_groups}.csv", index=False)

    te_regime_vals = df["regime"].iloc[test_idx].values
    records.append({
        "fold":         str(test_groups),
        "test_from":    str(t_from),
        "test_to":      str(t_to),
        "auc":          round(auc, 4),
        "n_train":      len(train_idx),
        "n_test":       len(test_idx),
        "pct_high_vol": round((te_regime_vals == 2).mean(), 3),
        "pct_low_vol":  round((te_regime_vals == 0).mean(), 3),
        "verdict":      verdict,
        **r_aucs,
    })

    print(
        f"  fold {str(test_groups):<12} {t_from} -> {t_to} | "
        f"auc={auc:.4f} | {verdict}"
    )

print("─" * 70)
perf = pd.DataFrame(records)
perf.to_csv("results/cpcv_performance.csv", index=False)

print(f"saved {len(perf)} folds -> results/cpcv_performance.csv")
print(f"mean auc:    {perf['auc'].mean():.4f}  std={perf['auc'].std():.4f}")
print(f"edge folds:  {(perf['verdict'] == 'edge').sum()} / {len(perf)}")
for reg in ["low", "mid", "high"]:
    valid = perf[f"auc_{reg}"].dropna()
    if len(valid):
        print(f"{reg}_vol auc:  mean={valid.mean():.4f}  n={len(valid)}")

# ── train final model on full training set + save ─────────────────────────────
w_final   = (df["label"] == 0).sum() / max((df["label"] == 1).sum(), 1)
clf_final = lgb.LGBMClassifier(
    **best_params, scale_pos_weight=w_final,
    random_state=42, n_jobs=-1, verbose=-1,
)
clf_final.fit(df[stable_feats], df["label"])
joblib.dump(clf_final, "results/final_model.pkl")
print("final model trained on full train set -> results/final_model.pkl")
# cpcv.py, combinatorial purged cross-validation
# self-contained: includes all feature/label/regime logic.
# analysis.py and causal_validator.py import from here.
#
# requires: results/dollar_bars.csv
# outputs:  results/cpcv_performance.csv
#           results/cpcv_probs_*.csv  (per-fold probabilities for sizing.py)
#           results/feature_stability.csv
#           auto-updates FINAL_FEATURES in config.py after stability selection

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score
from hmmlearn import hmm
import lightgbm as lgb
import os
import talib as ta

os.makedirs("results", exist_ok=True)

from config import (
    MAX_HOLD, PT_SL,
    N_GROUPS, K_TEST, EMBARGO, MIN_SAMPLES_LEAF, FEATURES, BARS_PER_YEAR,
    FINAL_FEATURES, EDGE_THRESHOLD, MIN_AUC, STABILITY_THRESHOLD
)


# ── feature builder ───────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # define your features here

    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ── labeling ──────────────────────────────────────────────────────────────────
def get_daily_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    return np.log(close).diff().rolling(lookback).std()


def make_labels(close: pd.Series, max_hold: int = MAX_HOLD, pt_sl=PT_SL) -> pd.Series:
    vol    = get_daily_vol(close)
    labels = {}
    for i in range(len(close) - max_hold):
        v = vol.iloc[i]
        if pd.isna(v) or v < 0.001:
            continue
        price = close.iloc[i]
        upper = price * (1 + pt_sl[0] * v)
        lower = price * (1 - pt_sl[1] * v)
        label = np.nan
        for p in close.iloc[i + 1: i + 1 + max_hold]:
            if p >= upper:
                label = 1.0
                break
            elif p <= lower:
                label = 0.0
                break
        labels[close.index[i]] = label
    return pd.Series(labels, name='label')


# ── regime fitting ────────────────────────────────────────────────────────────
def fit_hmm_regimes(vol_vals: np.ndarray) -> np.ndarray:
    model = hmm.GaussianHMM(n_components=3, covariance_type='full',
                             n_iter=100, random_state=42)
    model.fit(vol_vals.reshape(-1, 1))
    raw   = model.predict(vol_vals.reshape(-1, 1))
    order = np.argsort(model.means_.flatten())
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[s] for s in raw])


# ── cpcv helpers ──────────────────────────────────────────────────────────────
def purge_embargo(train_idx: np.ndarray, test_idx: np.ndarray, embargo: int) -> np.ndarray:
    test_min, test_max = test_idx.min(), test_idx.max()
    purge_zone = np.arange(test_min - embargo, test_max + embargo + 1)
    return train_idx[~np.isin(train_idx, purge_zone)]


def regime_aucs(y_te: np.ndarray, prob: np.ndarray, regimes: np.ndarray) -> dict:
    out = {}
    for r, name in [(0, "low"), (1, "mid"), (2, "high")]:
        mask = regimes == r
        if mask.sum() >= 10 and len(np.unique(y_te[mask])) == 2:
            out[f"auc_{name}"] = round(roc_auc_score(y_te[mask], prob[mask]), 4)
        else:
            out[f"auc_{name}"] = np.nan
    return out


def train_lgbm(X_tr, y_tr):
    w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, num_leaves=15,
        min_child_samples=MIN_SAMPLES_LEAF, learning_rate=0.05,
        importance_type="gain",
        scale_pos_weight=w, random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


# ── config auto-update ────────────────────────────────────────────────────────
def update_config_features(stable_features: list, stab_df: pd.DataFrame,
                            config_path: str = "config.py") -> None:
    lines = ["FINAL_FEATURES = ["]
    for f in stable_features:
        row = stab_df[stab_df["feature"] == f].iloc[0]
        lines.append(f"    '{f}',  # stability={row['stability_rate']:.0%}  mdi={row['mean_mdi']:.5f}")
    lines.append("]")
    new_block = "\n".join(lines)

    with open(config_path, "r") as fh:
        src = fh.read()

    pattern = r"FINAL_FEATURES\s*=\s*\[.*?\]"
    if re.search(pattern, src, flags=re.DOTALL):
        updated = re.sub(pattern, new_block, src, flags=re.DOTALL)
        with open(config_path, "w") as fh:
            fh.write(updated)
        print(f"  config.py updated -> {len(stable_features)} stable features written")
    else:
        print("  WARNING: FINAL_FEATURES block not found in config.py, paste manually:")
        print(new_block)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("loading dollar bars...")
    df_raw = pd.read_csv("results/dollar_bars.csv", parse_dates=True, index_col=0)
    print(f"  raw bars: {len(df_raw)}")

    print("building features + labels...")
    df     = build_features(df_raw)
    labels = make_labels(df["close"])
    df     = df.join(labels)

    feats = [f for f in FEATURES if f in df.columns]
    df    = df.dropna(subset=["label"] + feats)
    df["label"] = df["label"].astype(int)
    n     = len(df)
    print(f"  usable bars: {n}")
    print(f"  label balance: {df['label'].mean():.3f}")
    print(f"  date range: {df.index[0].date()} to {df.index[-1].date()}")

    df["regime"] = fit_hmm_regimes(df["volatility_7b"].values)

    group_edges = np.linspace(0, n, N_GROUPS + 1, dtype=int)
    groups      = [np.arange(group_edges[i], group_edges[i + 1]) for i in range(N_GROUPS)]
    n_folds     = len(list(combinations(range(N_GROUPS), K_TEST)))

    print(f"\ncpcv: N={N_GROUPS} groups, K={K_TEST} test groups, embargo={EMBARGO} bars")
    print(f"total folds: {n_folds}  |  model: lgbm")
    print("─" * 70)

    records     = []
    mdi_records = []

    for test_groups in combinations(range(N_GROUPS), K_TEST):
        train_groups = [g for g in range(N_GROUPS) if g not in test_groups]
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

        clf  = train_lgbm(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y_te, prob)

        # mdi via gain importance (normalized)
        imp_raw = clf.feature_importances_
        imp     = imp_raw / imp_raw.sum() if imp_raw.sum() > 0 else imp_raw
        mdi_records.append(dict(zip(feats, imp)))

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
            "model":     "lgbm",
        }).to_csv(f"results/cpcv_probs_{test_groups}.csv", index=False)

        te_regime_vals = df["regime"].iloc[test_idx].values
        records.append({
            "fold":      str(test_groups),
            "test_from": str(t_from),
            "test_to":   str(t_to),
            "auc":       round(auc, 4),
            "n_train":   len(train_idx),
            "n_test":    len(test_idx),
            "pct_high_vol": round((te_regime_vals == 2).mean(), 3),
            "pct_low_vol":  round((te_regime_vals == 0).mean(), 3),
            "verdict":   verdict,
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
    print(f"mean auc:      {perf['auc'].mean():.4f}  std={perf['auc'].std():.4f}")
    print(f"edge folds:    {(perf['verdict'] == 'edge').sum()} / {len(perf)}")
    for reg in ["low", "mid", "high"]:
        valid = perf[f"auc_{reg}"].dropna()
        if len(valid):
            print(f"{reg}_vol auc:  mean={valid.mean():.4f}  n={len(valid)}")

    # ── mdi stability across folds ───────────────────────
    print("─" * 70)
    print("feature stability across folds:")

    mdi_df   = pd.DataFrame(mdi_records).fillna(0)
    n_folds_ = len(mdi_df)
    half     = len(feats) // 2

    top_half_counts = (mdi_df.rank(axis=1, ascending=False) <= half).sum()
    stability       = (top_half_counts / n_folds_).sort_values(ascending=False)
    mean_imp        = mdi_df.mean().sort_values(ascending=False)

    stab_df = pd.DataFrame({
        "feature":        stability.index,
        "stability_rate": stability.values.round(3),
        "mean_mdi":       mean_imp.reindex(stability.index).values.round(5),
    })

    stable_features = stab_df[stab_df["stability_rate"] >= STABILITY_THRESHOLD]["feature"].tolist()
    unstable        = stab_df[stab_df["stability_rate"] <  STABILITY_THRESHOLD]["feature"].tolist()

    print(f"  threshold: top-half rank in >= {STABILITY_THRESHOLD:.0%} of {n_folds_} folds")
    print(f"  stable features: {len(stable_features)} / {len(feats)}")
    print()
    print(f"  {'feature':<28} {'stability':>10}   {'mean_mdi':>10}")
    print("  " + "─" * 52)
    for _, row in stab_df.iterrows():
        marker = "  KEEP" if row["stability_rate"] >= STABILITY_THRESHOLD else "  drop"
        print(f"  {row['feature']:<28} {row['stability_rate']:>10.1%}   {row['mean_mdi']:>10.5f}{marker}")

    if unstable:
        print(f"\n  features to remove: {unstable}")

    stab_df.to_csv("results/feature_stability.csv", index=False)
    print("\n  saved -> results/feature_stability.csv")

    # ── auto-update config.py ─────────────────────────────────────────────────
    print("─" * 70)
    update_config_features(stable_features, stab_df)

    # ── final features validation run ─────────────────────────────────────────
    # re-runs cpcv with only the stable features to confirm selection
    # improves oos auc. same folds, same purging, same embargo.
    final_feats = [f for f in FINAL_FEATURES if f in df.columns]

    if final_feats and set(final_feats) != set(feats):
        print("─" * 70)
        print(f"validation run: {len(feats)} -> {len(final_feats)} features")

        final_records = []

        for test_groups in combinations(range(N_GROUPS), K_TEST):
            train_groups = [g for g in range(N_GROUPS) if g not in test_groups]
            test_idx     = np.concatenate([groups[g] for g in test_groups])
            train_idx    = np.concatenate([groups[g] for g in train_groups])
            train_idx    = purge_embargo(train_idx, test_idx, EMBARGO)

            if len(train_idx) < 100 or len(test_idx) < 20:
                continue

            X_tr = df[final_feats].iloc[train_idx]
            y_tr = df["label"].iloc[train_idx]
            X_te = df[final_feats].iloc[test_idx]
            y_te = df["label"].iloc[test_idx]

            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                continue

            clf = train_lgbm(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            auc  = roc_auc_score(y_te, prob)
            final_records.append({"fold": str(test_groups), "auc": auc})

        final_perf = pd.DataFrame(final_records)
        print(f"  all features mean auc:    {perf['auc'].mean():.4f}")
        print(f"  final features mean auc:  {final_perf['auc'].mean():.4f}")
        print(f"  delta:                    {final_perf['auc'].mean() - perf['auc'].mean():+.4f}")
    else:
        print("  final features identical to full feature set, skipping validation run")
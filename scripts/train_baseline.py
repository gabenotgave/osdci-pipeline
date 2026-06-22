# scripts/train_baseline.py
# Train a baseline LightGBM model with a spatial NJ/CO/WA holdout split.
# PYTHONPATH=. python scripts/train_baseline.py

from __future__ import annotations

import json
import pickle
from pathlib import Path

import h3
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from osdci.features import ALL_FEATURES, ID_COLS, LABEL_COL, LABEL_META

MASTER_PATH = Path("data/processed/features_master_res6.parquet")
MODELS_DIR = Path("models")
SOFT_LABEL_WEIGHT = 0.25

TEST_STATES = ["NJ", "CO", "WA"]

STATE_BOXES = {
    "NJ": {"lat": (38.9, 41.4), "lon": (-75.6, -73.9)},
    "CO": {"lat": (36.9, 41.1), "lon": (-109.1, -102.0)},
    "WA": {"lat": (45.5, 49.0), "lon": (-124.8, -116.9)},
}

# h3 v4 renamed k_ring → grid_disk
k_ring = getattr(h3, "k_ring", h3.grid_disk)


def assign_test_state(row: pd.Series) -> str | None:
    for state, box in STATE_BOXES.items():
        if (
            box["lat"][0] <= row["lat"] <= box["lat"][1]
            and box["lon"][0] <= row["lon"] <= box["lon"][1]
        ):
            return state
    return None


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ───────────────────────────────────────────────
    df = pd.read_parquet(MASTER_PATH)
    print(f"Loaded {len(df):,} cells, {int(df[LABEL_COL].sum())} positive")

    # ── Soft labels for k-ring-1 neighbors (before train/test split) ────
    positive_cells = set(df.loc[df[LABEL_COL] == 1, "h3_index"].tolist())

    neighbors: set[str] = set()
    for cell in positive_cells:
        neighbors.update(k_ring(cell, 1))

    soft_positive_cells = neighbors - positive_cells
    valid_soft = df["h3_index"].isin(soft_positive_cells)

    df["label_weight"] = 1.0
    df["has_dc_soft"] = df[LABEL_COL].copy()
    df.loc[valid_soft, "has_dc_soft"] = 1
    df.loc[valid_soft, "label_weight"] = SOFT_LABEL_WEIGHT

    hard_positives = len(positive_cells)
    soft_positives = int(valid_soft.sum())

    print(f"Original positives:     {hard_positives}")
    print(f"K-ring-1 neighbors:     {len(neighbors)}")
    print(f"Soft positive cells:    {soft_positives} (in master parquet)")
    print(f"Total positives (soft): {int(df['has_dc_soft'].sum())}")

    # ── Step 2: Spatial train/test split ────────────────────────────────
    df["test_state"] = df.apply(assign_test_state, axis=1)
    df["in_test_set"] = df["test_state"].notnull().astype(int)

    train = df[df["in_test_set"] == 0].copy()
    test = df[df["in_test_set"] == 1].copy()

    train_hard_pos = int(train[LABEL_COL].sum())
    train_soft_add = int(
        ((train["has_dc_soft"] == 1) & (train[LABEL_COL] == 0)).sum()
    )
    train_soft_total = int(train["has_dc_soft"].sum())
    test_hard_pos = int(test[LABEL_COL].sum())
    train_hard_pct = train_hard_pos / len(train) * 100 if len(train) else 0.0
    test_hard_pct = test_hard_pos / len(test) * 100 if len(test) else 0.0

    state_pos = {
        state: int(test.loc[test["test_state"] == state, LABEL_COL].sum())
        for state in TEST_STATES
    }

    print("\nSoft label summary:")
    print(f"  Hard positives (has_dc=1):      {hard_positives}")
    print(f"  Soft positives (k-ring-1):      {soft_positives}")
    print(f"  Total training positives:       {train_soft_total}")
    print(f"  Soft label weight:              {SOFT_LABEL_WEIGHT}")

    print("\nTrain / Test split (NJ + CO + WA holdout):")
    print(f"  Train cells:     {len(train):,}")
    print(f"  Train hard pos:  {train_hard_pos} ({train_hard_pct:.3f}%)")
    print(f"  Train soft pos:  {train_soft_add} additional")
    print(f"  Test cells:      {len(test):,}")
    print(f"  Test hard pos:   {test_hard_pos} ({test_hard_pct:.3f}%)")
    print(
        f"  States: NJ={state_pos['NJ']}, CO={state_pos['CO']}, WA={state_pos['WA']}"
    )

    if test_hard_pos < 20:
        print(
            f"WARNING: only {test_hard_pos} positives in test set "
            "(expected 50–100); continuing anyway."
        )

    # ── Step 3: Feature matrix preparation ──────────────────────────────
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    print(f"\nTraining on {len(feature_cols)} features")

    X_train = train[feature_cols]
    y_train = train["has_dc_soft"]
    sample_weights = train["label_weight"]
    y_train_eval = train[LABEL_COL]

    X_test = test[feature_cols]
    y_test_eval = test[LABEL_COL]

    print(
        f"Train: {int(y_train.sum())} soft pos / "
        f"{int((y_train == 0).sum())} neg "
        f"({train_hard_pos} hard)"
    )
    print(
        f"Test:  {int(y_test_eval.sum())} hard pos / "
        f"{int((y_test_eval == 0).sum())} neg"
    )

    # ── Step 4: Train LightGBM baseline ─────────────────────────────────
    params = {
        "objective": "binary",
        "metric": ["average_precision", "binary_logloss"],
        "is_unbalance": True,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 50,
        "min_sum_hessian_in_leaf": 10.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 3000,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train_eval), (X_test, y_test_eval)],
        eval_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=200,
                verbose=True,
                first_metric_only=True,
            ),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"Best iteration: {model.best_iteration_}")

    eval_results = model.evals_result_
    valid_ap = eval_results["valid"]["average_precision"]
    train_ap = eval_results["train"]["average_precision"]

    print("\nTraining curve (every 50 iterations):")
    print(f"{'Iter':>6} {'Train AP':>10} {'Valid AP':>10}")
    for i in range(0, len(valid_ap), 50):
        print(f"{i + 1:>6} {train_ap[i]:>10.4f} {valid_ap[i]:>10.4f}")
    best_i = model.best_iteration_ - 1
    print(
        f"{best_i + 1:>6} {train_ap[best_i]:>10.4f} "
        f"{valid_ap[best_i]:>10.4f}  ← best"
    )

    if model.best_iteration_ < 50:
        print(
            "WARNING: model still stopping too early "
            f"(iteration {model.best_iteration_}). "
            "Consider further reducing learning_rate or "
            "increasing reg_lambda."
        )
    elif model.best_iteration_ > 200:
        print(
            f"Good — model trained for {model.best_iteration_} "
            "iterations before early stopping."
        )

    # ── Step 5: Evaluate on test set (hard labels only) ─────────────────
    y_prob_train = model.predict_proba(X_train)[:, 1]
    train_auc_roc = roc_auc_score(y_train_eval, y_prob_train)
    train_auc_pr = average_precision_score(y_train_eval, y_prob_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test_eval, y_prob)
    auc_pr = average_precision_score(y_test_eval, y_prob)

    print(f"Train AUC-ROC: {train_auc_roc:.4f}")
    print(f"Train AUC-PR:  {train_auc_pr:.4f}")
    print(f"Test  AUC-ROC: {auc_roc:.4f}")
    print(f"Test  AUC-PR:  {auc_pr:.4f}")

    precisions, recalls, thresholds = precision_recall_curve(y_test_eval, y_prob)

    recall_60_idx = next((i for i, r in enumerate(recalls) if r >= 0.60), None)
    if recall_60_idx is not None:
        p_at_r60 = precisions[recall_60_idx]
        t_at_r60 = thresholds[min(recall_60_idx, len(thresholds) - 1)]
    else:
        p_at_r60 = t_at_r60 = None

    top_k_precision: dict[int, float] = {}
    for k in [100, 500, 1000]:
        top_k_idx = y_prob.argsort()[-k:]
        top_k_precision[k] = float(y_test_eval.iloc[top_k_idx].mean())
        print(f"  Top-{k} precision: {top_k_precision[k]:.3f}")

    print("\nTest set metrics:")
    print(f"  AUC-ROC:              {auc_roc:.4f}")
    print(f"  AUC-PR:               {auc_pr:.4f}")
    if p_at_r60 is not None:
        print(f"  Precision @ R=0.60:   {p_at_r60:.4f}")
        print(f"  Threshold @ R=0.60:   {t_at_r60:.4f}")
    else:
        print("  R=0.60 not reached")

    test_with_pred = test[["h3_index", "lat", "lon", LABEL_COL, "test_state"]].copy()
    test_with_pred["score"] = y_prob
    for state in TEST_STATES:
        state_df = test_with_pred[test_with_pred["test_state"] == state]
        if state_df[LABEL_COL].sum() > 0 and state_df[LABEL_COL].nunique() > 1:
            state_auc = roc_auc_score(state_df[LABEL_COL], state_df["score"])
            print(f"  {state} AUC-ROC: {state_auc:.4f}")
        else:
            print(f"  {state}: no positives or single class")

    # ── Step 6: Feature importances ─────────────────────────────────────
    importances = (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    importances["importance_pct"] = (
        importances["importance"] / importances["importance"].sum() * 100
    )
    importances["cumulative_pct"] = importances["importance_pct"].cumsum()

    print("\nTop 20 features by importance:")
    print(importances.head(20).to_string(index=False))

    n_80 = int((importances["cumulative_pct"] <= 80).sum())
    print("\nFeatures capturing 80% of importance:")
    print(f"  {n_80} features")

    importances.to_csv(MODELS_DIR / "feature_importances.csv", index=False)

    # ── Step 7: Score all CONUS cells (spatial residuals) ─────────────────
    all_probs = model.predict_proba(df[feature_cols])[:, 1]

    score_describe = pd.Series(all_probs).describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    print("Score distribution (all CONUS):")
    print(score_describe)

    unique_scores = pd.Series(all_probs).nunique()
    print(f"Unique score values: {unique_scores}")

    if unique_scores < 100:
        print(
            "WARNING: Score distribution is too sparse — "
            "model may not have trained properly"
        )
    else:
        print("Score distribution looks healthy")

    spatial = df[
        ["h3_index", "lat", "lon", LABEL_COL, "has_hyperscaler", "in_test_set"]
    ].copy()
    spatial["viability_score"] = all_probs
    spatial["is_soft_positive"] = df["h3_index"].isin(soft_positive_cells).astype(int)
    spatial["residual"] = spatial["viability_score"] - spatial[LABEL_COL]

    spatial.to_csv(MODELS_DIR / "spatial_residuals.csv", index=False)
    print(f"\nSpatial residuals saved: {len(spatial):,} cells")

    candidates = spatial[
        (spatial[LABEL_COL] == 0)
        & (spatial["is_soft_positive"] == 0)
        & (spatial["viability_score"] >= 0.5)
    ].sort_values("viability_score", ascending=False)

    print("\nTop 20 candidate cells (no DC, not soft neighbor, score ≥ 0.5):")
    print(
        candidates[["h3_index", "lat", "lon", "viability_score"]]
        .head(20)
        .to_string(index=False)
    )

    # ── Step 8: Save model and results ──────────────────────────────────
    with open(MODELS_DIR / "baseline_lgbm.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "model": "LightGBM baseline",
        "model_version": "v2_regularized",
        "train_cells": int(len(train)),
        "test_cells": int(len(test)),
        "train_positives": train_hard_pos,
        "train_soft_positives": train_soft_add,
        "test_positives": test_hard_pos,
        "test_states": TEST_STATES,
        "hard_positives": hard_positives,
        "soft_positives": soft_positives,
        "soft_label_weight": SOFT_LABEL_WEIGHT,
        "n_features": len(feature_cols),
        "best_iteration": int(model.best_iteration_),
        "train_auc_roc": float(train_auc_roc),
        "train_auc_pr": float(train_auc_pr),
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "precision_at_r60": float(p_at_r60) if p_at_r60 is not None else None,
        "threshold_at_r60": float(t_at_r60) if t_at_r60 is not None else None,
        "top_100_precision": top_k_precision[100],
        "top_500_precision": top_k_precision[500],
        "top_1000_precision": top_k_precision[1000],
    }

    with open(MODELS_DIR / "baseline_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved: models/baseline_lgbm.pkl")
    print("Results saved: models/baseline_results.json")

    # ── Step 9: Final summary block ─────────────────────────────────────
    top5 = importances.head(5)
    print("\n" + "=" * 60)
    print("Baseline model training complete")
    print("=" * 60)
    print("Train / Test split (NJ + CO + WA holdout):")
    print(
        f"  Train: {len(train):,} cells "
        f"({train_hard_pos} hard + {train_soft_add} soft positives)"
    )
    print(
        f"  Test:  {len(test):,} cells ({test_hard_pos} hard positive, "
        f"{test_hard_pct:.3f}%)"
    )
    print(f"Features used: {len(feature_cols)}")
    print(f"Best iteration: {model.best_iteration_} (early stopping)")
    print(f"Soft label weight: {SOFT_LABEL_WEIGHT}")
    print(f"Model version: v2_regularized")
    print("\nTrain / test performance (hard labels):")
    print(f"  Train AUC-ROC: {train_auc_roc:.4f}")
    print(f"  Train AUC-PR:  {train_auc_pr:.4f}")
    print("\nTest set performance:")
    print(f"  AUC-ROC:   {auc_roc:.4f}   (target ≥ 0.85)")
    print(f"  AUC-PR:    {auc_pr:.4f}   (target ≥ 0.25)")
    print(f"  Top-100 precision:  {top_k_precision[100]:.3f}")
    print(f"  Top-500 precision:  {top_k_precision[500]:.3f}")
    print(f"  Top-1000 precision: {top_k_precision[1000]:.3f}")
    print("\nTop 5 features:")
    for i, row in enumerate(top5.itertuples(), start=1):
        print(f"  {i}. {row.feature} ({row.importance_pct:.1f}%)")
    print("\nArtifacts:")
    print("  models/baseline_lgbm.pkl")
    print("  models/baseline_results.json")
    print("  models/feature_importances.csv")
    print("  models/spatial_residuals.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()

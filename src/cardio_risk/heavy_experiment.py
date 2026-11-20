from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import make_dataset

SEEDS = (27, 41, 73, 101, 137)
SHIFTS = {"in_distribution": 0.0, "scanner_shift_moderate": 0.35, "scanner_shift_severe": 0.70}


def _shifted_dataset(n: int, seed: int, severity: float):
    """Generate outcomes first, then corrupt only observed imaging descriptors."""
    x, y, names = make_dataset(n, seed, scanner_shift=0.0)
    if severity:
        rng = np.random.default_rng(seed + 50_000)
        imaging = x[:, :4]
        scale = 1.0 + severity * np.array([.45, -.30, .25, -.20])
        bias = severity * np.array([.35, -.25, .20, .15])
        x[:, :4] = imaging * scale + bias + rng.normal(0, severity * .85, imaging.shape)
    return x, y, names


def _models(seed: int):
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=seed))
    nonlinear = HistGradientBoostingClassifier(max_iter=180, learning_rate=.055, max_leaf_nodes=15,
                                                l2_regularization=1.0, random_state=seed)
    robust = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(C=.7, max_iter=1500, random_state=seed)),
        method="sigmoid", cv=3,
    )
    return {"logistic_baseline": baseline, "nonlinear": nonlinear, "robust_calibrated": robust}


def _training_data(seed: int, robust: bool):
    if not robust:
        return make_dataset(2400, seed)[:2]
    blocks = [_shifted_dataset(1000, seed + i * 17, s)[:2]
              for i, s in enumerate((0.0, 0.35, 0.70))]
    return np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks])


def _metric_row(model_name, condition, seed, y, prob, x):
    pred = prob >= .5
    groups = x[:, 4] >= np.median(x[:, 4])
    group_rates = [float(np.mean(pred[g] == y[g])) for g in (groups, ~groups)]
    return {
        "model": model_name, "condition": condition, "seed": seed,
        "auroc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "accuracy_gap": abs(group_rates[0] - group_rates[1]),
    }


def run():
    rows = []
    for seed in SEEDS:
        standard_x, standard_y = _training_data(seed, False)
        robust_x, robust_y = _training_data(seed, True)
        models = _models(seed)
        for name, model in models.items():
            x_train, y_train = (robust_x, robust_y) if name == "robust_calibrated" else (standard_x, standard_y)
            model.fit(x_train, y_train)
            for condition, shift in SHIFTS.items():
                x, y, _ = _shifted_dataset(1000, seed + 1000, shift)
                rows.append(_metric_row(name, condition, seed, y, model.predict_proba(x)[:, 1], x))
    summary = {}
    for model in sorted({r["model"] for r in rows}):
        summary[model] = {}
        for condition in SHIFTS:
            subset = [r for r in rows if r["model"] == model and r["condition"] == condition]
            summary[model][condition] = {}
            for metric in ("auroc", "average_precision", "brier", "accuracy_gap"):
                values = np.array([r[metric] for r in subset])
                margin = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
                summary[model][condition][metric] = {
                    "mean": float(values.mean()), "ci95_low": float(values.mean() - margin),
                    "ci95_high": float(values.mean() + margin),
                }
    base = summary["logistic_baseline"]["scanner_shift_severe"]["auroc"]["mean"]
    proposed = summary["robust_calibrated"]["scanner_shift_severe"]["auroc"]["mean"]
    return rows, {"protocol": {"seeds": list(SEEDS), "train_standard": 2400,
                               "train_robust": 3000, "test_per_condition": 1000,
                               "shifts": SHIFTS}, "summary": summary,
                  "primary_comparison": {"metric": "severe-shift AUROC", "baseline": base,
                                         "proposed": proposed, "absolute_gain": proposed - base}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/heavy"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, report = run()
    (args.output / "heavy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report["primary_comparison"], indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import make_dataset


def evaluate(seed: int = 27):
    x_train, y_train, names = make_dataset(1600, seed)
    x_test, y_test, _ = make_dataset(600, seed + 1, scanner_shift=0.35)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    metrics = {
        "auroc": roc_auc_score(y_test, prob),
        "average_precision": average_precision_score(y_test, prob),
        "brier_score": brier_score_loss(y_test, prob),
        "accuracy": accuracy_score(y_test, prob >= 0.5),
    }
    coef = model.named_steps["logisticregression"].coef_[0]
    importance = sorted(zip(names, coef), key=lambda item: abs(item[1]), reverse=True)
    return metrics, importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics, importance = evaluate(args.seed)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output / "feature_importance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "standardized_coefficient"])
        writer.writerows(importance)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

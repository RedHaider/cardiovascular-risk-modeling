from cardio_risk.data import make_dataset
from cardio_risk.experiment import evaluate
from cardio_risk.heavy_experiment import run


def test_generator_is_reproducible():
    x1, y1, _ = make_dataset(30, 4)
    x2, y2, _ = make_dataset(30, 4)
    assert (x1 == x2).all() and (y1 == y2).all()


def test_experiment_has_valid_metrics():
    metrics, importance = evaluate(9)
    assert all(0 <= value <= 1 for value in metrics.values())
    assert len(importance) == 6


def test_heavy_experiment_is_complete():
    rows, report = run()
    assert len(rows) == 45
    assert report["protocol"]["seeds"] == [27, 41, 73, 101, 137]
    assert all(0 <= row["auroc"] <= 1 for row in rows)

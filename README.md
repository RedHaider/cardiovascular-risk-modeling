# Cardio Image Risk Lite

An original, privacy-safe portfolio benchmark exploring AI-assisted
cardiovascular imaging. It trains an
explainable classifier on synthetic chest-image descriptors to estimate a
cardiovascular-risk label and reports discrimination, calibration, and feature
importance.

This is not a reproduction of a professor's method and is not a clinical tool.
It uses no patient data.

## Run

```powershell
python -m pip install -e .
python -m cardio_risk.experiment --output outputs/demo
pytest -q
```

## Research question

Can a small, interpretable model combine synthetic imaging biomarkers and
basic clinical covariates while remaining reasonably calibrated under a
simulated scanner shift?

## Outputs

- `metrics.json`: AUROC, average precision, Brier score, and accuracy
- `feature_importance.csv`: standardized logistic-regression coefficients

## Ethical scope

The generator is educational. Its variables, relationships, and labels are
simulated and must not be interpreted as medical evidence.

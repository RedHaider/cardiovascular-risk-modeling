# Cardio Image Risk: Shift and Calibration Benchmark

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

## Heavy benchmark

The expanded protocol compares logistic regression, nonlinear gradient boosting,
and a calibrated shift-augmented model across five independent seeds. Each seed
uses separate training data and 1,000 held-out examples for each of three scanner
conditions. It reports AUROC, average precision, Brier score, subgroup accuracy
gap, and 95% confidence intervals.

```powershell
python -m cardio_risk.heavy_experiment --output outputs/heavy
```

The improvement claim is restricted to this reproducible synthetic protocol. It
does not claim to improve a professor's published model or establish clinical
utility.

### Verified result

Across five seeds, measurement-error augmentation plus sigmoid calibration
improved severe-shift AUROC from **0.836** to **0.839** (+0.003) and reduced
Brier score from **0.170** to **0.157** (-0.013, a 7.5% reduction). The AUROC
confidence intervals overlap, so the ranking improvement is described as
modest; the clearer benefit is calibration under simulated scanner artifacts.
The subgroup accuracy gap did not improve and is reported as a limitation.
See `results/heavy_metrics.csv` for confidence intervals.

## Ethical scope

The generator is educational. Its variables, relationships, and labels are
simulated and must not be interpreted as medical evidence.

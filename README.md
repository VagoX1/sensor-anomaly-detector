# Sensor anomaly detector

A runnable SKAB project that detects abnormal pump operation and compares
Isolation Forest with a supervised Extra Trees classifier.
Uses sensor values, trailing means, standard deviations, and changes. Labels are
excluded from features. Windows never cross experiment boundaries.

## Quick start

With Python and pip installed, open a terminal in the project folder:

```powershell
python -m pip install -r requirements.txt
python download_data.py
python train.py
```

Downloads require internet. The downloader skips existing files.

## Outputs

`outputs/metrics.json`: point precision, recall, F1, average precision, event recall,
median delay for detected events, estimated false alert starts per normal hour,
and the exact experiment split.

`outputs/test_*.png`: pressure, temperature, and anomaly scores.

`outputs/test_*_predictions.csv`: sensor readings, true labels, scores, predictions.

`outputs/model.joblib`: trained model, threshold, feature columns, and window length.
Load joblib artifacts only from sources you trust.

## Evaluation choices

Entire experiments are assigned deterministically to roughly 60% training, 20%
validation, and 20% test. Only fully normal windows train Isolation Forest.
Extra Trees uses labeled training windows, including anomalies; it is a supervised
alternative, not an unsupervised detector. Neither model needs scaling.

The default compares Isolation Forest and three Extra Trees leaf-size settings.
For each model, it searches trailing exponential smoothing spans (1, 10, 30, 60)
and validation-score quantile thresholds. Selection maximizes average point F1
across the seven validation experiments, giving each experiment equal weight.
Models, smoothing, and thresholds are selected without using test labels.
Smoothing resets per experiment and never uses future readings.

`outputs/model_comparison.json` records the best validation configuration for
each candidate. `outputs/validation_search.json` also contains the Isolation
Forest threshold search. Saved models include their smoothing configuration.

For a normal-only training approach with labeled validation calibration:
`python train.py --unsupervised --output outputs_improved`.
For the original 99th-percentile threshold:
`python train.py --baseline --output outputs_baseline`.
These modes use the same experiment split as the default.

The first 29 readings of each experiment are omitted with the default window.
Window size counts readings, not seconds. Missing sensor values are forward-filled
without accessing future readings; initial unavailable windows are omitted.
The split tests held-out experiments, not future deployment or unseen equipment.
False alerts count starts of false-positive runs, including predictions that
continue after an anomaly ends. Normal
exposure is estimated from the median sampling interval. Detection delay is measured
from the first retained labeled point; events during warm-up are not evaluated.

`normal_readings_flagged_fraction` measures how much normal data is flagged;
event recall alone can look good for a detector that stays in alarm. Detection
delay can be zero because an alarm was already active before the event.

Try `python train.py --baseline --window 60 --normal-quantile 0.995` for a more conservative
baseline. Choose settings using validation experiments rather than repeatedly
optimizing test results. Run checks with `python -m unittest`.
Next steps: compare a convolutional autoencoder, analyze errors per experiment,
and add a live visualization after establishing a reliable baseline.

## Data source

[SKAB](https://github.com/waico/SKAB) is an industrial sensor anomaly benchmark.
Downloaded data remains subject to the upstream license (GPL-3.0); see its repository.
Data and generated outputs are ignored by Git.

## Results from the initial comparison

On the same seven held-out experiments, the original baseline achieved point
F1 0.059 and detected 2/7 events. The validation-selected supervised Extra Trees
model achieved precision 0.798, recall 0.838, F1 0.817, average precision 0.835,
and detected 7/7 events. It flagged 11.5% of normal readings, with about 130
false-positive run starts per normal hour. This is still too noisy for unattended
alerts; the improvement is in classification, not proven operational readiness.
The comparison also changes the learning setup: supervised training uses anomaly
labels that the baseline does not. Results are specific to this small test split.

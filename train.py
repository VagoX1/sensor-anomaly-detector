"""Compare detectors on validation experiments and evaluate the selected model."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'outputs' / '.matplotlib'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

def smooth_scores(scores, span):
    """Trailing smoothing resets for every experiment; never sees future scores."""
    return pd.Series(scores).ewm(span=span, adjust=False).mean().to_numpy()

def calibrate(validation):
    """Choose smoothing and threshold on labeled validation experiments only."""
    trials = []
    best = None
    for span in [1, 10, 30, 60]:
        smoothed = [(y, smooth_scores(scores, span)) for y, scores in validation]
        thresholds = np.unique(np.quantile(np.concatenate([s for _, s in smoothed]),
                                           np.linspace(0.05, 0.99, 80)))
        for threshold in thresholds:
            f1s = [precision_recall_fscore_support(y, s >= threshold, average='binary',
                                                  zero_division=0)[2] for y, s in smoothed]
            trial = {'span': span, 'threshold': float(threshold),
                     'validation_macro_f1': float(np.mean(f1s))}
            trials.append(trial)
            if best is None or trial['validation_macro_f1'] > best['validation_macro_f1']:
                best = trial
    return best, trials

def score_model(model, x):
    if isinstance(model, IsolationForest):
        return -model.score_samples(x)
    return model.predict_proba(x)[:, list(model.classes_).index(1)]

def load_experiment(path, window):
    df = pd.read_csv(path, sep=';', parse_dates=['datetime'])
    if 'anomaly' not in df:
        return None
    df = df.sort_values('datetime').set_index('datetime')
    if df.index.has_duplicates or df['anomaly'].isna().any():
        raise ValueError(f'Invalid timestamps or missing labels: {path}')
    sensors = df.drop(columns=['anomaly', 'changepoint'], errors='ignore').apply(pd.to_numeric)
    # Forward fill is causal; initial missing values remain unavailable.
    sensors = sensors.replace([np.inf, -np.inf], np.nan).ffill()
    rolling = sensors.rolling(window, min_periods=window)
    x = pd.concat([sensors.add_suffix('_value'), rolling.mean().add_suffix('_mean'),
                   rolling.std().add_suffix('_std'), sensors.diff().add_suffix('_delta')], axis=1)
    # Never train on windows that contain an anomalous observation.
    clean = df['anomaly'].rolling(window, min_periods=window).max().eq(0)
    available = x.notna().all(axis=1)
    return df.loc[available], x.loc[available], clean.loc[available]

def event_metrics(y, predictions, timestamps):
    starts = np.flatnonzero((y == 1) & np.r_[True, y[:-1] == 0])
    ends = np.flatnonzero((y == 1) & np.r_[y[1:] == 0, True])
    delays = []
    for start, end in zip(starts, ends):
        hits = np.flatnonzero(predictions[start:end + 1])
        if len(hits):
            delays.append((timestamps[start + hits[0]] - timestamps[start]).total_seconds())
    false_starts = predictions.astype(bool) & (y == 0)
    false_starts &= np.r_[True, ~false_starts[:-1]]
    # Estimate observed normal exposure using median sampling interval.
    interval = pd.Series(timestamps).diff().dt.total_seconds().median()
    hours = float((y == 0).sum() * interval / 3600)
    return len(starts), len(delays), delays, int(false_starts.sum()), hours

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--normal-quantile', type=float, default=0.99)
    parser.add_argument('--baseline', action='store_true', help='Use the original normal-only threshold')
    parser.add_argument('--unsupervised', action='store_true', help='Exclude the supervised classifier')
    args = parser.parse_args()
    if args.window < 2 or not 0 < args.normal_quantile < 1:
        parser.error('window must be >= 2 and normal-quantile must be between 0 and 1')
    experiments = {}
    for path in sorted(args.data.rglob('*.csv')):
        loaded = load_experiment(path, args.window)
        if loaded is not None and len(loaded[0]) and loaded[0]['anomaly'].eq(1).any():
            experiments[str(path.relative_to(args.data))] = loaded
    if len(experiments) < 5:
        raise SystemExit('At least five labeled experiments required. Run python download_data.py first.')
    train_ids, heldout = train_test_split(sorted(experiments), test_size=0.4, random_state=42)
    val_ids, test_ids = train_test_split(heldout, test_size=0.5, random_state=42)
    columns = experiments[train_ids[0]][1].columns
    if any(not x.columns.equals(columns) for _, x, _ in experiments.values()):
        raise ValueError('Experiments have different sensor columns.')
    train_x = pd.concat([experiments[k][1].loc[experiments[k][2]] for k in train_ids])
    normal_val = pd.concat([experiments[k][1].loc[experiments[k][2]] for k in val_ids])
    if train_x.empty or normal_val.empty:
        raise ValueError('No clean normal windows in training or validation split.')
    model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(train_x)
    baseline_threshold = float(np.quantile(-model.score_samples(normal_val), args.normal_quantile))
    validation = [(experiments[k][0]['anomaly'].to_numpy(dtype=int),
                   -model.score_samples(experiments[k][1])) for k in val_ids]
    selected, trials = calibrate(validation)
    selected['model_type'] = 'isolation_forest'
    candidates = [dict(selected)]
    if not args.baseline and not args.unsupervised:
        labeled_x = pd.concat([experiments[k][1] for k in train_ids])
        labeled_y = pd.concat([experiments[k][0]['anomaly'] for k in train_ids]).astype(int)
        for leaf_size in [5, 20, 50]:
            classifier = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=leaf_size,
                                             class_weight='balanced', random_state=42, n_jobs=-1)
            classifier.fit(labeled_x, labeled_y)
            candidate_validation = [(experiments[k][0]['anomaly'].to_numpy(dtype=int),
                                     score_model(classifier, experiments[k][1])) for k in val_ids]
            candidate, _ = calibrate(candidate_validation)
            candidate.update(model_type='extra_trees_supervised', min_samples_leaf=leaf_size)
            candidates.append(dict(candidate))
            if candidate['validation_macro_f1'] > selected['validation_macro_f1']:
                selected, model = candidate, classifier
    if args.baseline:
        selected = {'span': 1, 'threshold': baseline_threshold, 'model_type': 'isolation_forest'}
    threshold = selected['threshold']
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'threshold': threshold, 'window': args.window,
                 'smoothing_span': selected['span'],
                 'configuration': selected,
                 'feature_columns': list(columns)}, args.output / 'model.joblib')
    all_y, all_scores, all_predictions = [], [], []
    events = detected = false_alerts = 0
    normal_hours = 0.0
    delays = []
    for number, key in enumerate(test_ids):
        df, x, _ = experiments[key]
        scores = smooth_scores(score_model(model, x), selected['span'])
        predictions = (scores >= threshold).astype(int)
        y = df['anomaly'].to_numpy(dtype=int)
        all_y.extend(y); all_scores.extend(scores); all_predictions.extend(predictions)
        count, hits, lag, false, hours = event_metrics(y, predictions, df.index)
        events += count; detected += hits; delays.extend(lag)
        false_alerts += false; normal_hours += hours
        result = df.copy()
        result['score'] = scores
        result['prediction'] = predictions
        result.to_csv(args.output / f'test_{number}_predictions.csv')
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        for ax, sensor in zip(axes[:2], ['Pressure', 'Temperature']):
            df[sensor].plot(ax=ax)
            ax.fill_between(df.index, 0, 1, where=y.astype(bool),
                            transform=ax.get_xaxis_transform(), color='red', alpha=.15)
            ax.set_ylabel(sensor)
        axes[2].plot(df.index, scores, label='Anomaly score')
        axes[2].axhline(threshold, color='red', linestyle='--', label='Alert threshold')
        axes[2].legend()
        fig.suptitle(f'{key} — red shading: labeled anomaly')
        fig.tight_layout()
        fig.savefig(args.output / f'test_{number}.png', dpi=140)
        plt.close(fig)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y, all_predictions, average='binary', zero_division=0)
    metrics = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
               'average_precision': float(average_precision_score(all_y, all_scores)),
               'normal_readings_flagged_fraction': float(np.mean(np.array(all_predictions)[np.array(all_y) == 0])),
               'event_recall': detected / events if events else None,
               'median_detection_delay_seconds': float(np.median(delays)) if delays else None,
               'false_alert_starts_per_normal_hour': false_alerts / normal_hours if normal_hours else None,
               'threshold': threshold, 'training_normal_windows': len(train_x),
               'training_total_windows': sum(len(experiments[k][1]) for k in train_ids),
               'selected_configuration': selected,
               'splits': {'train': train_ids, 'validation': val_ids, 'test': test_ids}}
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (args.output / 'validation_search.json').write_text(json.dumps(
        {'isolation_forest_trials': trials, 'best_per_model': candidates}, indent=2), encoding='utf-8')
    (args.output / 'model_comparison.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    print(f'Plots, predictions, and model saved to {args.output}')

if __name__ == '__main__':
    main()

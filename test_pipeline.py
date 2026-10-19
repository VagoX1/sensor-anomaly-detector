"""Check causal features, clean-window filtering, and event accounting."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from train import load_experiment, event_metrics, smooth_scores, calibrate, score_model
from sklearn.ensemble import ExtraTreesClassifier

class PipelineTests(unittest.TestCase):
    def test_supervised_score_matches_positive_probability(self):
        x = pd.DataFrame({'value': [0, 0.1, 0.2, 5, 6, 7]})
        model = ExtraTreesClassifier(n_estimators=10, random_state=42).fit(x, [0, 0, 0, 1, 1, 1])
        scores = score_model(model, x)
        np.testing.assert_array_equal(scores, model.predict_proba(x)[:, 1])
        self.assertGreater(scores[-1], scores[0])

    def test_smoothing_is_causal(self):
        original = smooth_scores([1, 2, 3, 4, 5], 10)
        changed = smooth_scores([1, 2, 3, 100, 200], 10)
        np.testing.assert_array_equal(original[:3], changed[:3])

    def test_calibration_separates_validation_classes(self):
        selected, _ = calibrate([(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))])
        self.assertEqual(selected['validation_macro_f1'], 1.0)

    def test_false_alarm_after_anomaly_counts(self):
        result = event_metrics(np.array([1, 1, 0, 0]), np.ones(4, dtype=int),
                               pd.date_range('2020-01-01', periods=4, freq='s'))
        self.assertEqual(result[3], 1)

    def test_features_do_not_use_future_or_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'sample.csv'
            df = pd.DataFrame({'datetime': pd.date_range('2020-01-01', periods=8, freq='s'),
                               'sensor': range(8), 'anomaly': [0, 0, 0, 1, 0, 0, 0, 0]})
            df.to_csv(path, sep=';', index=False)
            _, original, clean = load_experiment(path, 3)
            self.assertFalse(any('anomaly' in c for c in original.columns))
            self.assertEqual(clean.tolist(), [True, False, False, False, True, True])
            df.loc[6:, 'sensor'] = 999
            df.to_csv(path, sep=';', index=False)
            _, changed, _ = load_experiment(path, 3)
            pd.testing.assert_frame_equal(original.iloc[:4], changed.iloc[:4])

    def test_event_delay_and_false_alert_runs(self):
        times = pd.date_range('2020-01-01', periods=8, freq='s')
        result = event_metrics(np.array([0, 0, 1, 1, 0, 0, 1, 1]),
                               np.array([1, 1, 0, 1, 0, 0, 0, 0]), times)
        self.assertEqual(result[:4], (2, 1, [1.0], 1))

if __name__ == '__main__':
    unittest.main()

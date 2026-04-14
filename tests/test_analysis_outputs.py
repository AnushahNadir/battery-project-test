import unittest

import numpy as np
import pandas as pd

from src.analysis.rul import add_rul
from src.modeling.anomaly_detection import detect_anomalies


class _DummyModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, df):
        return self._preds


class TestAnalysisOutputs(unittest.TestCase):
    def test_add_rul_expected_values(self):
        cycle_table = pd.DataFrame(
            {
                "battery_id": ["A", "A", "A", "B", "B"],
                "cycle_index": [1, 2, 3, 1, 2],
                "filename": ["a1", "a2", "a3", "b1", "b2"],
                "capacity": [1.0, 0.8, 0.6, 2.0, 1.5],
            }
        )

        out = add_rul(cycle_table, alpha=0.7)
        a = out[out["battery_id"] == "A"].sort_values("cycle_index")
        b = out[out["battery_id"] == "B"].sort_values("cycle_index")

        self.assertListEqual(a["RUL"].tolist(), [2, 1, 0])
        self.assertListEqual(b["RUL"].tolist(), [1, 0])
        self.assertTrue((out["RUL"] >= 0).all())

    def test_detect_anomalies_non_zero_index(self):
        df = pd.DataFrame(
            {
                "battery_id": ["B1", "B1", "B1"],
                "cycle_index": [10, 11, 12],
                "RUL": [1.0, 2.0, 100.0],
            },
            index=[100, 200, 300],
        )
        model = _DummyModel([1.0, 2.0, 0.0])

        anomalies = detect_anomalies(df=df, model=model, top_fraction=0.2)

        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0].battery_id, "B1")
        self.assertEqual(anomalies[0].cycle_index, 12)


if __name__ == "__main__":
    unittest.main()

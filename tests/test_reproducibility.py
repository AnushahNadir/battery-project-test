import unittest

import numpy as np
import pandas as pd

from src.modeling.dl_sequence_model import TemporalSequenceModel
from src.modeling.ml_model import XGBoostRULModel


class TestFeaturePreparation(unittest.TestCase):
    def test_prepare_features_converts_bracketed_numeric_strings(self):
        model = XGBoostRULModel()
        df = pd.DataFrame(
            {
                "capacity": ["[1.0]"],
                "temp_mean": ["[25.0]"],
                "temp_max": ["26.0"],
                "v_min": ["3.1"],
                "v_mean": ["3.3"],
                "i_mean": ["-1.2"],
                "i_min": ["-1.5"],
                "energy_j": ["120.0"],
                "ah_est": ["0.2"],
                "duration_s": ["1000"],
            }
        )

        X = model._prepare_features(df)
        self.assertEqual(X.shape, (1, len(model.feature_columns)))
        self.assertTrue(np.isfinite(X).all())

    def test_sequence_model_trains_and_predicts_shape(self):
        rows = []
        for bid in ["B1", "B2"]:
            for cyc in range(1, 7):
                rows.append(
                    {
                        "battery_id": bid,
                        "cycle_index": cyc,
                        "RUL": float(10 - cyc),
                        "capacity": 1.0 - 0.03 * cyc,
                        "temp_mean": 25.0 + 0.1 * cyc,
                        "temp_max": 27.0 + 0.2 * cyc,
                        "v_min": 3.1,
                        "v_mean": 3.3,
                        "i_mean": -1.1,
                        "i_min": -1.3,
                        "energy_j": 100.0 + cyc,
                        "ah_est": 0.2,
                        "duration_s": 1000 + 5 * cyc,
                    }
                )
        df = pd.DataFrame(rows)

        seq_model = TemporalSequenceModel(sequence_length=4, force_backend="mlp")
        seq_model.fit(df)
        preds = seq_model.predict(df)

        self.assertEqual(len(preds), len(df))
        self.assertTrue(np.isfinite(preds).all())


if __name__ == "__main__":
    unittest.main()

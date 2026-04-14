import subprocess
import sys
import unittest
from pathlib import Path


class TestPipelineArtifactsIntegration(unittest.TestCase):
    def test_run_full_pipeline_creates_required_artifacts(self):
        root = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, '-m', 'src.modeling.run_full_pipeline']
        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)

        self.assertEqual(
            result.returncode,
            0,
            msg=(result.stdout[-2000:] + '\n' + result.stderr[-2000:]).strip(),
        )

        out = root / 'data/processed/modeling'
        models = root / 'trained_models'
        required = [
            'anomalies.json',
            'uncertainty_estimates.json',
            'survival_risk_predictions.csv',
            'survival_risk_metrics.json',
            'feature_importance.json',
            'degradation_hypotheses.json',
            'counterfactual_examples.json',
            'final_system_report.md',
        ]

        missing = [name for name in required if not (out / name).exists()]
        self.assertFalse(missing, msg=f'Missing artifacts: {missing}')
        self.assertTrue((models / 'dl_sequence_model.pkl').exists(), msg='Missing trained DL sequence model artifact')


if __name__ == '__main__':
    unittest.main()

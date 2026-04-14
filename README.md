# Battery Project 3

End-to-end battery analysis pipeline with:
- Stage 1-2 preprocessing from raw metadata + time-series files
- Stage 3-6 modeling, uncertainty, anomaly detection, reasoning, and supervisor report
- A third trajectory-aware sequence model (TCN when PyTorch is available, otherwise sequence-MLP fallback)
- Notebook visualization and smoke checks

## 1) Setup (PowerShell)

```powershell
cd D:atteryattery-project3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Run preprocessing (Stage 1-2)

```powershell
python -m src.main `
  --metadata dataaw\metadata.csv `
  --raw_root dataaw\data `
  --out_dir outputsun_001 `
  --alpha 0.7 `
  --non_interactive
```

Expected artifacts in `outputs\run_001`:
- `cycle_features_with_rul.csv`
- `validation_report.json`
- `confidence_report.json`
- `qc_report.json`

## 3) Run modeling pipeline (Stage 3-6)

```powershell
python -m src.modeling.run_full_pipeline
```

Expected artifacts in `data\processed\modeling`:
- `uncertainty_estimates.json`
- `anomalies.json`
- `feature_importance.json`
- `degradation_hypotheses.json`
- `counterfactual_examples.json`
- `final_system_report.md`

## 4) Validate notebook

```powershell
python scripts\check_notebook.py battery_analysis.ipynb
```

## 5) Run tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## 6) Create release snapshot

```powershell
python scripts\make_snapshot.py --tag release_candidate
```


# React Dashboard Component

`dashboard/BatteryDashboard.jsx` is a drop-in React component wired to the
pipeline artifacts produced by this project.

## Data files expected by the component

- `/data/processed/modeling/uncertainty_estimates.json`
- `/data/processed/modeling/conformal_coverage_report.json`
- `/data/processed/modeling/drift_report.json`
- `/data/processed/hostile_validation/hostile_results.json`
- `/trained_models/model_metrics.json`
- `/data/processed/cycle_features_with_rul.csv`

## Notes

- The component normalizes real schema keys used in this repo:
  - `tc_id` from hostile results
  - `rul_lower_5` / `rul_upper_95` from uncertainty output
  - `mean_shift` from drift output
- If you mount this in a React app, configure static serving so the paths
  above are accessible at runtime.

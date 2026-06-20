import json
import numpy as np
import pandas as pd
from pathlib import Path

mm   = json.load(open('trained_models/model_metrics.json'))
ms   = json.load(open('trained_models/multi_seed_benchmark_5seed.json'))
conf = json.load(open('data/processed/modeling/conformal_coverage_report.json'))
feat = json.load(open('data/processed/modeling/feature_importance.json'))
abl  = json.load(open('trained_models/ablation_results.json'))
drift = json.load(open('data/processed/modeling/drift_report.json'))

out = []

# T3
out.append('# Table T3: Model RMSE Across 5 Seeds\n')
out.append('| Model | Seed 42 RMSE | Mean (5 seeds) | Std (5 seeds) |')
out.append('|-------|-------------|----------------|---------------|')
xgb_vals = [r['xgb_rmse'] for r in ms['runs']]
dl_vals  = [r['dl_rmse']  for r in ms['runs']]
out.append(f'| XGBoost | {mm["rmse"]:.2f} | {np.mean(xgb_vals):.2f} | {np.std(xgb_vals):.2f} |')
out.append(f'| TCN/MLP (DL) | {mm["dl_sequence"]["rmse"]:.2f} | {np.mean(dl_vals):.2f} | {np.std(dl_vals):.2f} |')
out.append('| Statistical baseline | ~45.64 | 319.66 | 196.58 (auto-excluded from ensemble) |')
out.append('| Ensemble (ml=0.557, dl=0.443) | -- | Not computed per-seed | -- |\n')

# T4
out.append('# Table T4: Per-Battery XGBoost RMSE (Seed 42)\n')
out.append('| Battery | Temp group | XGBoost RMSE | DL RMSE |')
out.append('|---------|-----------|-------------|---------|')
pbr = mm.get('per_battery_rmse', {})
pbd = mm.get('dl_sequence', {}).get('per_battery_rmse', {})
temp_map = {'B0030':'hot','B0033':'room','B0036':'room','B0041':'cold','B0044':'cold','B0052':'cold'}
for bat in ['B0030','B0033','B0036','B0041','B0044','B0052']:
    xr = pbr.get(bat, 'N/A')
    dr = pbd.get(bat, 'N/A')
    xr_s = f'{xr:.2f}' if isinstance(xr, float) else str(xr)
    dr_s = f'{dr:.2f}' if isinstance(dr, float) else str(dr)
    out.append(f'| {bat} | {temp_map[bat]} | {xr_s} | {dr_s} |')
out.append('')

# T5
out.append('# Table T5: XGBoost Feature Importance (Seed 42)\n')
out.append('| Rank | Feature | Importance | Direction |')
out.append('|------|---------|-----------|-----------|')
features = feat if isinstance(feat, list) else feat.get('features', [])
for i, f in enumerate(features, 1):
    out.append(f'| {i} | {f["feature"]} | {f["importance"]:.3f} | {f.get("direction","--")} |')
out.append('')

# T6
out.append('# Table T6: Feature Ablation Results\n')
out.append('| Configuration | Mean RMSE | Std | Delta vs full |')
out.append('|---------------|----------|-----|--------------|')
full_rmse = abl['full']['summary']['xgb_rmse']['mean']
full_std  = abl['full']['summary']['xgb_rmse']['std']
out.append(f'| Full (10 features) | {full_rmse:.2f} | {full_std:.2f} | -- |')
for key, label in [('no_cycle_index','Remove cycle_index'), ('no_capacity','Remove capacity')]:
    if key in abl:
        m = abl[key]['summary']['xgb_rmse']['mean']
        s = abl[key]['summary']['xgb_rmse']['std']
        delta = m - full_rmse
        out.append(f'| {label} | {m:.2f} | {s:.2f} | +{delta:.2f} |')
out.append('')

# T7
out.append('# Table T7: Per-Group Conformal Coverage\n')
out.append('| Temp group | Strategy | q_hat (cycles) | Empirical coverage | vs 80% target |')
out.append('|-----------|---------|---------------|-------------------|--------------|')
pg = conf.get('per_group', {})
for grp in ['room', 'hot', 'cold']:
    if grp in pg:
        g = pg[grp]
        cov   = g.get('empirical_coverage', g.get('coverage', None))
        qhat  = g.get('q_hat', None)
        strat = g.get('strategy', '?')
        cov_s  = f'{cov*100:.1f}%' if cov is not None else '?'
        qhat_s = f'{qhat:.2f}' if qhat is not None else '?'
        if cov is not None:
            diff = (cov - 0.80) * 100
            diff_s = f'+{diff:.1f}%' if diff >= 0 else f'{diff:.1f}%'
        else:
            diff_s = '?'
        out.append(f'| {grp} | {strat} | {qhat_s} | {cov_s} | {diff_s} |')
overall_cov = conf.get('overall_empirical_coverage', None)
if overall_cov:
    diff_o = (overall_cov - 0.80) * 100
    out.append(f'| Overall | mixed | -- | {overall_cov*100:.2f}% | +{diff_o:.2f}% |')
out.append('')

# T9
out.append('# Table T9: Distribution Shift (PSI) -- Drift Report\n')
out.append('| Feature | PSI | Status | Train mean | Test mean |')
out.append('|---------|-----|--------|-----------|----------|')
all_feat = drift.get('features', drift.get('feature_drift', {}))
if isinstance(all_feat, dict):
    for fname, info in sorted(all_feat.items(), key=lambda x: -x[1].get('psi', 0)):
        psi    = info.get('psi', None)
        status = info.get('status', '?')
        tmean  = info.get('train_mean', None)
        temean = info.get('test_mean', None)
        psi_s  = f'{psi:.3f}' if psi is not None else '?'
        tm_s   = f'{tmean:.3f}' if tmean is not None else '?'
        te_s   = f'{temean:.3f}' if temean is not None else '?'
        out.append(f'| {fname} | {psi_s} | {status} | {tm_s} | {te_s} |')
elif isinstance(all_feat, list):
    for info in sorted(all_feat, key=lambda x: -x.get('psi', 0)):
        fname  = info.get('feature', info.get('name', '?'))
        psi    = info.get('psi', None)
        status = info.get('status', '?')
        tmean  = info.get('train_mean', None)
        temean = info.get('test_mean', None)
        psi_s  = f'{psi:.3f}' if psi is not None else '?'
        tm_s   = f'{tmean:.3f}' if tmean is not None else '--'
        te_s   = f'{temean:.3f}' if temean is not None else '--'
        out.append(f'| {fname} | {psi_s} | {status} | {tm_s} | {te_s} |')
else:
    print('drift keys:', list(drift.keys()))

Path('paperprep/tables.md').write_text('\n'.join(out), encoding='utf-8')
print('Written paperprep/tables.md (T3, T4, T5, T6, T7, T9)')

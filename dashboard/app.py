import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Battery AI Co-Scientist", page_icon="🔋", layout="wide")
st.title("🔋 Battery AI Co-Scientist Dashboard")

BASE_DIR = Path('.').resolve()
DATA_PATH = BASE_DIR / 'data/processed/cycle_features_with_rul.csv'
MODEL_DIR = BASE_DIR / 'data/processed/modeling'

ANOMALY_PATH     = MODEL_DIR / 'anomalies.json'
UNCERTAINTY_PATH = MODEL_DIR / 'uncertainty_estimates.json'
SURVIVAL_PATH    = MODEL_DIR / 'survival_risk_predictions.csv'
REPORT_PATH      = MODEL_DIR / 'final_system_report.md'


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


df         = load_csv(DATA_PATH)
anomalies  = load_json(ANOMALY_PATH) or []
uncertainty = load_json(UNCERTAINTY_PATH) or []
survival   = load_csv(SURVIVAL_PATH)

st.caption('Read-only decision-support dashboard. No automated battery control is performed.')

# -- Artifact status ---------------------------------------------------------
status_rows = [
    {'artifact': 'cycle_features_with_rul.csv',    'exists': DATA_PATH.exists()},
    {'artifact': 'anomalies.json',                 'exists': ANOMALY_PATH.exists()},
    {'artifact': 'uncertainty_estimates.json',     'exists': UNCERTAINTY_PATH.exists()},
    {'artifact': 'survival_risk_predictions.csv',  'exists': SURVIVAL_PATH.exists()},
    {'artifact': 'final_system_report.md',         'exists': REPORT_PATH.exists()},
]

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric('Cycles',           int(len(df))         if not df.empty         else 0)
col_b.metric('Anomalies',        int(len(anomalies)))
col_c.metric('Uncertainty rows', int(len(uncertainty)))
col_d.metric('Survival rows',    int(len(survival)))

st.subheader('Artifact Status')
st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

if df.empty:
    st.error('Missing processed cycle dataset. Run preprocessing first.')
    st.stop()

u_df = pd.DataFrame(uncertainty)
a_df = pd.DataFrame(anomalies)

if not a_df.empty:
    if "battery_id" in a_df.columns:
        a_df["battery_id"] = a_df["battery_id"].astype(str)
    if "cycle_index" in a_df.columns:
        a_df["cycle_index"] = pd.to_numeric(a_df["cycle_index"], errors="coerce")

# -- Resolve RUL column name robustly ----------------------------------------
# Pipeline writes 'rul_ensemble_mean'; older runs may use 'rul_median' or
# 'rul_ensemble'.  Accept whichever is present.
RUL_COL_CANDIDATES = ['rul_ensemble_mean', 'rul_median', 'rul_ensemble',
                       'rul_ml', 'rul_pred']
rul_col = next(
    (c for c in RUL_COL_CANDIDATES if not u_df.empty and c in u_df.columns),
    None,
)
has_uncertainty = rul_col is not None
has_rul_bands   = has_uncertainty and {'rul_lower_5', 'rul_upper_95'}.issubset(u_df.columns)

battery_options = sorted(df['battery_id'].dropna().astype(str).unique())

# -- Battery selector --------------------------------------------------------
selected_battery = st.selectbox('Battery ID (single-battery views)', battery_options, index=0)

all_ids = battery_options
id_sets = [set(all_ids)]
if has_uncertainty and 'battery_id' in u_df.columns:
    id_sets.append(set(u_df['battery_id'].dropna().astype(str)))
if not survival.empty and 'battery_id' in survival.columns:
    id_sets.append(set(survival['battery_id'].dropna().astype(str)))
shared_ids = sorted(set.intersection(*id_sets)) if id_sets else []

st.caption(
    f'Battery coverage: total={len(all_ids)} · '
    f'uncertainty={len(set(u_df["battery_id"].dropna().astype(str))) if has_uncertainty and "battery_id" in u_df.columns else 0} · '
    f'survival={len(set(survival["battery_id"].dropna().astype(str))) if not survival.empty and "battery_id" in survival.columns else 0} · '
    f'intersection={len(shared_ids)}'
)

df_b = df[df['battery_id'].astype(str) == selected_battery].sort_values('cycle_index')

# -- Multi-battery comparison -------------------------------------------------
st.subheader('Battery Comparison (Multi-select)')

compare_ids = st.multiselect(
    'Compare batteries side-by-side',
    options=battery_options,
    default=battery_options,   # all batteries selected by default
    key='compare_ids_v2',
)
st.caption(f'Selected for comparison: {len(compare_ids)}/{len(battery_options)} batteries')

if compare_ids:

    # -- Shared colour palette ---------------------------------------------
    # Use a qualitative palette that stays legible at 34 lines.
    # Cycle through tab20 + tab20b to get up to 40 distinct colours.
    _cmap1 = [cm.tab20(i)  for i in range(20)]
    _cmap2 = [cm.tab20b(i) for i in range(20)]
    _palette = _cmap1 + _cmap2
    colour_map = {bid: _palette[i % len(_palette)] for i, bid in enumerate(battery_options)}

    max_cycle_selected = 0
    for bid in compare_ids:
        d_max = pd.to_numeric(
            df[df['battery_id'].astype(str) == str(bid)]['cycle_index'],
            errors='coerce',
        ).max()
        if pd.notna(d_max):
            max_cycle_selected = max(max_cycle_selected, int(d_max))

    # -- Capacity degradation overlay --------------------------------------
    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 5))
    plotted_capacity = []
    total_anomaly_points = 0
    for bid in compare_ids:
        d = df[df['battery_id'].astype(str) == str(bid)].sort_values('cycle_index')
        if d.empty:
            continue
        d_plot = d.copy()
        d_plot["cycle_index"] = pd.to_numeric(d_plot["cycle_index"], errors="coerce")
        d_plot["capacity"] = pd.to_numeric(d_plot["capacity"], errors="coerce")
        d_plot = d_plot.dropna(subset=["cycle_index", "capacity"])
        if d_plot.empty:
            continue
        ax_cmp.plot(d_plot['cycle_index'], d_plot['capacity'],
                    linewidth=1.5, color=colour_map[bid], label=str(bid))
        plotted_capacity.append(str(bid))

        if {'battery_id', 'cycle_index'}.issubset(a_df.columns):
            an_bid = a_df[a_df["battery_id"] == str(bid)].copy()
            if not an_bid.empty:
                an_cycles = set(pd.to_numeric(an_bid["cycle_index"], errors="coerce").dropna().astype(int).tolist())
                d_plot["cycle_int"] = d_plot["cycle_index"].astype(int)
                marked = d_plot[d_plot["cycle_int"].isin(an_cycles)]
                if not marked.empty:
                    total_anomaly_points += int(len(marked))
                    ax_cmp.scatter(
                        marked["cycle_index"],
                        marked["capacity"],
                        color="red",
                        marker="o",
                        s=16,
                        alpha=0.9,
                        zorder=5,
                    )

    ax_cmp.set_xlabel('Cycle')
    ax_cmp.set_ylabel('Capacity (Ahr)')
    ax_cmp.set_title('Degradation Curve Overlay — All Selected Batteries')
    ax_cmp.grid(True, alpha=0.3)
    if len(compare_ids) <= 20:
        ax_cmp.legend(loc='upper right', fontsize=7, ncol=2)
    else:
        ax_cmp.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.7)
    st.pyplot(fig_cmp)

    missing_capacity = [b for b in compare_ids if str(b) not in plotted_capacity]
    st.caption(
        f'Capacity overlay: {len(plotted_capacity)}/{len(compare_ids)} batteries plotted'
        + (f' · missing: {", ".join(missing_capacity)}' if missing_capacity else '')
    )
    if {'battery_id', 'cycle_index'}.issubset(a_df.columns):
        st.caption(f'Anomaly markers plotted on overlay: {total_anomaly_points}')

    # -- RUL trajectory overlay --------------------------------------------
    st.caption(
        f'RUL source: **{rul_col}** from uncertainty_estimates.json where available, '
        f'otherwise raw RUL labels from cycle_features_with_rul.csv.'
        if has_uncertainty else
        'uncertainty_estimates.json not found — showing raw RUL labels only.'
    )

    show_rul_bands = st.checkbox(
        'Show RUL uncertainty bands (5%–95%)',
        value=False,           # off by default when many batteries are shown
        key='comparison_rul_bands',
        disabled=not has_rul_bands,
        help='Bands are clearest when ≤ 6 batteries are selected.',
    )
    if not has_rul_bands and has_uncertainty:
        st.caption('rul_lower_5 / rul_upper_95 columns not found in uncertainty_estimates.json.')

    fig_rul, ax_rul = plt.subplots(figsize=(12, 5))
    plotted_rul_ids   = []
    fallback_rul_ids  = []
    missing_rul_ids   = []

    for bid in compare_ids:
        col  = colour_map[bid]
        used_uncertainty = False

        # -- Try uncertainty estimates first -------------------------------
        if has_uncertainty and 'battery_id' in u_df.columns:
            u_b = u_df[u_df['battery_id'].astype(str) == str(bid)].sort_values('cycle_index')
            if not u_b.empty:
                x = pd.to_numeric(u_b['cycle_index'], errors='coerce')
                y = pd.to_numeric(u_b[rul_col],       errors='coerce')
                valid = x.notna() & y.notna()
                if valid.any():
                    ax_rul.plot(x[valid], y[valid],
                                linewidth=1.8, color=col, label=str(bid))
                    used_uncertainty = True
                    plotted_rul_ids.append(str(bid))

                    if show_rul_bands and has_rul_bands:
                        lo = pd.to_numeric(u_b['rul_lower_5'],  errors='coerce')
                        hi = pd.to_numeric(u_b['rul_upper_95'], errors='coerce')
                        vb = x.notna() & lo.notna() & hi.notna()
                        if vb.any():
                            ax_rul.fill_between(x[vb], lo[vb], hi[vb],
                                                color=col, alpha=0.12, linewidth=0)

        # -- Fallback: raw RUL labels --------------------------------------
        if not used_uncertainty:
            d_b = df[df['battery_id'].astype(str) == str(bid)].sort_values('cycle_index')
            rul_label_col = 'RUL' if 'RUL' in d_b.columns else (
                'rul' if 'rul' in d_b.columns else None
            )
            if rul_label_col and not d_b.empty:
                x = pd.to_numeric(d_b['cycle_index'],      errors='coerce')
                y = pd.to_numeric(d_b[rul_label_col],      errors='coerce')
                valid = x.notna() & y.notna()
                if valid.any():
                    ax_rul.plot(x[valid], y[valid],
                                linewidth=1.5, color=col, linestyle='--',
                                alpha=0.85, label=f'{bid} (label)')
                    fallback_rul_ids.append(str(bid))
                else:
                    missing_rul_ids.append(str(bid))
            else:
                missing_rul_ids.append(str(bid))

    n_plotted = len(plotted_rul_ids) + len(fallback_rul_ids)
    if n_plotted > 0:
        ax_rul.set_xlabel('Cycle')
        ax_rul.set_ylabel('Predicted RUL (cycles)')
        ax_rul.set_title('RUL Curve Overlay — All Selected Batteries')
        ax_rul.grid(True, alpha=0.3)
        ax_rul.set_ylim(bottom=0)

        # Legend: only show when count is manageable
        if len(compare_ids) <= 20:
            ax_rul.legend(loc='upper right', fontsize=7, ncol=2)
        else:
            ax_rul.legend(loc='upper right', fontsize=6, ncol=3, framealpha=0.7)

        # Annotation for solid vs dashed
        if fallback_rul_ids:
            ax_rul.plot([], [], 'k-',  linewidth=1.5, label='predicted (uncertainty)')
            ax_rul.plot([], [], 'k--', linewidth=1.5, alpha=0.7, label='raw label (train set)')
    else:
        ax_rul.text(0.1, 0.5,
                    'No RUL data found for selected batteries.',
                    transform=ax_rul.transAxes, fontsize=12)

    st.pyplot(fig_rul)

    caption_parts = [f'Predicted (uncertainty): {len(plotted_rul_ids)}']
    if fallback_rul_ids:
        caption_parts.append(f'Raw label fallback (dashed): {len(fallback_rul_ids)}')
    if missing_rul_ids:
        caption_parts.append(f'Missing: {len(missing_rul_ids)} ({", ".join(missing_rul_ids[:6])}{"…" if len(missing_rul_ids) > 6 else ""})')
    st.caption(' · '.join(caption_parts))

    # -- Comparison table --------------------------------------------------
    rows = []
    for bid in compare_ids:
        d = df[df['battery_id'].astype(str) == str(bid)].sort_values('cycle_index')
        if d.empty:
            continue
        last_d = d.iloc[-1]

        latest_u = pd.DataFrame()
        if has_uncertainty and 'battery_id' in u_df.columns:
            latest_u = (
                u_df[u_df['battery_id'].astype(str) == str(bid)]
                .sort_values('cycle_index')
                .tail(1)
            )

        if not latest_u.empty and rul_col:
            u_row      = latest_u.iloc[0]
            rul_latest = float(u_row[rul_col])            if rul_col in latest_u.columns               else float('nan')
            risk_score = float(u_row['failure_probability']) if 'failure_probability' in latest_u.columns else float('nan')
            risk_cat   = str(u_row['risk_category'])         if 'risk_category'       in latest_u.columns else 'UNKNOWN'
            interval_w = (
                float(u_row['rul_upper_95']) - float(u_row['rul_lower_5'])
                if has_rul_bands else float('nan')
            )
        else:
            rul_label_col = 'RUL' if 'RUL' in d.columns else ('rul' if 'rul' in d.columns else None)
            rul_latest = float(last_d[rul_label_col]) if rul_label_col else float('nan')
            risk_score = float('nan')
            risk_cat   = 'N/A (train)'
            interval_w = float('nan')

        rows.append({
            'battery_id':     str(bid),
            'n_cycles':       int(len(d)),
            'latest_cycle':   int(last_d['cycle_index'])  if 'cycle_index' in d.columns  else None,
            'latest_cap':     round(float(last_d['capacity']), 3)  if 'capacity'  in d.columns else float('nan'),
            'latest_rul_pred':round(rul_latest, 1),
            'fail_prob':      round(risk_score, 3),
            'risk_category':  risk_cat,
            'interval_width': round(interval_w, 1),
            'temp_mean':      round(float(last_d['temp_mean']), 1) if 'temp_mean' in d.columns else float('nan'),
            'v_mean':         round(float(last_d['v_mean']),    3) if 'v_mean'    in d.columns else float('nan'),
            'i_mean':         round(float(last_d['i_mean']),    3) if 'i_mean'    in d.columns else float('nan'),
        })

    cmp_df = pd.DataFrame(rows)
    if not cmp_df.empty:
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
    else:
        st.info('No comparable rows found for selected batteries.')

else:
    st.info('Select at least one battery to enable comparison view.')


# -- Single-battery views -----------------------------------------------------
st.subheader('Degradation Trajectory + Anomalies')
fig1, ax1 = plt.subplots(figsize=(10, 4))
df_b_plot = df_b.copy()
df_b_plot["cycle_index"] = pd.to_numeric(df_b_plot["cycle_index"], errors="coerce")
df_b_plot["capacity"] = pd.to_numeric(df_b_plot["capacity"], errors="coerce")
df_b_plot = df_b_plot.dropna(subset=["cycle_index", "capacity"])
ax1.plot(df_b_plot['cycle_index'], df_b_plot['capacity'], label='Capacity')
single_anomaly_count = 0
if {'battery_id', 'cycle_index'}.issubset(a_df.columns):
    an_b = a_df[a_df['battery_id'].astype(str) == selected_battery].copy()
    if not an_b.empty:
        an_b["cycle_index"] = pd.to_numeric(an_b["cycle_index"], errors="coerce")
        merged = df_b_plot.merge(an_b[['cycle_index']].drop_duplicates(), on='cycle_index', how='inner')
        ax1.scatter(merged['cycle_index'], merged['capacity'],
                    color='red', label='Anomalies', s=20, zorder=3)
        single_anomaly_count = int(len(merged))
ax1.set_xlabel('Cycle')
ax1.set_ylabel('Capacity')
ax1.set_title(f'Degradation — Battery {selected_battery}')
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)
st.caption(f'Anomaly points for {selected_battery}: {single_anomaly_count}')

st.subheader('Uncertainty')
fig2, ax2 = plt.subplots(figsize=(10, 4))
if has_uncertainty and 'battery_id' in u_df.columns:
    u_b = u_df[u_df['battery_id'].astype(str) == selected_battery].sort_values('cycle_index')
    if not u_b.empty and rul_col:
        ax2.plot(u_b['cycle_index'], pd.to_numeric(u_b[rul_col], errors='coerce'),
                 label=f'RUL ({rul_col})')
        if has_rul_bands:
            ax2.fill_between(
                u_b['cycle_index'],
                pd.to_numeric(u_b['rul_lower_5'],  errors='coerce'),
                pd.to_numeric(u_b['rul_upper_95'], errors='coerce'),
                alpha=0.3, label='90% Conformal Interval',
            )
    else:
        ax2.text(0.1, 0.5, 'No uncertainty rows for selected battery',
                 transform=ax2.transAxes)
else:
    ax2.text(0.1, 0.5, 'uncertainty_estimates.json not loaded',
             transform=ax2.transAxes)
ax2.set_xlabel('Cycle')
ax2.set_ylabel('RUL')
ax2.set_title(f'RUL Uncertainty — Battery {selected_battery}')
ax2.grid(True)
ax2.legend(loc='best')
st.pyplot(fig2)

st.subheader('Survival / Hazard Risk')
fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
if {'battery_id', 'cycle_index', 'hazard_prob', 'failure_prob_horizon'}.issubset(survival.columns):
    s_b = survival[survival['battery_id'].astype(str) == selected_battery].sort_values('cycle_index')
    if not s_b.empty:
        ax3.plot(s_b['cycle_index'], s_b['hazard_prob'],
                 color='tab:orange', label='Hazard Probability')
        ax4.plot(s_b['cycle_index'], s_b['failure_prob_horizon'],
                 color='tab:red', label='Failure Probability (Horizon)')
    else:
        ax3.text(0.1, 0.5, 'No survival rows for selected battery', transform=ax3.transAxes)
else:
    ax3.text(0.1, 0.5, 'Survival schema missing required columns', transform=ax3.transAxes)
ax3.set_ylabel('Hazard')
ax3.set_title(f'Survival Risk — Battery {selected_battery}')
ax3.grid(True)
ax3.legend(loc='best')
ax4.set_xlabel('Cycle')
ax4.set_ylabel('Failure Prob')
ax4.grid(True)
ax4.legend(loc='best')
st.pyplot(fig3)

st.subheader('Final System Report')
if REPORT_PATH.exists():
    st.markdown(REPORT_PATH.read_text(encoding='utf-8'))
else:
    st.warning('final_system_report.md not found in modeling output directory.')

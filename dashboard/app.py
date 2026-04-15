import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Battery AI Co-Scientist",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path(".").resolve()
MDL  = BASE / "data/processed/modeling"
TM   = BASE / "trained_models"

@st.cache_data
def load_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

@st.cache_data
def load_csv(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

# ── load artifacts ────────────────────────────────────────────────────────────
df          = load_csv(BASE / "data/processed/cycle_features_with_rul.csv")
uncertainty = load_json(MDL / "uncertainty_estimates.json") or []
anomalies   = load_json(MDL / "anomalies.json") or []
survival    = load_csv(MDL / "survival_risk_predictions.csv")
metrics     = load_json(TM  / "model_metrics.json") or {}
conformal   = load_json(MDL / "conformal_coverage_report.json") or {}
feat_imp    = load_json(MDL / "feature_importance.json") or []
hypotheses  = load_json(MDL / "degradation_hypotheses.json") or []
counterfact = load_json(MDL / "counterfactual_examples.json") or []
unc_metrics = load_json(MDL / "uncertainty_metrics.json") or {}
cv_report   = load_json(MDL / "groupkfold_cv_report.json") or {}
manifest    = load_json(MDL / "manifest.json") or {}
report_path = MDL / "final_system_report.md"

u_df = pd.DataFrame(uncertainty)
a_df = pd.DataFrame(anomalies)

RUL_COL   = next((c for c in ["rul_ensemble_mean","rul_median","rul_ensemble","rul_ml","rul_pred"]
                  if not u_df.empty and c in u_df.columns), None)
has_bands = RUL_COL is not None and {"rul_lower_5","rul_upper_95"}.issubset(u_df.columns)

battery_ids   = sorted(df["battery_id"].dropna().astype(str).unique()) if not df.empty else []
test_bats     = set(str(b) for b in metrics.get("split_metadata",{}).get("test_batteries",[]))
cal_bats      = set(str(b) for b in metrics.get("split_metadata",{}).get("cal_batteries",[]))

GC = {"room":"#2E8648","hot":"#C0392B","cold":"#007A8A"}

def temp_group(bid):
    n_str = "".join(filter(str.isdigit, str(bid)))
    if not n_str: return "room"
    n = int(n_str)
    if str(bid).upper().startswith("CS"): return "room"
    if 41<=n<=56: return "cold"
    if (29<=n<=32) or (38<=n<=40): return "hot"
    return "room"

def role(bid):
    if bid in test_bats: return "TEST","#2E8648"
    if bid in cal_bats:  return "CALIBRATION","#007A8A"
    return "TRAINING","#888888"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔋 Battery AI Co-Scientist")
if manifest:
    st.sidebar.caption(f"Last run: {str(manifest.get('completed_at',''))[:19]}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Select a Battery")

def bat_label(b):
    r, _ = role(b)
    tag = "★" if r=="TEST" else "◆" if r=="CALIBRATION" else "·"
    return f"{tag} {b}"

label_map   = {bat_label(b): b for b in battery_ids}
sel_label   = st.sidebar.selectbox("", list(label_map.keys()), label_visibility="collapsed")
selected    = label_map[sel_label]
grp         = temp_group(selected)
bat_role, role_color = role(selected)

grp_labels = {"room":"Room temp (~25°C)","hot":"Hot (43°C)","cold":"Cold (<10°C)"}

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='background:{GC[grp]}22;border-left:4px solid {GC[grp]};"
    f"padding:10px;border-radius:6px'>"
    f"<b style='color:{GC[grp]};font-size:1.1em'>{selected}</b><br>"
    f"<span style='color:#555'>{grp_labels[grp]}</span><br>"
    f"<span style='background:{role_color}22;color:{role_color};padding:2px 8px;"
    f"border-radius:10px;font-size:0.8em;font-weight:700'>{bat_role}</span>"
    f"</div>", unsafe_allow_html=True)

if bat_role != "TEST":
    st.sidebar.info(
        f"Full analysis (RUL prediction, uncertainty, survival risk, anomaly detection) "
        f"is only available for TEST batteries.\n\n"
        f"**Test batteries:** {', '.join(sorted(test_bats))}"
    )

st.sidebar.markdown("---")
view = st.sidebar.radio("View", ["Battery Report", "Project Summary"], label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# BATTERY REPORT — everything about the selected battery on one page
# ══════════════════════════════════════════════════════════════════════════════
if view == "Battery Report":

    st.title(f"Battery {selected} — Full Report")

    df_b = df[df["battery_id"].astype(str)==selected].sort_values("cycle_index")
    u_b  = u_df[u_df["battery_id"].astype(str)==selected].sort_values("cycle_index") \
           if RUL_COL and not u_df.empty and "battery_id" in u_df.columns else pd.DataFrame()
    a_b  = a_df[a_df["battery_id"].astype(str)==selected] \
           if not a_df.empty and "battery_id" in a_df.columns else pd.DataFrame()
    s_b  = survival[survival["battery_id"].astype(str)==selected].sort_values("cycle_index") \
           if not survival.empty and "battery_id" in survival.columns else pd.DataFrame()

    eol_thr = float(df_b["eol_capacity_threshold"].iloc[0]) \
              if "eol_capacity_threshold" in df_b.columns and len(df_b) else None
    eol_cyc = int(df_b["eol_cycle"].iloc[0]) \
              if "eol_cycle" in df_b.columns and len(df_b) else None
    init_cap = float(df_b["init_capacity"].iloc[0]) \
               if "init_capacity" in df_b.columns and len(df_b) else None
    final_cap = float(df_b["capacity"].iloc[-1]) if len(df_b) else None
    n_anom    = len(a_b)

    # Latest RUL from uncertainty estimates
    latest_rul = None
    latest_risk = None
    if not u_b.empty and RUL_COL:
        last_u = u_b.iloc[-1]
        latest_rul  = round(float(last_u[RUL_COL]))
        latest_risk = str(last_u.get("risk_category","—")) if "risk_category" in last_u else "—"

    # ── STAT CARDS ────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total cycles",    len(df_b))
    c2.metric("Starting capacity", f"{init_cap:.3f} Ah"  if init_cap  else "—")
    c3.metric("Current capacity",  f"{final_cap:.3f} Ah" if final_cap else "—")
    c4.metric("EOL at cycle",    eol_cyc if eol_cyc else "—",
              help=f"First cycle where capacity dropped below {eol_thr:.3f} Ah" if eol_thr else "")
    c5.metric("Latest RUL",
              f"{latest_rul} cycles" if latest_rul is not None else ("N/A — training battery" if bat_role!="TEST" else "—"),
              help="Predicted remaining useful life at the last recorded cycle")
    c6.metric("Anomalies", n_anom if bat_role=="TEST" else "N/A")

    st.markdown("---")

    # ── WHAT IS THIS BATTERY? ─────────────────────────────────────────────────
    role_explain = {
        "TEST":        "This battery was **never shown to the model during training**. All predictions, uncertainty intervals, survival risk, and anomalies shown below are genuine — the model had no prior knowledge of this battery.",
        "CALIBRATION": "This battery was used only to **calibrate the confidence intervals** (conformal prediction). It was not used for model training.",
        "TRAINING":    "This battery was used to **train the model**. RUL predictions, survival risk, and anomaly detection are not computed for training batteries — only the raw capacity data is shown.",
    }
    st.info(role_explain[bat_role])

    # ── CHART ROW 1: capacity + RUL side by side ──────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Capacity Fade")
        st.caption("How the battery's charge-holding ability decreases over cycles. "
                   "Red dashed line = End-of-Life threshold (70% of starting capacity). "
                   "Red dots = anomalous cycles.")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(df_b["cycle_index"], df_b["capacity"],
                color=GC[grp], lw=2, label="Capacity (Ah)")
        if eol_thr:
            ax.axhline(eol_thr, color="#C0392B", lw=1.5, ls="--",
                       label=f"EOL threshold ({eol_thr:.3f} Ah)")
        if not a_b.empty and "cycle_index" in a_b.columns:
            a_b2 = a_b.copy()
            a_b2["cycle_index"] = pd.to_numeric(a_b2["cycle_index"], errors="coerce")
            merged = df_b.merge(a_b2[["cycle_index"]].drop_duplicates(), on="cycle_index", how="inner")
            if not merged.empty:
                ax.scatter(merged["cycle_index"], merged["capacity"],
                           color="red", s=50, zorder=5, label=f"Anomaly ({len(merged)})")
        if eol_cyc:
            ax.axvline(eol_cyc, color="#E68A00", lw=1.2, ls=":",
                       alpha=0.8, label=f"EOL cycle ({eol_cyc})")
        ax.set_xlabel("Cycle"); ax.set_ylabel("Capacity (Ah)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)

    with col_right:
        st.subheader("RUL Prediction")
        st.caption("Remaining Useful Life predicted by the model at each cycle. "
                   "Shaded band = 90% confidence interval (true RUL is inside this band 90% of the time). "
                   "Dashed = true RUL.")
        if bat_role == "TEST" and not u_b.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            x  = pd.to_numeric(u_b["cycle_index"], errors="coerce")
            y  = pd.to_numeric(u_b[RUL_COL], errors="coerce")
            ax2.plot(x, y, color=GC[grp], lw=2, label="Predicted RUL")
            if "RUL" in df_b.columns:
                ax2.plot(df_b["cycle_index"], df_b["RUL"],
                         color="#1F3A5F", lw=1.5, ls="--", alpha=0.7, label="True RUL")
            if has_bands:
                lo = pd.to_numeric(u_b["rul_lower_5"],  errors="coerce")
                hi = pd.to_numeric(u_b["rul_upper_95"], errors="coerce")
                ax2.fill_between(x, lo, hi, alpha=0.2, color=GC[grp],
                                 label="90% confidence band")
            ax2.axhline(0, color="#C0392B", lw=1, ls=":", alpha=0.5)
            ax2.set_xlabel("Cycle"); ax2.set_ylabel("RUL (cycles)")
            ax2.set_ylim(bottom=0); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
            ax2.spines[["top","right"]].set_visible(False)
            st.pyplot(fig2); plt.close(fig2)
        else:
            st.info("RUL prediction is only available for test batteries.")

    st.markdown("---")

    # ── SURVIVAL RISK ─────────────────────────────────────────────────────────
    st.subheader("Survival Risk — Will this battery fail soon?")
    st.caption(
        "At each cycle we ask: *what is the chance this battery fails within the next 20 cycles?* "
        "Below 30% = LOW (safe). 30–70% = MEDIUM (monitor). Above 70% = HIGH (act now)."
    )

    if bat_role == "TEST" and not s_b.empty:
        rc = s_b["risk_category"].value_counts().to_dict() if "risk_category" in s_b.columns else {}
        horizon = 20

        # Context boxes
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if eol_cyc and eol_thr:
                high_start = None
                if not s_b.empty and "failure_prob_horizon" in s_b.columns:
                    high_rows = s_b[s_b["failure_prob_horizon"] >= 0.70]
                    if not high_rows.empty:
                        high_start = int(high_rows["cycle_index"].min())

                st.markdown(
                    f"**Key facts for {selected}:**\n\n"
                    f"- EOL threshold: **{eol_thr:.3f} Ah**\n"
                    f"- Battery reached EOL at: **cycle {eol_cyc}**\n"
                    f"- Maintenance window: **{horizon} cycles**\n"
                    + (f"- HIGH risk starts at: **cycle {high_start}**"
                       if high_start else "- No HIGH risk cycles detected")
                )
        with exp_col2:
            if eol_cyc:
                high_start = None
                if not s_b.empty and "failure_prob_horizon" in s_b.columns:
                    high_rows = s_b[s_b["failure_prob_horizon"] >= 0.70]
                    if not high_rows.empty:
                        high_start = int(high_rows["cycle_index"].min())
                if high_start:
                    cycles_safe = high_start - 1
                    st.markdown(
                        f"**Why the graph looks the way it does:**\n\n"
                        f"- Cycles 1–{cycles_safe}: EOL is more than {horizon} cycles away → **0% failure prob → LOW**\n"
                        f"- From cycle {high_start}: EOL enters the {horizon}-cycle window → **HIGH risk**\n\n"
                        f"The line jumps from 0% to 100% at cycle {high_start} "
                        f"because that is exactly when EOL becomes imminent."
                    )
                else:
                    st.markdown(
                        f"**Why all cycles show 0%:**\n\n"
                        f"This battery's EOL (cycle {eol_cyc}) is always more than "
                        f"{horizon} cycles beyond any recorded cycle, so failure "
                        f"probability never rises."
                    )

        # Risk category cards
        r1, r2, r3 = st.columns(3)
        for col_, cat, color_, meaning in [
            (r1,"LOW",    "#2E8648", f"Safe — EOL is more than {horizon} cycles away"),
            (r2,"MEDIUM", "#E68A00", f"Monitor — EOL is approaching"),
            (r3,"HIGH",   "#C0392B", f"Act now — EOL is within {horizon} cycles"),
        ]:
            n = rc.get(cat, 0)
            col_.markdown(
                f"<div style='background:{color_}22;border-left:5px solid {color_};"
                f"padding:12px;border-radius:6px;text-align:center'>"
                f"<div style='color:{color_};font-weight:700'>{cat}</div>"
                f"<div style='font-size:2em;font-weight:700'>{n}</div>"
                f"<div style='color:#666;font-size:0.8em'>cycles</div>"
                f"<div style='color:#666;font-size:0.78em;margin-top:4px'>{meaning}</div>"
                f"</div>", unsafe_allow_html=True)
        st.markdown("")

        # Chart
        fig3, ax3 = plt.subplots(figsize=(12, 3.5))
        fp = s_b["failure_prob_horizon"]
        cyc = s_b["cycle_index"]
        ax3.fill_between(cyc, 0, fp, where=fp<0.30,
                         color="#2E8648", alpha=0.3, label="LOW (<30%)")
        ax3.fill_between(cyc, 0, fp, where=(fp>=0.30)&(fp<0.70),
                         color="#E68A00", alpha=0.3, label="MEDIUM (30–70%)")
        ax3.fill_between(cyc, 0, fp, where=fp>=0.70,
                         color="#C0392B", alpha=0.3, label="HIGH (>70%)")
        ax3.plot(cyc, fp, color="#C0392B", lw=2)
        ax3.axhline(0.70, color="#C0392B", lw=1.2, ls="--", alpha=0.6)
        ax3.axhline(0.30, color="#E68A00", lw=1.2, ls="--", alpha=0.6)
        ax3.text(cyc.max()*0.99, 0.72, "HIGH (70%)",   ha="right", fontsize=8, color="#C0392B")
        ax3.text(cyc.max()*0.99, 0.32, "MEDIUM (30%)", ha="right", fontsize=8, color="#E68A00")
        if eol_cyc and eol_cyc <= int(cyc.max()):
            ax3.axvline(eol_cyc, color="#E68A00", lw=1.5, ls=":", alpha=0.8)
            ax3.text(eol_cyc+2, 0.5, f"EOL\n(cycle {eol_cyc})",
                     fontsize=8, color="#E68A00", va="center")
        ax3.set_xlabel("Cycle index")
        ax3.set_ylabel(f"P(fail within {horizon} cycles)")
        ax3.set_ylim(-0.05, 1.1)
        ax3.legend(fontsize=8, loc="upper left")
        ax3.grid(alpha=0.3); ax3.spines[["top","right"]].set_visible(False)
        st.pyplot(fig3); plt.close(fig3)

    elif bat_role != "TEST":
        st.info("Survival risk is only computed for test batteries.")
    else:
        st.info("No survival data for this battery.")

    st.markdown("---")

    # ── ANOMALIES ─────────────────────────────────────────────────────────────
    st.subheader("Anomaly Detection")
    st.caption("Cycles where the battery behaved significantly differently "
               "from what the model expected — the prediction error was in the top 5%.")

    if bat_role == "TEST":
        if not a_b.empty:
            st.markdown(f"**{len(a_b)} anomalous cycles detected** for {selected}:")
            show_df = a_b[["cycle_index","anomaly_score","anomaly_type","explanation"]].copy()
            show_df["anomaly_score"] = show_df["anomaly_score"].round(2)
            show_df.columns = ["Cycle","Anomaly Score","Type","What it means"]
            st.dataframe(show_df, use_container_width=True, hide_index=True)
            st.caption(
                "Anomaly score = |true RUL − predicted RUL|. "
                "Higher score = bigger deviation from expected behaviour."
            )
        else:
            st.success(f"No anomalies detected for {selected} — degradation was smooth and predictable.")
    else:
        st.info("Anomaly detection only runs on test batteries.")

    st.markdown("---")

    # ── COUNTERFACTUALS for this battery ─────────────────────────────────────
    st.subheader("What-If Analysis (Counterfactuals)")
    st.caption("If one feature had been different, how would the predicted RUL have changed?")

    if bat_role == "TEST" and counterfact:
        bat_cf = [c for c in counterfact if str(c.get("observation_id","")).startswith(selected)]
        if bat_cf:
            cf_rows = []
            for c in bat_cf:
                cf = c.get("counterfactual",{})
                cf_rows.append({
                    "Cycle":        c.get("observation_id","").replace(selected+"_cycle_",""),
                    "True RUL":     c.get("actual_rul",""),
                    "Predicted RUL":round(float(c.get("predicted_rul",0)),1),
                    "Feature changed": cf.get("feature_changed",""),
                    "Original":     round(float(cf.get("original_value",0)),3),
                    "Hypothetical": round(float(cf.get("counterfactual_value",0)),3),
                    "RUL would change by": round(float(cf.get("predicted_rul_change",0)),2),
                })
            cdf = pd.DataFrame(cf_rows)
            def color_rul(val):
                try:
                    v = float(val)
                    return "color:#2E8648;font-weight:bold" if v>0 \
                      else "color:#C0392B;font-weight:bold" if v<0 else ""
                except: return ""
            st.dataframe(
                cdf.style.map(color_rul, subset=["RUL would change by"]),
                use_container_width=True, hide_index=True)
            st.caption("Green = would have more life. Red = would have less life.")
        else:
            st.info(f"No counterfactual examples available for {selected}.")
    elif bat_role != "TEST":
        st.info("What-if analysis only available for test batteries.")


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT SUMMARY — overall results
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("Project Summary — Battery AI Co-Scientist")

    # Verdict
    verdict_line = ""
    if report_path.exists():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if "Overall Verdict" in line:
                verdict_line = line.replace("**","").replace("Overall Verdict:","").strip()
                break
    v_color = "#2E8648" if "PASS" in verdict_line and "CONDITIONAL" not in verdict_line \
              else "#E68A00" if "CONDITIONAL" in verdict_line else "#C0392B"
    st.markdown(
        f"<div style='background:{v_color}22;border-left:6px solid {v_color};"
        f"padding:14px 20px;border-radius:6px;margin-bottom:20px'>"
        f"<span style='font-size:1.3em;font-weight:700;color:{v_color}'>"
        f"Overall Verdict: {verdict_line or '—'}</span></div>",
        unsafe_allow_html=True)

    # Key metrics
    ml_rmse   = metrics.get("rmse",0)
    dl_rmse   = metrics.get("dl_sequence",{}).get("rmse",0)
    stat_rmse = metrics.get("baseline_rmse",0)
    cov       = conformal.get("overall_empirical_coverage",0)
    risk_dist = (load_json(MDL/"uncertainty_metrics.json") or {}).get("risk_distribution",{})

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("XGBoost RMSE",   f"{ml_rmse:.2f} cycles",   "target < 100")
    c2.metric("TCN RMSE",       f"{dl_rmse:.2f} cycles")
    c3.metric("Stat Baseline",  f"{stat_rmse:.1f} cycles",  "excluded")
    c4.metric("Coverage",       f"{cov*100:.1f}%",           "target 90%")
    c5.metric("Test batteries", str(len(test_bats)))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model RMSE Comparison")
        fig, ax = plt.subplots(figsize=(6,3.5))
        mdls = ["Statistical\nBaseline","TCN\n(Deep Learning)","XGBoost\n(ML)"]
        vals = [stat_rmse, dl_rmse, ml_rmse]
        bars = ax.bar(mdls, vals, color=["#C0392B","#007A8A","#2E8648"],
                      edgecolor="white", width=0.5)
        ax.axhline(100, color="#1F3A5F", lw=2, ls="--", label="Pass threshold (100)")
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                    f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("RMSE (cycles)"); ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.subheader("Risk Distribution (all observations)")
        if risk_dist:
            fig2, ax2 = plt.subplots(figsize=(5,3.5))
            cats   = ["LOW","MEDIUM","HIGH"]
            vals2  = [risk_dist.get(c,0) for c in cats]
            colors = ["#2E8648","#E68A00","#C0392B"]
            wedges,_,autotexts = ax2.pie(
                vals2, labels=cats, colors=colors, autopct="%1.1f%%",
                startangle=140, explode=[0.02,0.02,0.05],
                textprops={"fontsize":10})
            for at in autotexts:
                at.set_fontsize(10); at.set_color("white"); at.set_fontweight("bold")
            st.pyplot(fig2); plt.close(fig2)

    st.markdown("---")

    # Feature importance
    if feat_imp:
        st.subheader("What Does the Model Rely On?")
        fi_df = pd.DataFrame(feat_imp).sort_values("importance", ascending=True)
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        colors3 = ["#1F3A5F" if v>0.10 else "#007A8A" if v>0.07 else "#AAAAAA"
                   for v in fi_df["importance"]]
        ax3.barh(fi_df["feature"], fi_df["importance"], color=colors3, edgecolor="white")
        ax3.set_xlabel("Feature Importance (XGBoost)")
        ax3.set_title("Higher = model uses this feature more")
        ax3.spines[["top","right"]].set_visible(False)
        st.pyplot(fig3); plt.close(fig3)

    st.markdown("---")

    # Conformal coverage per group
    st.subheader("Conformal Coverage by Temperature Group")
    st.caption("Did the 90% confidence intervals actually contain the true RUL 90% of the time?")
    per_grp = conformal.get("per_group",{})
    gc1, gc2, gc3 = st.columns(3)
    for col_, grp_name, glabel in [
        (gc1,"room","Room temp"),
        (gc2,"hot","Hot (43°C)"),
        (gc3,"cold","Cold (<10°C)"),
    ]:
        g    = per_grp.get(grp_name,{})
        emp  = g.get("empirical_coverage",0)
        qhat = g.get("q_hat",0)
        strat= g.get("strategy","—")
        ok   = emp >= 0.90
        c_   = GC[grp_name]
        col_.markdown(
            f"<div style='border:2px solid {c_};border-radius:8px;padding:14px;text-align:center'>"
            f"<div style='color:{c_};font-weight:700'>{glabel}</div>"
            f"<div style='font-size:2.2em;font-weight:700'>{emp*100:.1f}%</div>"
            f"<div style='color:#555;font-size:0.85em'>q_hat = {qhat:.1f} cycles</div>"
            f"<div style='color:#555;font-size:0.85em'>Strategy: {strat.upper()}</div>"
            f"<div style='color:{'#2E8648' if ok else '#C0392B'};font-weight:700;margin-top:6px'>"
            f"{'Above target' if ok else 'Below target'}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Final report
    st.subheader("Stage-by-Stage Supervisor Report")
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))

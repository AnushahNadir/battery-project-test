import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const DATA_PATHS = {
  uncertainty: "/data/processed/modeling/uncertainty_estimates.json",
  conformal: "/data/processed/modeling/conformal_coverage_report.json",
  drift: "/data/processed/modeling/drift_report.json",
  hostile: "/data/processed/hostile_validation/hostile_results.json",
  metrics: "/trained_models/model_metrics.json",
  featuresCsv: "/data/processed/cycle_features_with_rul.csv",
};

const TEMP_GROUP_COLOR = { room: "#4ade80", hot: "#a78bfa", cold: "#38bdf8" };
const PSI_COLOR = { GREEN: "#4ade80", AMBER: "#facc15", RED: "#f87171", UNKNOWN: "#94a3b8" };

const DEGRADATION_SET = new Set(["capacity", "ah_est", "energy_j"]);

function inferGroup(id) {
  const n = Number(String(id).replace(/^B0*/, ""));
  if (Number.isNaN(n)) return "room";
  if (n >= 41 && n <= 56) return "cold";
  if ((n >= 29 && n <= 32) || (n >= 38 && n <= 40)) return "hot";
  return "room";
}

function colorFor(group, i) {
  const shades = {
    room: ["#4ade80", "#22c55e", "#86efac"],
    hot: ["#a78bfa", "#8b5cf6", "#c4b5fd"],
    cold: ["#38bdf8", "#0ea5e9", "#7dd3fc"],
  };
  return (shades[group] || ["#94a3b8"])[i % 3];
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const vals = line.split(",");
    const obj = {};
    header.forEach((h, i) => {
      obj[h] = vals[i] ?? "";
    });
    return obj;
  });
}

function normalizeMetrics(m) {
  return {
    xgboost_rmse: Number(m?.rmse ?? 0),
    xgboost_mae: Number(m?.mae ?? 0),
    dl_rmse: Number(m?.dl_sequence?.rmse ?? 0),
    dl_mae: Number(m?.dl_sequence?.mae ?? 0),
    stat_rmse: Number(m?.baseline_rmse ?? 0),
    trajectory_shape_error: Number(m?.trajectory_shape_error ?? 0),
  };
}

function normalizeConformal(c) {
  const groups = Object.fromEntries(
    Object.entries(c?.per_group || {}).map(([k, v]) => [
      k,
      {
        empirical: Number(v?.empirical_coverage ?? 0),
        q_hat: Number(v?.q_hat ?? 0),
        strategy: String(v?.strategy ?? "split"),
        gap: Number(v?.gap_vs_target ?? 0),
      },
    ])
  );

  return {
    target: Number(c?.target_coverage ?? 0.9),
    overall: Number(c?.overall_empirical_coverage ?? 0),
    groups,
  };
}

function normalizeDrift(d) {
  return (d?.features || []).map((f) => ({
    feature: String(f?.feature ?? ""),
    psi: Number(f?.psi ?? 0),
    status: String(f?.status ?? "UNKNOWN"),
    train_mean: Number(f?.train_mean ?? 0),
    actual_mean: Number(f?.actual_mean ?? 0),
    shift: Number(f?.mean_shift ?? 0),
    degradation: DEGRADATION_SET.has(String(f?.feature ?? "")),
  }));
}

function normalizeHostile(h) {
  return {
    total: Number(h?.total ?? 0),
    passed: Number(h?.passed ?? 0),
    failed: Number(h?.failed ?? 0),
    pass_rate: Number(h?.pass_rate ?? 0),
    cases: (h?.cases || []).map((c) => ({
      tc: String(c?.tc_id ?? ""),
      desc: String(c?.description ?? ""),
      result: String(c?.result ?? "FAIL"),
      conf: Number(c?.confidence_mean ?? 0),
    })),
  };
}

function buildCycleData(uncertaintyRows, featureRows) {
  const capMap = new Map();
  featureRows.forEach((r) => {
    const k = `${r.battery_id}|${Number(r.cycle_index)}`;
    const c = Number(r.capacity);
    if (Number.isFinite(c)) capMap.set(k, c);
  });

  const byBat = {};
  uncertaintyRows.forEach((u) => {
    const id = String(u.battery_id);
    const cyc = Number(u.cycle_index);
    const k = `${id}|${cyc}`;
    const cap = capMap.get(k);

    const pred = Number(u.rul_ensemble ?? u.rul_median ?? 0);
    const lo = Number(u.rul_lower_5 ?? 0);
    const hi = Number(u.rul_upper_95 ?? pred);
    const fp = Number(u.failure_probability ?? 0);

    if (!byBat[id]) byBat[id] = [];
    byBat[id].push({
      cycle: cyc,
      rul_true: null,
      rul_pred: Math.round(pred),
      rul_lower: Math.max(0, Math.round(lo)),
      rul_upper: Math.round(hi),
      failure_prob: Math.max(0, Math.min(1, fp)),
      risk: String(u.risk_category ?? "LOW"),
      capacity: Number.isFinite(cap) ? Number(cap.toFixed(3)) : null,
      cap_lower: Number.isFinite(cap) ? Number((cap - 0.04).toFixed(3)) : null,
      cap_upper: Number.isFinite(cap) ? Number((cap + 0.04).toFixed(3)) : null,
    });
  });

  Object.values(byBat).forEach((arr) => arr.sort((a, b) => a.cycle - b.cycle));
  return byBat;
}

function RiskBadge({ level }) {
  const colors = {
    HIGH: "bg-red-500/20 text-red-400 border-red-500/40",
    MEDIUM: "bg-orange-500/20 text-orange-400 border-orange-500/40",
    LOW: "bg-green-500/20 text-green-400 border-green-500/40",
  };
  return (
    <span className={`px-2 py-0.5 rounded border text-xs font-mono font-bold ${colors[level] || colors.LOW}`}>
      {level}
    </span>
  );
}

function StatusDot({ status }) {
  const c = { GREEN: "bg-green-400", AMBER: "bg-amber-400", RED: "bg-red-400", UNKNOWN: "bg-slate-400" };
  return <span className={`inline-block w-2 h-2 rounded-full ${c[status] || c.UNKNOWN}`} />;
}

function MetricCard({ label, value, sub, accent }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 flex flex-col gap-1">
      <div className="text-xs text-slate-400 uppercase tracking-widest font-mono">{label}</div>
      <div className={`text-2xl font-bold font-mono ${accent || "text-slate-100"}`}>{value}</div>
      {sub && <div className="text-xs text-slate-500 font-mono">{sub}</div>}
    </div>
  );
}

function SectionHeader({ title, subtitle }) {
  return (
    <div className="flex items-baseline gap-3 mb-4">
      <h2 className="text-sm font-bold uppercase tracking-[0.2em] text-slate-300 font-mono">{title}</h2>
      {subtitle && <span className="text-xs text-slate-500 font-mono">{subtitle}</span>}
      <div className="flex-1 h-px bg-slate-700/50" />
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-600 rounded-lg p-3 text-xs font-mono shadow-xl">
      <div className="text-slate-400 mb-2">Cycle {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || p.stroke }} className="flex gap-3 justify-between">
          <span>{p.name}</span>
          <span className="font-bold">{typeof p.value === "number" ? p.value.toFixed(1) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function BatteryDashboard() {
  const [activeTab, setActiveTab] = useState("comparison");
  const [chartMode, setChartMode] = useState("rul");

  const [dataReady, setDataReady] = useState(false);
  const [loadError, setLoadError] = useState(null);

  const [BATTERIES, setBatteries] = useState({});
  const [CYCLE_DATA, setCycleData] = useState({});
  const [MODEL_METRICS, setModelMetrics] = useState({});
  const [CONFORMAL_COVERAGE, setConformalCoverage] = useState({ target: 0.9, overall: 0, groups: {} });
  const [DRIFT_FEATURES, setDriftFeatures] = useState([]);
  const [HOSTILE_RESULTS, setHostileResults] = useState({ total: 0, passed: 0, failed: 0, pass_rate: 0, cases: [] });

  const [selected, setSelected] = useState([]);

  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        const [u, c, d, h, m, csvText] = await Promise.all([
          fetch(DATA_PATHS.uncertainty).then((r) => r.json()),
          fetch(DATA_PATHS.conformal).then((r) => r.json()),
          fetch(DATA_PATHS.drift).then((r) => r.json()),
          fetch(DATA_PATHS.hostile).then((r) => r.json()),
          fetch(DATA_PATHS.metrics).then((r) => r.json()),
          fetch(DATA_PATHS.featuresCsv).then((r) => r.text()),
        ]);

        if (!alive) return;

        const featureRows = parseCsv(csvText);
        const cycleData = buildCycleData(u, featureRows);
        const batteryIds = Object.keys(cycleData);

        const batteries = Object.fromEntries(
          batteryIds.map((id, i) => {
            const group = inferGroup(id);
            return [id, { group, color: colorFor(group, i), trainTest: "test" }];
          })
        );

        setCycleData(cycleData);
        setBatteries(batteries);
        setModelMetrics(normalizeMetrics(m));
        setConformalCoverage(normalizeConformal(c));
        setDriftFeatures(normalizeDrift(d));
        setHostileResults(normalizeHostile(h));
        setSelected(batteryIds.slice(0, 3));
        setDataReady(true);
      } catch (err) {
        if (!alive) return;
        setLoadError(err?.message || "Failed to load dashboard data");
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  const toggleBattery = (id) => {
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  };

  const overlayData = useMemo(() => {
    if (!selected.length) return [];
    const maxCycles = Math.max(...selected.map((id) => CYCLE_DATA[id]?.length || 0));
    return Array.from({ length: maxCycles }, (_, i) => {
      const row = { cycle: i + 1 };
      selected.forEach((id) => {
        const d = CYCLE_DATA[id]?.[i];
        if (!d) return;
        if (chartMode === "rul") {
          row[`${id}_pred`] = d.rul_pred;
          row[`${id}_true`] = d.rul_true;
        } else {
          row[`${id}_cap`] = d.capacity;
        }
      });
      return row;
    });
  }, [CYCLE_DATA, selected, chartMode]);

  const comparisonRows = useMemo(
    () =>
      selected.map((id) => {
        const cycles = CYCLE_DATA[id] || [];
        const last = cycles[cycles.length - 1] || {};
        const meta = BATTERIES[id] || { group: "room", color: "#94a3b8" };
        const grp = CONFORMAL_COVERAGE.groups[meta.group] || {};
        return {
          id,
          ...meta,
          n_cycles: cycles.length,
          final_rul: Number(last.rul_pred ?? 0),
          final_risk: String(last.risk ?? "LOW"),
          final_fp: Number(last.failure_prob ?? 0),
          final_cap: Number(last.capacity ?? 0),
          interval_width: Number((last.rul_upper ?? 0) - (last.rul_lower ?? 0)),
          q_hat: Number(grp.q_hat ?? 0),
          coverage: Number(grp.empirical ?? 0),
        };
      }),
    [selected, BATTERIES, CYCLE_DATA, CONFORMAL_COVERAGE]
  );

  const tabs = [
    { id: "comparison", label: "Comparison" },
    { id: "risk", label: "Risk & Uncertainty" },
    { id: "drift", label: "Drift Monitor" },
    { id: "model", label: "Model QA" },
    { id: "hostile", label: "Hostile Tests" },
  ];

  if (loadError) {
    return <div className="p-6 text-red-400 font-mono">Load error: {loadError}</div>;
  }

  if (!dataReady) {
    return <div className="p-6 text-slate-400 font-mono">Loading pipeline outputs...</div>;
  }

  const driftOverall = String((CONFORMAL_COVERAGE?.overall ?? 0) >= (CONFORMAL_COVERAGE?.target ?? 0.9) ? "STABLE" : "CHECK");
  const driftStateClass = DRIFT_FEATURES.some((f) => !f.degradation && f.status === "RED") ? "text-red-400" : "text-green-400";
  const driftAlertCount = DRIFT_FEATURES.filter((f) => !f.degradation && f.status === "RED").length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100" style={{ fontFamily: "'IBM Plex Mono', 'Courier New', monospace" }}>
      <header className="border-b border-slate-800 bg-slate-900/80 sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-xs text-slate-400 uppercase tracking-widest">Battery RUL Pipeline</span>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex gap-4 text-xs">
              <span className="text-slate-400">
                XGB RMSE <span className="text-amber-400 font-bold">{Number(MODEL_METRICS.xgboost_rmse).toFixed(2)}</span>
              </span>
              <span className="text-slate-400">
                TCN RMSE <span className="text-amber-400 font-bold">{Number(MODEL_METRICS.dl_rmse).toFixed(2)}</span>
              </span>
              <span className="text-slate-400">
                Coverage <span className="text-green-400 font-bold">{(CONFORMAL_COVERAGE.overall * 100).toFixed(1)}%</span>
              </span>
            </div>
            <div className="flex items-center gap-2 bg-green-500/10 border border-green-500/30 rounded px-3 py-1">
              <StatusDot status={driftAlertCount > 0 ? "RED" : "GREEN"} />
              <span className={`text-xs ${driftStateClass}`}>Drift: {driftAlertCount > 0 ? "ALERT" : "STABLE"}</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-screen-2xl mx-auto px-6 py-6 space-y-6">
        <div>
          <SectionHeader title="Battery Selector" subtitle={`${selected.length} selected`} />
          <div className="flex flex-wrap gap-2">
            {Object.entries(BATTERIES).map(([id, meta]) => {
              const isSelected = selected.includes(id);
              const groupColor = TEMP_GROUP_COLOR[meta.group] || "#94a3b8";
              return (
                <button
                  key={id}
                  onClick={() => toggleBattery(id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-xs font-mono font-bold transition-all duration-150 ${
                    isSelected
                      ? "bg-slate-800 border-slate-500 text-slate-100 shadow-lg"
                      : "bg-slate-900 border-slate-800 text-slate-500 hover:border-slate-600"
                  }`}
                >
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: isSelected ? meta.color : "#475569" }} />
                  <span>{id}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ color: groupColor, backgroundColor: `${groupColor}22` }}>
                    {meta.group}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex gap-1 border-b border-slate-800">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`px-4 py-2 text-xs font-mono font-bold uppercase tracking-wider transition-colors ${
                activeTab === t.id ? "text-amber-400 border-b-2 border-amber-400 -mb-px" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {activeTab === "comparison" && (
          <div className="space-y-6">
            <div className="flex items-center gap-4">
              <div className="flex gap-1 bg-slate-900 border border-slate-800 rounded-lg p-1">
                {[["rul", "RUL Trajectory"], ["capacity", "Capacity Fade"]].map(([mode, label]) => (
                  <button
                    key={mode}
                    onClick={() => setChartMode(mode)}
                    className={`px-3 py-1.5 text-xs font-mono rounded transition-colors ${
                      chartMode === mode ? "bg-amber-500/20 text-amber-400 border border-amber-500/40" : "text-slate-500 hover:text-slate-300"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
              <div className="text-xs text-slate-400 mb-4 uppercase tracking-widest">
                {chartMode === "rul" ? "RUL Prediction Overlay" : "Capacity Fade Overlay"}
              </div>
              <ResponsiveContainer width="100%" height={340}>
                <LineChart data={overlayData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="cycle" stroke="#475569" tick={{ fontSize: 11, fontFamily: "monospace" }} />
                  <YAxis stroke="#475569" tick={{ fontSize: 11, fontFamily: "monospace" }} />
                  <Tooltip content={<CustomTooltip />} />
                  {selected.map((id) => {
                    const color = BATTERIES[id]?.color || "#94a3b8";
                    const key = chartMode === "rul" ? `${id}_pred` : `${id}_cap`;
                    return (
                      <Line
                        key={id}
                        type="monotone"
                        dataKey={key}
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        name={id}
                        connectNulls
                        activeDot={{ r: 4, fill: color }}
                      />
                    );
                  })}
                  {chartMode === "rul" && <ReferenceLine y={0} stroke="#f87171" strokeDasharray="6 3" strokeWidth={1} opacity={0.5} />}
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-800">
                <span className="text-xs text-slate-400 uppercase tracking-widest">Battery Comparison Table</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-800">
                      {["Battery", "Group", "Cycles", "Final RUL", "Risk", "Fail Prob", "Capacity", "Interval Width", "qhat", "Coverage"].map((h) => (
                        <th key={h} className="text-left px-4 py-2.5 text-slate-500 font-normal uppercase tracking-wider whitespace-nowrap">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonRows.map((row, i) => (
                      <tr key={row.id} className={`border-b border-slate-800/60 ${i % 2 === 0 ? "" : "bg-slate-900/20"}`}>
                        <td className="px-4 py-3 font-bold text-slate-200">{row.id}</td>
                        <td className="px-4 py-3">{row.group}</td>
                        <td className="px-4 py-3 text-slate-300">{row.n_cycles}</td>
                        <td className="px-4 py-3 font-bold text-slate-100">{row.final_rul}</td>
                        <td className="px-4 py-3">
                          <RiskBadge level={row.final_risk} />
                        </td>
                        <td className="px-4 py-3 text-slate-300">{(row.final_fp * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 text-slate-300">{Number(row.final_cap).toFixed(3)}</td>
                        <td className="px-4 py-3 text-slate-300">{Number(row.interval_width).toFixed(1)}</td>
                        <td className="px-4 py-3 text-slate-300">{Number(row.q_hat).toFixed(2)}</td>
                        <td className="px-4 py-3 text-slate-300">{(row.coverage * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === "risk" && (
          <div className="space-y-6">
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(CONFORMAL_COVERAGE.groups).map(([grp, d]) => {
                const hit = d.empirical >= CONFORMAL_COVERAGE.target;
                const color = TEMP_GROUP_COLOR[grp] || "#94a3b8";
                return (
                  <div key={grp} className="border border-slate-800 rounded-lg p-4">
                    <div className="text-xs uppercase tracking-wider mb-2" style={{ color }}>
                      {grp}
                    </div>
                    <div className={`text-3xl font-bold font-mono mb-1 ${hit ? "" : "text-red-400"}`}>{(d.empirical * 100).toFixed(1)}%</div>
                    <div className="text-[11px] text-slate-500">qhat={Number(d.q_hat).toFixed(2)} strategy={d.strategy}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === "drift" && (
          <div className="space-y-6">
            <div className="grid grid-cols-4 gap-4">
              <MetricCard label="Overall Status" value={driftAlertCount > 0 ? "ALERT" : "STABLE"} accent={driftAlertCount > 0 ? "text-red-400" : "text-green-400"} />
              <MetricCard label="Features Monitored" value={String(DRIFT_FEATURES.length)} />
              <MetricCard label="Active Alerts" value={String(driftAlertCount)} accent={driftAlertCount > 0 ? "text-red-400" : "text-green-400"} />
              <MetricCard label="Degradation Flags" value={String(DRIFT_FEATURES.filter((f) => f.degradation && f.status === "RED").length)} />
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
              <SectionHeader title="PSI by Feature" subtitle="Thresholds: amber=0.10 red=0.20" />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={DRIFT_FEATURES} layout="vertical" margin={{ top: 0, right: 24, left: 90, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" domain={[0, 0.5]} stroke="#475569" tick={{ fontSize: 11, fontFamily: "monospace" }} />
                  <YAxis type="category" dataKey="feature" stroke="#475569" tick={{ fontSize: 11, fontFamily: "monospace" }} width={88} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine x={0.1} stroke="#facc15" strokeDasharray="4 2" strokeWidth={1} opacity={0.7} />
                  <ReferenceLine x={0.2} stroke="#f87171" strokeDasharray="4 2" strokeWidth={1} opacity={0.7} />
                  <Bar dataKey="psi" name="PSI" radius={[0, 3, 3, 0]}>
                    {DRIFT_FEATURES.map((f, i) => (
                      <Cell
                        key={i}
                        fill={f.degradation ? "#94a3b8" : f.psi >= 0.2 ? "#f87171" : f.psi >= 0.1 ? "#facc15" : "#4ade80"}
                        opacity={f.degradation ? 0.5 : 1}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === "model" && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard label="XGBoost RMSE" value={Number(MODEL_METRICS.xgboost_rmse).toFixed(2)} accent="text-amber-400" sub={`MAE ${Number(MODEL_METRICS.xgboost_mae).toFixed(2)}`} />
              <MetricCard label="TCN RMSE" value={Number(MODEL_METRICS.dl_rmse).toFixed(2)} accent="text-sky-400" sub={`MAE ${Number(MODEL_METRICS.dl_mae).toFixed(2)}`} />
              <MetricCard label="Baseline RMSE" value={Number(MODEL_METRICS.stat_rmse).toFixed(2)} accent="text-red-400" />
              <MetricCard label="Shape Error" value={Number(MODEL_METRICS.trajectory_shape_error).toFixed(3)} accent="text-amber-400" />
            </div>
          </div>
        )}

        {activeTab === "hostile" && (
          <div className="space-y-6">
            <div className="grid grid-cols-4 gap-4">
              <MetricCard label="Tests Passed" value={`${HOSTILE_RESULTS.passed}/${HOSTILE_RESULTS.total}`} accent="text-green-400" />
              <MetricCard label="Failed" value={String(HOSTILE_RESULTS.failed)} accent={HOSTILE_RESULTS.failed > 0 ? "text-red-400" : "text-green-400"} />
              <MetricCard label="Pass Rate" value={`${(HOSTILE_RESULTS.pass_rate * 100).toFixed(1)}%`} />
              <MetricCard label="Cases" value={String(HOSTILE_RESULTS.cases.length)} />
            </div>
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-800 flex justify-between items-center">
                <span className="text-xs text-slate-400 uppercase tracking-widest">Hostile Test Results</span>
                <span className="text-xs text-green-400 font-mono">
                  {HOSTILE_RESULTS.passed}/{HOSTILE_RESULTS.total} PASS
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-0">
                {HOSTILE_RESULTS.cases.map((c, i) => (
                  <div key={c.tc} className={`flex items-center gap-4 px-5 py-3 border-b border-slate-800/60 ${i % 2 === 0 ? "" : "md:border-l border-slate-800/60"}`}>
                    <div className="flex items-center gap-2 w-16 flex-shrink-0">
                      <div className={`w-1.5 h-1.5 rounded-full ${c.result === "PASS" ? "bg-green-400" : "bg-red-400"}`} />
                      <span className="text-xs font-mono font-bold text-slate-400">{c.tc}</span>
                    </div>
                    <span className="text-xs text-slate-300 flex-1 truncate">{c.desc}</span>
                    <span className={`text-xs font-bold font-mono ${c.result === "PASS" ? "text-green-400" : "text-red-400"}`}>{c.result}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


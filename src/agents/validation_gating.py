# src/agents/validation_gating.py
"""
Validation & Gating Agent

Responsibilities:
- Validate mapped columns against physical plausibility
- Detect impossible ranges, unit inconsistencies, distribution mismatch
- Produce validation_report.md with explicit PASS / REVIEW / FAIL decision

HARD RULE:
- FAIL is a SUCCESS if it is justified and well explained
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhysicsCheck:
    """Result of a single physics check."""
    check_name: str
    column: str
    passed: bool
    message: str
    severity: str  # "ERROR" | "WARNING" | "INFO"
    expected: Optional[str] = None
    observed: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "check_name": self.check_name,
            "column": self.column,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass
class DistributionCheck:
    """Result of distribution comparison vs NASA reference."""
    column: str
    canonical: str
    nasa_mean: float
    nasa_std: float
    observed_mean: float
    observed_std: float
    z_score: float
    is_anomalous: bool
    message: str
    
    def to_dict(self) -> Dict:
        return {
            "column": self.column,
            "canonical": self.canonical,
            "nasa_mean": round(self.nasa_mean, 4),
            "nasa_std": round(self.nasa_std, 4),
            "observed_mean": round(self.observed_mean, 4),
            "observed_std": round(self.observed_std, 4),
            "z_score": round(self.z_score, 2),
            "is_anomalous": self.is_anomalous,
            "message": self.message,
        }


@dataclass
class ValidationResult:
    """Complete validation result."""
    dataset_type: str
    decision: str  # "PASS" | "REVIEW" | "FAIL"
    decision_reason: str
    physics_checks: List[PhysicsCheck]
    distribution_checks: List[DistributionCheck]
    error_count: int = 0
    warning_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "dataset_type": self.dataset_type,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "physics_checks": [c.to_dict() for c in self.physics_checks],
            "distribution_checks": [c.to_dict() for c in self.distribution_checks],
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
            }
        }


class ValidationGatingAgent:
    """Agent that validates mapped data against physical constraints."""
    
    def __init__(self):
        # NASA reference statistics (from actual NASA battery data)
        self.nasa_reference = {
            "voltage_measured": {"mean": 3.5, "std": 0.4, "min": 2.0, "max": 4.5},
            "voltage_load": {"mean": 2.8, "std": 0.8, "min": 0.0, "max": 4.5},
            "current_measured": {"mean": -0.5, "std": 0.8, "min": -2.5, "max": 0.5},
            "current_load": {"mean": -0.9, "std": 0.5, "min": -2.0, "max": 0.0},
            "temperature_measured": {"mean": 28.0, "std": 5.0, "min": 10.0, "max": 50.0},
            "time": {"mean": 3000.0, "std": 1500.0, "min": 0.0, "max": 10000.0},
            "capacity": {"mean": 1.5, "std": 0.3, "min": 0.5, "max": 2.5},
            "ambient_temperature": {"mean": 24.0, "std": 2.0, "min": 18.0, "max": 30.0},
            "Re": {"mean": 0.06, "std": 0.02, "min": 0.03, "max": 0.15},
            "Rct": {"mean": 0.2, "std": 0.05, "min": 0.08, "max": 0.35},
        }
        
        # Physical impossibility thresholds
        self.impossible_ranges = {
            "voltage_measured": {"min": -0.5, "max": 10.0, "unit": "V"},
            "voltage_load": {"min": -0.5, "max": 10.0, "unit": "V"},
            "current_measured": {"min": -100.0, "max": 100.0, "unit": "A"},
            "current_load": {"min": -100.0, "max": 100.0, "unit": "A"},
            "temperature_measured": {"min": -50.0, "max": 150.0, "unit": "C"},
            "time": {"min": 0.0, "max": 1e6, "unit": "s"},
            "capacity": {"min": 0.0, "max": 100.0, "unit": "Ah"},
            "ambient_temperature": {"min": -50.0, "max": 80.0, "unit": "C"},
            "Re": {"min": 0.0, "max": 10.0, "unit": "Ohm"},
            "Rct": {"min": 0.0, "max": 10.0, "unit": "Ohm"},
        }
    
    def _check_impossible_range(self, values: pd.Series, canonical: str, 
                                 col_name: str) -> PhysicsCheck:
        """Check if values are in physically impossible ranges."""
        if canonical not in self.impossible_ranges:
            return PhysicsCheck(
                check_name="impossible_range",
                column=col_name,
                passed=True,
                message=f"No range constraints defined for '{canonical}'",
                severity="INFO",
            )
        
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if len(numeric) == 0:
            return PhysicsCheck(
                check_name="impossible_range",
                column=col_name,
                passed=False,
                message="No numeric values found",
                severity="ERROR",
            )
        
        constraints = self.impossible_ranges[canonical]
        vmin, vmax = numeric.min(), numeric.max()
        
        if vmin < constraints["min"]:
            return PhysicsCheck(
                check_name="impossible_range",
                column=col_name,
                passed=False,
                message=f"Values below physical minimum for {canonical}",
                severity="ERROR",
                expected=f">= {constraints['min']} {constraints['unit']}",
                observed=f"min = {vmin:.4f}",
            )
        
        if vmax > constraints["max"]:
            return PhysicsCheck(
                check_name="impossible_range",
                column=col_name,
                passed=False,
                message=f"Values above physical maximum for {canonical}",
                severity="ERROR",
                expected=f"<= {constraints['max']} {constraints['unit']}",
                observed=f"max = {vmax:.4f}",
            )
        
        return PhysicsCheck(
            check_name="impossible_range",
            column=col_name,
            passed=True,
            message=f"Values within physical bounds for {canonical}",
            severity="INFO",
            expected=f"[{constraints['min']}, {constraints['max']}] {constraints['unit']}",
            observed=f"[{vmin:.4f}, {vmax:.4f}]",
        )
    
    def _check_unit_consistency(self, values: pd.Series, canonical: str,
                                 col_name: str) -> PhysicsCheck:
        """Check if values suggest wrong units (e.g., mV instead of V)."""
        if canonical not in self.nasa_reference:
            return PhysicsCheck(
                check_name="unit_consistency",
                column=col_name,
                passed=True,
                message="No unit reference available",
                severity="INFO",
            )
        
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if len(numeric) == 0:
            return PhysicsCheck(
                check_name="unit_consistency",
                column=col_name,
                passed=True,
                message="Non-numeric column, unit check skipped",
                severity="INFO",
            )
        
        ref = self.nasa_reference[canonical]
        obs_mean = numeric.mean()
        
        # Check for obvious unit scaling issues
        scale_factors = [1000, 1/1000, 100, 1/100]  # mV vs V, etc.
        
        for scale in scale_factors:
            scaled_mean = obs_mean * scale
            if abs(scaled_mean - ref["mean"]) < ref["std"]:
                return PhysicsCheck(
                    check_name="unit_consistency",
                    column=col_name,
                    passed=False,
                    message=f"Possible unit mismatch: values may need scaling by {scale}",
                    severity="WARNING",
                    expected=f"mean ~ {ref['mean']:.4f}",
                    observed=f"mean = {obs_mean:.4f}, scaled_mean = {scaled_mean:.4f}",
                )
        
        return PhysicsCheck(
            check_name="unit_consistency",
            column=col_name,
            passed=True,
            message="No obvious unit scaling issues",
            severity="INFO",
        )
    
    def _check_distribution_vs_nasa(self, values: pd.Series, canonical: str,
                                     col_name: str) -> Optional[DistributionCheck]:
        """Compare distribution to NASA reference."""
        if canonical not in self.nasa_reference:
            return None
        
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if len(numeric) < 5:
            return None
        
        ref = self.nasa_reference[canonical]
        obs_mean = float(numeric.mean())
        obs_std = float(numeric.std()) if len(numeric) > 1 else 0.0
        
        # Z-score of observed mean relative to NASA reference
        z_score = abs(obs_mean - ref["mean"]) / ref["std"] if ref["std"] > 0 else 0.0
        
        is_anomalous = z_score > 3.0  # More than 3 std from NASA mean
        
        if is_anomalous:
            message = f"ANOMALY: Distribution significantly differs from NASA reference (z={z_score:.1f})"
        else:
            message = f"Distribution consistent with NASA reference (z={z_score:.1f})"
        
        return DistributionCheck(
            column=col_name,
            canonical=canonical,
            nasa_mean=ref["mean"],
            nasa_std=ref["std"],
            observed_mean=obs_mean,
            observed_std=obs_std,
            z_score=z_score,
            is_anomalous=is_anomalous,
            message=message,
        )
    
    def _check_time_monotonicity(self, df: pd.DataFrame) -> Optional[PhysicsCheck]:
        """Check if time column is monotonically increasing."""
        time_cols = [c for c in df.columns if "time" in c.lower() or "elapsed" in c.lower()]
        
        if not time_cols:
            return None
        
        col = time_cols[0]
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        
        if len(numeric) < 2:
            return None
        
        is_monotonic = numeric.is_monotonic_increasing
        
        if is_monotonic:
            return PhysicsCheck(
                check_name="time_monotonicity",
                column=col,
                passed=True,
                message="Time column is monotonically increasing",
                severity="INFO",
            )
        else:
            return PhysicsCheck(
                check_name="time_monotonicity",
                column=col,
                passed=False,
                message="Time column is NOT monotonically increasing",
                severity="WARNING",
            )
    
    def _check_missing_values(self, df: pd.DataFrame, 
                               required_cols: List[str]) -> List[PhysicsCheck]:
        """Check for excessive missing values in required columns."""
        checks = []
        
        for col in required_cols:
            if col in df.columns:
                missing_ratio = df[col].isna().mean()
                
                if missing_ratio > 0.5:
                    checks.append(PhysicsCheck(
                        check_name="missing_values",
                        column=col,
                        passed=False,
                        message=f"Excessive missing values: {missing_ratio:.1%}",
                        severity="ERROR",
                        expected="< 50% missing",
                        observed=f"{missing_ratio:.1%} missing",
                    ))
                elif missing_ratio > 0.1:
                    checks.append(PhysicsCheck(
                        check_name="missing_values",
                        column=col,
                        passed=True,
                        message=f"Moderate missing values: {missing_ratio:.1%}",
                        severity="WARNING",
                        observed=f"{missing_ratio:.1%} missing",
                    ))
        
        return checks
    
    def validate(self, df: pd.DataFrame, mappings: Dict[str, str],
                 dataset_type: str) -> ValidationResult:
        """Validate a dataframe with its column mappings."""
        physics_checks = []
        distribution_checks = []
        
        # Check each mapped column
        for source_col, canonical in mappings.items():
            if source_col not in df.columns or canonical is None:
                continue
            
            # Physics range check
            range_check = self._check_impossible_range(df[source_col], canonical, source_col)
            physics_checks.append(range_check)
            
            # Unit consistency check
            unit_check = self._check_unit_consistency(df[source_col], canonical, source_col)
            physics_checks.append(unit_check)
            
            # Distribution check
            dist_check = self._check_distribution_vs_nasa(df[source_col], canonical, source_col)
            if dist_check:
                distribution_checks.append(dist_check)
        
        # Time monotonicity (for timeseries)
        if dataset_type == "timeseries":
            time_check = self._check_time_monotonicity(df)
            if time_check:
                physics_checks.append(time_check)
        
        # Missing values
        required = list(mappings.keys())
        missing_checks = self._check_missing_values(df, required)
        physics_checks.extend(missing_checks)
        
        # Count errors and warnings
        error_count = sum(1 for c in physics_checks if c.severity == "ERROR" and not c.passed)
        warning_count = sum(1 for c in physics_checks if c.severity == "WARNING" and not c.passed)
        warning_count += sum(1 for c in distribution_checks if c.is_anomalous)
        
        # Determine decision
        if error_count > 0:
            decision = "FAIL"
            decision_reason = f"FAIL: {error_count} physics error(s) detected"
        elif warning_count > 2:
            decision = "REVIEW"
            decision_reason = f"REVIEW: Multiple warnings ({warning_count}) require human inspection"
        elif warning_count > 0:
            decision = "REVIEW"
            decision_reason = f"REVIEW: {warning_count} warning(s) detected"
        else:
            decision = "PASS"
            decision_reason = "PASS: All physics and distribution checks passed"
        
        return ValidationResult(
            dataset_type=dataset_type,
            decision=decision,
            decision_reason=decision_reason,
            physics_checks=physics_checks,
            distribution_checks=distribution_checks,
            error_count=error_count,
            warning_count=warning_count,
        )


def run_validation_gating(
    metadata_path: Path,
    timeseries_path: Path,
    mapping_table: Dict,
    output_dir: Path,
) -> tuple[ValidationResult, ValidationResult]:
    """Run validation on both metadata and timeseries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = ValidationGatingAgent()
    
    # Extract mappings from mapping table
    def extract_mappings(mapping_data: Dict) -> Dict[str, str]:
        result = {}
        for m in mapping_data.get("mappings", []):
            if m["decision"] == "ACCEPTED" and m["target_canonical"]:
                result[m["source_column"]] = m["target_canonical"]
        return result
    
    meta_mappings = extract_mappings(mapping_table.get("metadata_mapping", {}))
    ts_mappings = extract_mappings(mapping_table.get("timeseries_mapping", {}))
    
    # Validate
    meta_df = pd.read_csv(metadata_path)
    meta_result = agent.validate(meta_df, meta_mappings, "metadata")
    
    ts_df = pd.read_csv(timeseries_path)
    ts_result = agent.validate(ts_df, ts_mappings, "timeseries")
    
    # Generate markdown report
    report_md = generate_validation_report_md(meta_result, ts_result)
    report_path = output_dir / "validation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    
    # Also save JSON
    json_report = {
        "metadata_validation": meta_result.to_dict(),
        "timeseries_validation": ts_result.to_dict(),
    }
    json_path = output_dir / "validation_result.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    
    return meta_result, ts_result


def generate_validation_report_md(meta: ValidationResult, ts: ValidationResult) -> str:
    """Generate markdown validation report."""
    lines = [
        "# Validation & Gating Report",
        "",
        "## Summary",
        "",
        f"| Dataset | Decision | Errors | Warnings |",
        f"|---------|----------|--------|----------|",
        f"| Metadata | **{meta.decision}** | {meta.error_count} | {meta.warning_count} |",
        f"| Timeseries | **{ts.decision}** | {ts.error_count} | {ts.warning_count} |",
        "",
        "---",
        "",
        "## Decision Rationale",
        "",
        f"### Metadata: {meta.decision}",
        f"> {meta.decision_reason}",
        "",
        f"### Timeseries: {ts.decision}",
        f"> {ts.decision_reason}",
        "",
        "---",
        "",
        "## Physics Checks",
        "",
    ]
    
    # Metadata physics checks
    lines.append("### Metadata Checks")
    lines.append("")
    for check in meta.physics_checks:
        icon = "✅" if check.passed else ("⚠️" if check.severity == "WARNING" else "❌")
        lines.append(f"- {icon} **{check.check_name}** ({check.column}): {check.message}")
        if check.expected:
            lines.append(f"  - Expected: {check.expected}")
        if check.observed:
            lines.append(f"  - Observed: {check.observed}")
    
    lines.append("")
    
    # Timeseries physics checks
    lines.append("### Timeseries Checks")
    lines.append("")
    for check in ts.physics_checks:
        icon = "✅" if check.passed else ("⚠️" if check.severity == "WARNING" else "❌")
        lines.append(f"- {icon} **{check.check_name}** ({check.column}): {check.message}")
        if check.expected:
            lines.append(f"  - Expected: {check.expected}")
        if check.observed:
            lines.append(f"  - Observed: {check.observed}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Distribution vs NASA Reference")
    lines.append("")
    
    all_dist = meta.distribution_checks + ts.distribution_checks
    if all_dist:
        lines.append("| Column | Canonical | NASA Mean | Observed Mean | Z-Score | Status |")
        lines.append("|--------|-----------|-----------|---------------|---------|--------|")
        for d in all_dist:
            status = "⚠️ ANOMALY" if d.is_anomalous else "✅ OK"
            lines.append(f"| {d.column} | {d.canonical} | {d.nasa_mean:.2f} | {d.observed_mean:.2f} | {d.z_score:.1f} | {status} |")
    else:
        lines.append("*No distribution checks performed (no accepted mappings)*")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Final Verdict")
    lines.append("")
    
    overall = "FAIL" if meta.decision == "FAIL" or ts.decision == "FAIL" else (
        "REVIEW" if meta.decision == "REVIEW" or ts.decision == "REVIEW" else "PASS"
    )
    
    lines.append(f"**Overall Decision: {overall}**")
    lines.append("")
    
    if overall == "FAIL":
        lines.append("> [!CAUTION]")
        lines.append("> This dataset has FAILED validation. Do NOT proceed with modeling.")
    elif overall == "REVIEW":
        lines.append("> [!WARNING]")
        lines.append("> This dataset requires human review before proceeding.")
    else:
        lines.append("> [!NOTE]")
        lines.append("> This dataset has passed all validation checks.")
    
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path
    import json
    
    base = Path("d:/Energy Project/code/battery-project3")
    
    # Load mapping table
    mapping_path = base / "data/processed/hostile_validation/mapping_table.json"
    if mapping_path.exists():
        mapping_table = json.loads(mapping_path.read_text())
    else:
        mapping_table = {}
    
    meta_result, ts_result = run_validation_gating(
        metadata_path=base / "data/raw/hostile/hostile_metadata.csv",
        timeseries_path=base / "data/raw/hostile/timeseries/TS001.csv",
        mapping_table=mapping_table,
        output_dir=base / "data/processed/hostile_validation",
    )
    
    logger.info("=== VALIDATION GATING REPORT ===")
    logger.info(f"Metadata: {meta_result.decision} ({meta_result.error_count} errors, {meta_result.warning_count} warnings)")
    logger.info(f"Timeseries: {ts_result.decision} ({ts_result.error_count} errors, {ts_result.warning_count} warnings)")

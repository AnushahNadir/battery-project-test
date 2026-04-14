# src/agents/schema_inspector.py
"""
Schema & Dataset Inspector Agent

Responsibilities:
- Load and inspect incoming datasets
- Compare column names against canonical NASA battery schema
- Compute per-column confidence scores with decomposed evidence
- HALT execution if required fields have confidence < LOW_THRESHOLD
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.pipeline.schema import SYNONYMS_META, SYNONYMS_TS
from src.config import get_config

logger = logging.getLogger(__name__)


# --- Thresholds ---
_CFG = get_config()
LOW_THRESHOLD = float(_CFG.schema_inspector.low_threshold)
REVIEW_THRESHOLD = float(_CFG.schema_inspector.review_threshold)
REQUIRED_EVIDENCE_SCORE = float(_CFG.schema_inspector.required_evidence_score)

# --- Canonical required fields ---
REQUIRED_META_FIELDS = ["battery_id", "type", "filename", "capacity"]
REQUIRED_TS_FIELDS = ["time", "voltage_measured", "current_measured", "temperature_measured"]


@dataclass
class EvidenceDecomposition:
    """Decomposed evidence scores for a single column mapping."""
    metadata_match: float = 0.0      # Did README/docs mention this column?
    unit_plausibility: float = 0.0   # Are the values in expected units?
    value_range: float = 0.0         # Are values in physically plausible range?
    cross_column_consistency: float = 0.0  # Does this column relate logically to others?
    
    def total_score(self) -> float:
        """Weighted average of evidence components."""
        weights = {
            "metadata_match": 0.35,
            "unit_plausibility": 0.25,
            "value_range": 0.25,
            "cross_column_consistency": 0.15,
        }
        return (
            weights["metadata_match"] * self.metadata_match +
            weights["unit_plausibility"] * self.unit_plausibility +
            weights["value_range"] * self.value_range +
            weights["cross_column_consistency"] * self.cross_column_consistency
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "metadata_match": round(self.metadata_match, 3),
            "unit_plausibility": round(self.unit_plausibility, 3),
            "value_range": round(self.value_range, 3),
            "cross_column_consistency": round(self.cross_column_consistency, 3),
            "total_score": round(self.total_score(), 3),
        }


@dataclass
class ColumnInspectionResult:
    """Inspection result for a single column."""
    column_name: str
    candidate_canonical: Optional[str] = None
    confidence: float = 0.0
    evidence: EvidenceDecomposition = field(default_factory=EvidenceDecomposition)
    is_ambiguous: bool = False
    is_rejected: bool = False
    rejection_reason: Optional[str] = None
    value_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "column_name": self.column_name,
            "candidate_canonical": self.candidate_canonical,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence.to_dict(),
            "is_ambiguous": self.is_ambiguous,
            "is_rejected": self.is_rejected,
            "rejection_reason": self.rejection_reason,
            "value_stats": self.value_stats,
        }


@dataclass
class SchemaReport:
    """Complete schema inspection report."""
    dataset_path: str
    dataset_type: str  # "metadata" or "timeseries"
    total_columns: int
    inspected_columns: List[ColumnInspectionResult]
    required_fields_status: Dict[str, str]  # field -> "FOUND" | "MISSING" | "LOW_CONFIDENCE"
    overall_confidence: float
    halt_recommended: bool
    halt_reason: Optional[str] = None
    ambiguous_fields: List[str] = field(default_factory=list)
    rejected_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "dataset_path": self.dataset_path,
            "dataset_type": self.dataset_type,
            "total_columns": self.total_columns,
            "inspected_columns": [c.to_dict() for c in self.inspected_columns],
            "required_fields_status": self.required_fields_status,
            "overall_confidence": round(self.overall_confidence, 3),
            "halt_recommended": self.halt_recommended,
            "halt_reason": self.halt_reason,
            "ambiguous_fields": self.ambiguous_fields,
            "rejected_fields": self.rejected_fields,
        }


class SchemaInspectorAgent:
    """Agent that inspects datasets and produces schema reports."""
    
    def __init__(self, readme_path: Optional[Path] = None):
        self.readme_content = ""
        if readme_path and readme_path.exists():
            self.readme_content = readme_path.read_text(encoding="utf-8", errors="ignore").lower()
        
        # Build reverse lookup: synonym -> canonical
        self.meta_lookup = self._build_reverse_lookup(SYNONYMS_META)
        self.ts_lookup = self._build_reverse_lookup(SYNONYMS_TS)
    
    def _build_reverse_lookup(self, synonyms: Dict[str, List[str]]) -> Dict[str, str]:
        """Build reverse mapping from synonym to canonical name."""
        lookup = {}
        for canonical, syns in synonyms.items():
            for syn in syns:
                lookup[syn.lower().strip()] = canonical
        return lookup
    
    def _normalize_name(self, name: str) -> str:
        """Normalize column name for matching."""
        return re.sub(r"[^a-z0-9]", "", name.lower())
    
    def _compute_metadata_match(self, col_name: str) -> float:
        """Check if column is mentioned in README with meaningful context."""
        if not self.readme_content:
            return 0.0
        
        col_lower = col_name.lower()
        
        # Direct mention
        if col_lower in self.readme_content:
            # Check for unit or description context
            context_patterns = [
                r"{}[:\s]+[a-z]+".format(re.escape(col_lower)),  # "column: description"
                r"{}.*(?:in\s+)?(?:volts?|amps?|celsius|ohms?|seconds?)".format(re.escape(col_lower)),  # units
            ]
            for pattern in context_patterns:
                if re.search(pattern, self.readme_content):
                    return 0.9
            return 0.6  # Mentioned but without context
        
        return 0.0
    
    def _compute_unit_plausibility(self, col_name: str, values: pd.Series, 
                                    canonical_candidate: Optional[str]) -> float:
        """Check if values are in expected units for the canonical field."""
        if canonical_candidate is None:
            return 0.0
        
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if len(numeric) == 0:
            return 0.0
        
        vmin, vmax = numeric.min(), numeric.max()
        vmean = numeric.mean()
        
        # Expected ranges for canonical fields
        expected_ranges = {
            "voltage_measured": (1.5, 5.0),
            "voltage_load": (0.0, 5.0),
            "current_measured": (-10.0, 10.0),
            "current_load": (-10.0, 10.0),
            "temperature_measured": (-20.0, 80.0),
            "time": (0.0, 100000.0),
            "capacity": (0.0, 10.0),
            "ambient_temperature": (-20.0, 60.0),
            "Re": (0.0, 1.0),
            "Rct": (0.0, 1.0),
        }
        
        if canonical_candidate in expected_ranges:
            exp_min, exp_max = expected_ranges[canonical_candidate]
            # Score based on how well values fit expected range
            if exp_min <= vmin and vmax <= exp_max:
                return 1.0
            elif exp_min <= vmean <= exp_max:
                return 0.6
            else:
                return 0.2
        
        return 0.5  # Unknown canonical, neutral score
    
    def _compute_value_range(self, values: pd.Series) -> Tuple[float, Dict]:
        """Compute value range plausibility and statistics."""
        stats = {}
        numeric = pd.to_numeric(values, errors="coerce")
        
        if numeric.isna().all():
            stats = {"type": "non_numeric", "missing_ratio": float(values.isna().mean())}
            return 0.3, stats
        
        non_null = numeric.dropna()
        stats = {
            "type": "numeric",
            "count": len(non_null),
            "missing_ratio": round(float(values.isna().mean()), 3),
            "min": round(float(non_null.min()), 4) if len(non_null) > 0 else None,
            "max": round(float(non_null.max()), 4) if len(non_null) > 0 else None,
            "mean": round(float(non_null.mean()), 4) if len(non_null) > 0 else None,
            "std": round(float(non_null.std()), 4) if len(non_null) > 1 else None,
        }
        
        # Check for physically implausible values
        if stats["min"] is not None:
            if stats["min"] < -1e6 or stats["max"] > 1e6:
                return 0.2, stats  # Extreme values
        
        return 0.7, stats
    
    def _compute_cross_column_consistency(self, col_name: str, df: pd.DataFrame,
                                           canonical_candidate: Optional[str]) -> float:
        """Check if column is consistent with other columns (e.g., V and I for battery)."""
        if canonical_candidate is None:
            return 0.0
        
        # Simple heuristics
        col_lower = col_name.lower()
        other_cols = [c.lower() for c in df.columns if c != col_name]
        
        # Voltage should co-occur with current
        if canonical_candidate in ["voltage_measured", "voltage_load"]:
            if any("current" in c or "i_" in c for c in other_cols):
                return 0.8
            return 0.4
        
        # Current should co-occur with voltage
        if canonical_candidate in ["current_measured", "current_load"]:
            if any("volt" in c or "v_" in c for c in other_cols):
                return 0.8
            return 0.4
        
        # Time should exist in timeseries
        if canonical_candidate == "time":
            if any("volt" in c or "current" in c for c in other_cols):
                return 0.9
            return 0.5
        
        return 0.5  # Neutral
    
    def _find_canonical_candidate(self, col_name: str, lookup: Dict[str, str]) -> Optional[str]:
        """Find a canonical candidate for the column name."""
        normalized = self._normalize_name(col_name)
        col_lower = col_name.lower().strip()
        
        # Exact match in lookup
        if col_lower in lookup:
            return lookup[col_lower]
        
        # Partial match
        for syn, canonical in lookup.items():
            if normalized in self._normalize_name(syn) or self._normalize_name(syn) in normalized:
                return canonical
        
        return None
    
    def _is_ambiguous_column(self, col_name: str, confidence: float, 
                              canonical: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Determine if a column is ambiguous and should be flagged."""
        col_lower = col_name.lower()
        
        # Explicit ambiguity markers
        ambiguous_keywords = ["mystery", "unknown", "internal", "proprietary", "ratio"]
        for keyword in ambiguous_keywords:
            if keyword in col_lower:
                return True, f"Column name contains ambiguous keyword: '{keyword}'"
        
        # Low confidence without clear canonical
        if confidence < REQUIRED_EVIDENCE_SCORE and canonical is None:
            return True, "Low confidence and no canonical match found"
        
        return False, None
    
    def _should_reject_column(self, col_name: str, evidence: EvidenceDecomposition,
                               canonical: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Determine if a column should be rejected (not mapped)."""
        col_lower = col_name.lower()
        
        # Reject if explicitly marked as internal/proprietary in README
        if self.readme_content:
            reject_patterns = [
                r"{}.*internal\s+use".format(re.escape(col_lower)),
                r"{}.*proprietary".format(re.escape(col_lower)),
                r"{}.*units?\s+unknown".format(re.escape(col_lower)),
            ]
            for pattern in reject_patterns:
                if re.search(pattern, self.readme_content):
                    return True, "README indicates column is internal/proprietary/unknown units"
        
        # Reject if evidence is insufficient
        if evidence.total_score() < LOW_THRESHOLD:
            return True, f"Insufficient evidence (score={evidence.total_score():.2f} < {LOW_THRESHOLD})"
        
        # Reject if canonical candidate doesn't match expected type
        if canonical is None and evidence.metadata_match < 0.5:
            return True, "No canonical match and insufficient metadata evidence"
        
        return False, None
    
    def inspect_dataframe(self, df: pd.DataFrame, path: str, 
                          dataset_type: str) -> SchemaReport:
        """Inspect a dataframe and produce a schema report."""
        lookup = self.meta_lookup if dataset_type == "metadata" else self.ts_lookup
        required = REQUIRED_META_FIELDS if dataset_type == "metadata" else REQUIRED_TS_FIELDS
        
        inspected = []
        ambiguous_fields = []
        rejected_fields = []
        required_status = {f: "MISSING" for f in required}
        
        for col in df.columns:
            # Find canonical candidate
            canonical = self._find_canonical_candidate(col, lookup)
            
            # Compute evidence components
            evidence = EvidenceDecomposition(
                metadata_match=self._compute_metadata_match(col),
                unit_plausibility=self._compute_unit_plausibility(col, df[col], canonical),
                value_range=self._compute_value_range(df[col])[0],
                cross_column_consistency=self._compute_cross_column_consistency(col, df, canonical),
            )
            
            _, value_stats = self._compute_value_range(df[col])
            confidence = evidence.total_score()
            
            # Check ambiguity
            is_ambiguous, ambig_reason = self._is_ambiguous_column(col, confidence, canonical)
            
            # Check rejection
            is_rejected, reject_reason = self._should_reject_column(col, evidence, canonical)
            
            result = ColumnInspectionResult(
                column_name=col,
                candidate_canonical=canonical,
                confidence=confidence,
                evidence=evidence,
                is_ambiguous=is_ambiguous,
                is_rejected=is_rejected,
                rejection_reason=reject_reason or ambig_reason,
                value_stats=value_stats,
            )
            inspected.append(result)
            
            if is_ambiguous:
                ambiguous_fields.append(col)
            if is_rejected:
                rejected_fields.append(col)
            
            # Update required field status
            if canonical in required:
                if is_rejected or confidence < LOW_THRESHOLD:
                    required_status[canonical] = "LOW_CONFIDENCE"
                else:
                    required_status[canonical] = "FOUND"
        
        # Compute overall confidence
        confidences = [r.confidence for r in inspected if not r.is_rejected]
        overall_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        # Determine if we should halt
        halt_recommended = False
        halt_reason = None
        
        missing_required = [f for f, s in required_status.items() if s == "MISSING"]
        low_conf_required = [f for f, s in required_status.items() if s == "LOW_CONFIDENCE"]
        
        if missing_required:
            halt_recommended = True
            halt_reason = f"Missing required fields: {missing_required}"
        elif low_conf_required:
            halt_recommended = True
            halt_reason = f"Required fields with low confidence: {low_conf_required}"
        elif overall_confidence < LOW_THRESHOLD:
            halt_recommended = True
            halt_reason = f"Overall confidence ({overall_confidence:.2f}) below threshold ({LOW_THRESHOLD})"
        
        return SchemaReport(
            dataset_path=path,
            dataset_type=dataset_type,
            total_columns=len(df.columns),
            inspected_columns=inspected,
            required_fields_status=required_status,
            overall_confidence=overall_confidence,
            halt_recommended=halt_recommended,
            halt_reason=halt_reason,
            ambiguous_fields=ambiguous_fields,
            rejected_fields=rejected_fields,
        )


def run_schema_inspector(
    metadata_path: Path,
    timeseries_path: Path,
    readme_path: Optional[Path],
    output_dir: Path,
) -> Tuple[SchemaReport, SchemaReport]:
    """Run schema inspection on both metadata and timeseries files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = SchemaInspectorAgent(readme_path=readme_path)
    
    # Inspect metadata
    meta_df = pd.read_csv(metadata_path)
    meta_report = agent.inspect_dataframe(meta_df, str(metadata_path), "metadata")
    
    # Inspect timeseries
    ts_df = pd.read_csv(timeseries_path)
    ts_report = agent.inspect_dataframe(ts_df, str(timeseries_path), "timeseries")
    
    # Save reports
    combined_report = {
        "metadata_report": meta_report.to_dict(),
        "timeseries_report": ts_report.to_dict(),
    }
    
    report_path = output_dir / "schema_report.json"
    report_path.write_text(json.dumps(combined_report, indent=2), encoding="utf-8")
    
    return meta_report, ts_report


if __name__ == "__main__":
    # Test run
    from pathlib import Path
    
    base = Path("d:/Energy Project/code/battery-project3")
    meta_report, ts_report = run_schema_inspector(
        metadata_path=base / "data/raw/hostile/hostile_metadata.csv",
        timeseries_path=base / "data/raw/hostile/timeseries/TS001.csv",
        readme_path=base / "data/raw/hostile/README.txt",
        output_dir=base / "data/processed/hostile_validation",
    )
    
    logger.info("=== SCHEMA INSPECTOR REPORT ===")
    logger.info(f"Metadata halt recommended: {meta_report.halt_recommended}")
    logger.info(f"Metadata halt reason: {meta_report.halt_reason}")
    logger.info(f"Timeseries halt recommended: {ts_report.halt_recommended}")
    logger.info(f"Timeseries halt reason: {ts_report.halt_reason}")

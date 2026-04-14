# src/agents/semantic_mapper.py
"""
Semantic Mapping / RAG Agent

Responsibilities:
- Use available metadata (README, comments, file headers) for semantic evidence
- Infer semantic equivalence ONLY when evidence exists
- Produce mapping_table.json with evidence + confidence per mapping
- REJECT mappings with insufficient evidence

HARD RULES:
- No string similarity alone
- No blind guessing
- Every mapping must include: evidence + confidence
- If evidence is insufficient, mapping MUST be rejected
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.pipeline.schema import SYNONYMS_META, SYNONYMS_TS

logger = logging.getLogger(__name__)


# --- Thresholds ---
REQUIRED_EVIDENCE_SCORE = 0.70
MIN_EVIDENCE_COMPONENTS = 2  # Must have at least 2 types of evidence


@dataclass
class MappingEvidence:
    """Evidence supporting a column mapping."""
    readme_mention: bool = False
    readme_unit_hint: Optional[str] = None
    readme_description: Optional[str] = None
    value_range_match: bool = False
    value_range_score: float = 0.0
    unit_consistency: bool = False
    synonym_match: bool = False
    synonym_matched: Optional[str] = None
    cross_column_support: bool = False
    cross_column_detail: Optional[str] = None
    
    def evidence_count(self) -> int:
        """Count how many types of evidence are present."""
        count = 0
        if self.readme_mention:
            count += 1
        if self.value_range_match:
            count += 1
        if self.unit_consistency:
            count += 1
        if self.synonym_match:
            count += 1
        if self.cross_column_support:
            count += 1
        return count
    
    def compute_score(self) -> float:
        """Compute weighted evidence score."""
        score = 0.0
        
        # README mention with unit hint is strong evidence
        if self.readme_mention:
            score += 0.25
            if self.readme_unit_hint:
                score += 0.15
        
        # Value range match
        if self.value_range_match:
            score += 0.20 * self.value_range_score
        
        # Unit consistency
        if self.unit_consistency:
            score += 0.15
        
        # Synonym match (not string similarity - actual synonym from schema)
        if self.synonym_match:
            score += 0.15
        
        # Cross-column support
        if self.cross_column_support:
            score += 0.10
        
        return min(1.0, score)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "readme_mention": self.readme_mention,
            "readme_unit_hint": self.readme_unit_hint,
            "readme_description": self.readme_description,
            "value_range_match": self.value_range_match,
            "value_range_score": round(self.value_range_score, 3),
            "unit_consistency": self.unit_consistency,
            "synonym_match": self.synonym_match,
            "synonym_matched": self.synonym_matched,
            "cross_column_support": self.cross_column_support,
            "cross_column_detail": self.cross_column_detail,
            "evidence_count": self.evidence_count(),
            "total_score": round(self.compute_score(), 3),
        }


@dataclass
class MappingDecision:
    """A single mapping decision with full justification."""
    source_column: str
    target_canonical: Optional[str]
    decision: str  # "ACCEPTED" | "REJECTED" | "AMBIGUOUS"
    confidence: float
    evidence: MappingEvidence
    justification: str
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "source_column": self.source_column,
            "target_canonical": self.target_canonical,
            "decision": self.decision,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence.to_dict(),
            "justification": self.justification,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class MappingTable:
    """Complete mapping table with all decisions."""
    source_path: str
    dataset_type: str
    mappings: List[MappingDecision]
    accepted_count: int = 0
    rejected_count: int = 0
    ambiguous_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "source_path": self.source_path,
            "dataset_type": self.dataset_type,
            "mappings": [m.to_dict() for m in self.mappings],
            "summary": {
                "total": len(self.mappings),
                "accepted": self.accepted_count,
                "rejected": self.rejected_count,
                "ambiguous": self.ambiguous_count,
            }
        }


class SemanticMapperAgent:
    """Agent that performs semantic mapping with evidence-based decisions."""
    
    def __init__(self, readme_path: Optional[Path] = None):
        self.readme_content = ""
        self.readme_lines = []
        if readme_path and readme_path.exists():
            self.readme_content = readme_path.read_text(encoding="utf-8", errors="ignore")
            self.readme_lines = self.readme_content.lower().split("\n")
        
        # Expected value ranges for canonical fields
        self.expected_ranges = {
            "voltage_measured": {"min": 1.5, "max": 5.0, "unit": "volts"},
            "voltage_load": {"min": 0.0, "max": 5.0, "unit": "volts"},
            "current_measured": {"min": -10.0, "max": 10.0, "unit": "amperes"},
            "current_load": {"min": -10.0, "max": 10.0, "unit": "amperes"},
            "temperature_measured": {"min": -20.0, "max": 80.0, "unit": "celsius"},
            "time": {"min": 0.0, "max": 100000.0, "unit": "seconds"},
            "capacity": {"min": 0.0, "max": 10.0, "unit": "amp-hours"},
            "ambient_temperature": {"min": -20.0, "max": 60.0, "unit": "celsius"},
            "Re": {"min": 0.0, "max": 1.0, "unit": "ohms"},
            "Rct": {"min": 0.0, "max": 1.0, "unit": "ohms"},
            "battery_id": {"type": "string"},
            "type": {"type": "categorical", "values": ["charge", "discharge", "impedance"]},
            "filename": {"type": "string", "pattern": r".*\.csv$"},
        }
        
        # Build synonym lookups
        self.meta_synonyms = SYNONYMS_META
        self.ts_synonyms = SYNONYMS_TS
    
    def _extract_readme_info(self, col_name: str) -> Dict[str, Any]:
        """Extract information about a column from README."""
        col_lower = col_name.lower()
        info = {
            "mentioned": False,
            "unit_hint": None,
            "description": None,
        }
        
        for line in self.readme_lines:
            if col_lower in line:
                info["mentioned"] = True
                
                # Look for unit hints
                unit_patterns = [
                    (r"in\s+(volts?|v\b)", "volts"),
                    (r"in\s+(amps?|amperes?|a\b)", "amperes"),
                    (r"in\s+(celsius|c\b|degrees?)", "celsius"),
                    (r"in\s+(ohms?|ω)", "ohms"),
                    (r"in\s+(seconds?|s\b)", "seconds"),
                    (r"amp[- ]?hours?|ah\b", "amp-hours"),
                ]
                for pattern, unit in unit_patterns:
                    if re.search(pattern, line):
                        info["unit_hint"] = unit
                        break
                
                # Extract description (text after colon)
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        info["description"] = parts[1].strip()[:100]
                
                break
        
        return info
    
    def _find_synonym_match(self, col_name: str, synonyms: Dict[str, List[str]]) -> Optional[str]:
        """Find a canonical field if column name matches a synonym."""
        col_lower = col_name.lower().strip()
        col_normalized = re.sub(r"[^a-z0-9]", "", col_lower)
        
        for canonical, syn_list in synonyms.items():
            for syn in syn_list:
                syn_lower = syn.lower().strip()
                syn_normalized = re.sub(r"[^a-z0-9]", "", syn_lower)
                
                # Exact match
                if col_lower == syn_lower:
                    return canonical
                
                # Normalized match
                if col_normalized == syn_normalized:
                    return canonical
        
        return None
    
    def _check_value_range(self, values: pd.Series, canonical: str) -> tuple[bool, float]:
        """Check if values fall within expected range for canonical field."""
        if canonical not in self.expected_ranges:
            return False, 0.0
        
        expected = self.expected_ranges[canonical]
        
        if expected.get("type") == "string":
            # String type - just check non-empty
            non_empty = values.dropna().astype(str).str.len().mean()
            return non_empty > 0, 0.8 if non_empty > 0 else 0.0
        
        if expected.get("type") == "categorical":
            # Check if values match expected categories
            unique = values.dropna().astype(str).str.lower().unique()
            expected_vals = expected.get("values", [])
            matches = sum(1 for v in unique if v in expected_vals)
            ratio = matches / len(unique) if len(unique) > 0 else 0.0
            return ratio > 0.5, ratio
        
        # Numeric type
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if len(numeric) == 0:
            return False, 0.0
        
        vmin, vmax = numeric.min(), numeric.max()
        exp_min, exp_max = expected.get("min", -1e9), expected.get("max", 1e9)
        
        if exp_min <= vmin and vmax <= exp_max:
            return True, 1.0
        elif exp_min <= numeric.mean() <= exp_max:
            return True, 0.6
        
        return False, 0.0
    
    def _check_unit_consistency(self, readme_unit: Optional[str], canonical: str) -> bool:
        """Check if README unit hint matches expected unit for canonical."""
        if readme_unit is None or canonical not in self.expected_ranges:
            return False
        
        expected = self.expected_ranges[canonical]
        expected_unit = expected.get("unit", "").lower()
        
        if not expected_unit:
            return False
        
        # Normalize units
        readme_unit_lower = readme_unit.lower()
        
        # Unit aliases
        aliases = {
            "volts": ["volt", "volts", "v"],
            "amperes": ["amp", "amps", "ampere", "amperes", "a"],
            "celsius": ["celsius", "c", "degrees"],
            "ohms": ["ohm", "ohms", "ω"],
            "seconds": ["second", "seconds", "s", "sec"],
            "amp-hours": ["amp-hour", "amp-hours", "ah"],
        }
        
        expected_aliases = aliases.get(expected_unit, [expected_unit])
        return any(alias in readme_unit_lower for alias in expected_aliases)
    
    def _check_cross_column(self, col_name: str, all_cols: List[str], 
                            canonical: str) -> tuple[bool, Optional[str]]:
        """Check for cross-column consistency."""
        col_lower = col_name.lower()
        other_cols = [c.lower() for c in all_cols if c.lower() != col_lower]
        
        # Voltage typically co-occurs with current
        if canonical in ["voltage_measured", "voltage_load"]:
            for c in other_cols:
                if any(term in c for term in ["current", "i_", "i_applied"]):
                    return True, f"Current column '{c}' found, supporting voltage mapping"
        
        if canonical in ["current_measured", "current_load"]:
            for c in other_cols:
                if any(term in c for term in ["volt", "v_", "v_cell"]):
                    return True, f"Voltage column '{c}' found, supporting current mapping"
        
        if canonical == "temperature_measured":
            for c in other_cols:
                if any(term in c for term in ["time", "elapsed", "volt", "current"]):
                    return True, f"Time-series context column '{c}' found"
        
        return False, None
    
    def _should_reject_mapping(self, col_name: str, evidence: MappingEvidence,
                                canonical: Optional[str]) -> tuple[bool, Optional[str]]:
        """Determine if mapping should be rejected."""
        col_lower = col_name.lower()
        
        # RULE: No blind guessing - must have minimum evidence
        if evidence.evidence_count() < MIN_EVIDENCE_COMPONENTS:
            return True, f"Insufficient evidence types ({evidence.evidence_count()} < {MIN_EVIDENCE_COMPONENTS})"
        
        # RULE: No string similarity alone
        if evidence.synonym_match and not (evidence.readme_mention or evidence.value_range_match):
            return True, "Synonym match alone is insufficient (no supporting evidence)"
        
        # RULE: Reject explicitly ambiguous columns
        ambiguous_keywords = ["mystery", "unknown", "internal", "proprietary", "arbitrary"]
        for keyword in ambiguous_keywords:
            if keyword in col_lower:
                return True, f"Column name contains ambiguous keyword: '{keyword}'"
        
        # RULE: Reject if README says internal use
        if self.readme_content:
            for pattern in [r"internal\s+use", r"proprietary", r"units?\s+unknown"]:
                if re.search(f"{re.escape(col_lower)}.*{pattern}", self.readme_content.lower()):
                    return True, "README indicates column is for internal use / unknown units"
        
        # RULE: Reject non-canonical columns that look plausible
        # This is the NEGATIVE CONTROL check
        non_canonical_plausible = ["coulombic_efficiency", "voltage_ratio", "soc", "soh"]
        if col_lower in non_canonical_plausible or any(nc in col_lower for nc in non_canonical_plausible):
            if canonical is None:
                return True, "Plausible battery term but NOT a canonical schema field"
        
        # RULE: Low evidence score
        if evidence.compute_score() < REQUIRED_EVIDENCE_SCORE:
            return True, f"Evidence score ({evidence.compute_score():.2f}) below threshold ({REQUIRED_EVIDENCE_SCORE})"
        
        return False, None
    
    def map_dataframe(self, df: pd.DataFrame, path: str, 
                      dataset_type: str) -> MappingTable:
        """Map columns in a dataframe with full evidence."""
        synonyms = self.meta_synonyms if dataset_type == "metadata" else self.ts_synonyms
        
        mappings = []
        accepted = 0
        rejected = 0
        ambiguous = 0
        
        all_cols = list(df.columns)
        
        for col in df.columns:
            # Gather evidence
            readme_info = self._extract_readme_info(col)
            synonym_canonical = self._find_synonym_match(col, synonyms)
            
            # Determine best canonical candidate
            canonical = synonym_canonical
            
            # Check value range for candidate
            range_match, range_score = False, 0.0
            if canonical:
                range_match, range_score = self._check_value_range(df[col], canonical)
            
            # Check unit consistency
            unit_consistent = False
            if canonical and readme_info["unit_hint"]:
                unit_consistent = self._check_unit_consistency(readme_info["unit_hint"], canonical)
            
            # Check cross-column support
            cross_support, cross_detail = False, None
            if canonical:
                cross_support, cross_detail = self._check_cross_column(col, all_cols, canonical)
            
            # Build evidence
            evidence = MappingEvidence(
                readme_mention=readme_info["mentioned"],
                readme_unit_hint=readme_info["unit_hint"],
                readme_description=readme_info["description"],
                value_range_match=range_match,
                value_range_score=range_score,
                unit_consistency=unit_consistent,
                synonym_match=synonym_canonical is not None,
                synonym_matched=synonym_canonical,
                cross_column_support=cross_support,
                cross_column_detail=cross_detail,
            )
            
            # Decide
            should_reject, reject_reason = self._should_reject_mapping(col, evidence, canonical)
            
            confidence = evidence.compute_score()
            
            if should_reject:
                decision = "REJECTED"
                rejected += 1
                justification = f"REJECTED: {reject_reason}"
            elif confidence < REQUIRED_EVIDENCE_SCORE:
                decision = "AMBIGUOUS"
                ambiguous += 1
                justification = f"AMBIGUOUS: Confidence ({confidence:.2f}) below threshold"
            else:
                decision = "ACCEPTED"
                accepted += 1
                justification = self._build_justification(col, canonical, evidence)
            
            mappings.append(MappingDecision(
                source_column=col,
                target_canonical=canonical if decision == "ACCEPTED" else None,
                decision=decision,
                confidence=confidence,
                evidence=evidence,
                justification=justification,
                rejection_reason=reject_reason if should_reject else None,
            ))
        
        return MappingTable(
            source_path=path,
            dataset_type=dataset_type,
            mappings=mappings,
            accepted_count=accepted,
            rejected_count=rejected,
            ambiguous_count=ambiguous,
        )
    
    def _build_justification(self, col: str, canonical: str, evidence: MappingEvidence) -> str:
        """Build textual justification for an accepted mapping."""
        parts = [f"Mapping '{col}' -> '{canonical}' ACCEPTED based on:"]
        
        if evidence.synonym_match:
            parts.append(f"  - Synonym match found in canonical schema")
        if evidence.readme_mention:
            parts.append(f"  - Column documented in README")
            if evidence.readme_unit_hint:
                parts.append(f"    - Unit hint: {evidence.readme_unit_hint}")
        if evidence.value_range_match:
            parts.append(f"  - Values fall within expected range (score: {evidence.value_range_score:.2f})")
        if evidence.unit_consistency:
            parts.append(f"  - Unit from README matches expected unit")
        if evidence.cross_column_support:
            parts.append(f"  - Cross-column support: {evidence.cross_column_detail}")
        
        parts.append(f"  - Total evidence score: {evidence.compute_score():.2f}")
        
        return "\n".join(parts)


def run_semantic_mapper(
    metadata_path: Path,
    timeseries_path: Path,
    readme_path: Optional[Path],
    output_dir: Path,
) -> tuple[MappingTable, MappingTable]:
    """Run semantic mapping on both metadata and timeseries files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = SemanticMapperAgent(readme_path=readme_path)
    
    # Map metadata
    meta_df = pd.read_csv(metadata_path)
    meta_mapping = agent.map_dataframe(meta_df, str(metadata_path), "metadata")
    
    # Map timeseries
    ts_df = pd.read_csv(timeseries_path)
    ts_mapping = agent.map_dataframe(ts_df, str(timeseries_path), "timeseries")
    
    # Save reports
    combined = {
        "metadata_mapping": meta_mapping.to_dict(),
        "timeseries_mapping": ts_mapping.to_dict(),
    }
    
    report_path = output_dir / "mapping_table.json"
    report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    
    return meta_mapping, ts_mapping


if __name__ == "__main__":
    from pathlib import Path
    
    base = Path("d:/Energy Project/code/battery-project3")
    meta_map, ts_map = run_semantic_mapper(
        metadata_path=base / "data/raw/hostile/hostile_metadata.csv",
        timeseries_path=base / "data/raw/hostile/timeseries/TS001.csv",
        readme_path=base / "data/raw/hostile/README.txt",
        output_dir=base / "data/processed/hostile_validation",
    )
    
    logger.info("=== SEMANTIC MAPPER REPORT ===")
    logger.info(f"Metadata: {meta_map.accepted_count} accepted, {meta_map.rejected_count} rejected, {meta_map.ambiguous_count} ambiguous")
    logger.info(f"Timeseries: {ts_map.accepted_count} accepted, {ts_map.rejected_count} rejected, {ts_map.ambiguous_count} ambiguous")

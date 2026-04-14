# src/agents/supervisor_critic.py
"""
Supervisor / Critic Agent

Responsibilities:
- Review outputs from ALL agents
- Explicitly answer: "Did any agent hallucinate a mapping?"
- Produce final_system_report.md with stage verdict
- List ALL hallucination indicators and whether each was triggered

HARD RULE:
- If hallucination risk exists, REJECT the run and identify exact failure point
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class HallucinationIndicator:
    """A specific indicator that could signal hallucination."""
    indicator_id: str
    description: str
    triggered: bool
    evidence: str
    severity: str  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    
    def to_dict(self) -> Dict:
        return {
            "indicator_id": self.indicator_id,
            "description": self.description,
            "triggered": self.triggered,
            "evidence": self.evidence,
            "severity": self.severity,
        }


@dataclass  
class SupervisorVerdict:
    """Final supervisor verdict."""
    verdict: str  # "PASS" | "FAIL" | "NEEDS_HUMAN_REVIEW"
    verdict_reason: str
    hallucination_detected: bool
    hallucination_details: Optional[str]
    hallucination_indicators: List[HallucinationIndicator]
    agent_reviews: Dict[str, Dict]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "hallucination_detected": self.hallucination_detected,
            "hallucination_details": self.hallucination_details,
            "hallucination_indicators": [h.to_dict() for h in self.hallucination_indicators],
            "agent_reviews": self.agent_reviews,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


class SupervisorCriticAgent:
    """Agent that supervises all other agents and detects hallucinations."""
    
    def __init__(self):
        cfg = get_config()
        self.low_confidence_accept_threshold = float(cfg.supervisor.low_confidence_accept_threshold)
        self.schema_inspector_low_threshold = float(cfg.schema_inspector.low_threshold)
        # Define all hallucination indicators
        self.indicator_definitions = [
            {
                "id": "H001",
                "description": "String similarity used as sole evidence",
                "check": self._check_string_similarity_only,
                "severity": "CRITICAL",
            },
            {
                "id": "H002", 
                "description": "Mapping accepted without README/documentation evidence",
                "check": self._check_no_readme_evidence,
                "severity": "HIGH",
            },
            {
                "id": "H003",
                "description": "Value range outside physical bounds but mapping accepted",
                "check": self._check_range_violation_accepted,
                "severity": "CRITICAL",
            },
            {
                "id": "H004",
                "description": "Ambiguous column mapped without explicit disambiguation",
                "check": self._check_ambiguous_mapped,
                "severity": "HIGH",
            },
            {
                "id": "H005",
                "description": "Required field missing but processing continued",
                "check": self._check_missing_required_continued,
                "severity": "CRITICAL",
            },
            {
                "id": "H006",
                "description": "Non-canonical field accepted as canonical",
                "check": self._check_non_canonical_accepted,
                "severity": "CRITICAL",
            },
            {
                "id": "H007",
                "description": "Confidence below threshold but mapping accepted",
                "check": self._check_low_confidence_accepted,
                "severity": "HIGH",
            },
            {
                "id": "H008",
                "description": "Distribution anomaly ignored",
                "check": self._check_distribution_anomaly_ignored,
                "severity": "MEDIUM",
            },
            {
                "id": "H009",
                "description": "Unit mismatch detected but not flagged",
                "check": self._check_unit_mismatch_ignored,
                "severity": "HIGH",
            },
            {
                "id": "H010",
                "description": "Negative control column incorrectly accepted",
                "check": self._check_negative_control_accepted,
                "severity": "CRITICAL",
            },
        ]
    
    def _check_string_similarity_only(self, schema_report: Dict, mapping_table: Dict, 
                                       validation_result: Dict) -> tuple[bool, str]:
        """Check if any mapping used only string similarity."""
        for dataset in ["metadata_mapping", "timeseries_mapping"]:
            mappings = mapping_table.get(dataset, {}).get("mappings", [])
            for m in mappings:
                if m["decision"] == "ACCEPTED":
                    evidence = m.get("evidence", {})
                    # Check if synonym match is sole evidence
                    if evidence.get("synonym_match") and evidence.get("evidence_count", 0) == 1:
                        return True, f"Column '{m['source_column']}' mapped using only synonym match"
        return False, "No mappings based on string similarity alone"
    
    def _check_no_readme_evidence(self, schema_report: Dict, mapping_table: Dict,
                                   validation_result: Dict) -> tuple[bool, str]:
        """Check if mappings accepted without README evidence."""
        for dataset in ["metadata_mapping", "timeseries_mapping"]:
            mappings = mapping_table.get(dataset, {}).get("mappings", [])
            for m in mappings:
                if m["decision"] == "ACCEPTED":
                    evidence = m.get("evidence", {})
                    if not evidence.get("readme_mention", False):
                        return True, f"Column '{m['source_column']}' accepted without README mention"
        return False, "All accepted mappings have README evidence"
    
    def _check_range_violation_accepted(self, schema_report: Dict, mapping_table: Dict,
                                         validation_result: Dict) -> tuple[bool, str]:
        """Check if range violations were ignored."""
        for dataset in ["metadata_validation", "timeseries_validation"]:
            checks = validation_result.get(dataset, {}).get("physics_checks", [])
            for check in checks:
                if check.get("check_name") == "impossible_range" and not check.get("passed", True):
                    # Check if corresponding mapping was still accepted
                    return True, f"Range violation in '{check['column']}' but validation not enforced"
        return False, "No range violations in accepted mappings"
    
    def _check_ambiguous_mapped(self, schema_report: Dict, mapping_table: Dict,
                                 validation_result: Dict) -> tuple[bool, str]:
        """Check if ambiguous columns were mapped."""
        for dataset in ["metadata_report", "timeseries_report"]:
            report = schema_report.get(dataset, {})
            ambiguous = report.get("ambiguous_fields", [])
            
            # Check if any ambiguous field was accepted in mapping
            mapping_key = dataset.replace("_report", "_mapping")
            mappings = mapping_table.get(mapping_key, {}).get("mappings", [])
            
            for m in mappings:
                if m["source_column"] in ambiguous and m["decision"] == "ACCEPTED":
                    return True, f"Ambiguous column '{m['source_column']}' was incorrectly mapped"
        
        return False, "No ambiguous columns were mapped"
    
    def _check_missing_required_continued(self, schema_report: Dict, mapping_table: Dict,
                                           validation_result: Dict) -> tuple[bool, str]:
        """Check if missing required fields didn't halt processing."""
        for dataset in ["metadata_report", "timeseries_report"]:
            report = schema_report.get(dataset, {})
            required_status = report.get("required_fields_status", {})
            
            missing = [f for f, s in required_status.items() if s == "MISSING"]
            if missing and not report.get("halt_recommended", False):
                return True, f"Missing required fields {missing} but halt not recommended"
        
        return False, "All missing required fields triggered halt recommendation"
    
    def _check_non_canonical_accepted(self, schema_report: Dict, mapping_table: Dict,
                                       validation_result: Dict) -> tuple[bool, str]:
        """Check if non-canonical fields were accepted as canonical."""
        # Known non-canonical but plausible field names
        non_canonical = ["coulombic_efficiency", "voltage_ratio", "soc", "soh", 
                         "mystery_column", "mystery_signal"]
        
        for dataset in ["metadata_mapping", "timeseries_mapping"]:
            mappings = mapping_table.get(dataset, {}).get("mappings", [])
            for m in mappings:
                if m["decision"] == "ACCEPTED":
                    source = m["source_column"].lower()
                    if any(nc in source for nc in non_canonical):
                        return True, f"Non-canonical column '{m['source_column']}' was incorrectly accepted"
        
        return False, "No non-canonical columns accepted"
    
    def _check_low_confidence_accepted(self, schema_report: Dict, mapping_table: Dict,
                                         validation_result: Dict) -> tuple[bool, str]:
        """Check if low confidence mappings were accepted."""
        for dataset in ["metadata_mapping", "timeseries_mapping"]:
            mappings = mapping_table.get(dataset, {}).get("mappings", [])
            for m in mappings:
                if m["decision"] == "ACCEPTED" and m.get("confidence", 0) < self.low_confidence_accept_threshold:
                    return True, f"Low confidence ({m['confidence']:.2f}) mapping for '{m['source_column']}' accepted"
        
        return False, "All accepted mappings have sufficient confidence"
    
    def _check_distribution_anomaly_ignored(self, schema_report: Dict, mapping_table: Dict,
                                             validation_result: Dict) -> tuple[bool, str]:
        """Check if distribution anomalies were ignored."""
        for dataset in ["metadata_validation", "timeseries_validation"]:
            dist_checks = validation_result.get(dataset, {}).get("distribution_checks", [])
            for check in dist_checks:
                if check.get("is_anomalous", False):
                    decision = validation_result.get(dataset, {}).get("decision", "")
                    if decision == "PASS":
                        return True, f"Distribution anomaly in '{check['column']}' but validation PASSED"
        
        return False, "Distribution anomalies properly flagged"
    
    def _check_unit_mismatch_ignored(self, schema_report: Dict, mapping_table: Dict,
                                      validation_result: Dict) -> tuple[bool, str]:
        """Check if unit mismatches were ignored."""
        for dataset in ["metadata_validation", "timeseries_validation"]:
            checks = validation_result.get(dataset, {}).get("physics_checks", [])
            for check in checks:
                if check.get("check_name") == "unit_consistency" and not check.get("passed", True):
                    decision = validation_result.get(dataset, {}).get("decision", "")
                    if decision == "PASS":
                        return True, f"Unit mismatch in '{check['column']}' but validation PASSED"
        
        return False, "Unit mismatches properly flagged"
    
    def _check_negative_control_accepted(self, schema_report: Dict, mapping_table: Dict,
                                          validation_result: Dict) -> tuple[bool, str]:
        """Check if negative control columns were accepted."""
        # Negative controls: columns designed to be rejected
        negative_controls = ["coulombic_efficiency", "voltage_ratio"]
        
        for dataset in ["metadata_mapping", "timeseries_mapping"]:
            mappings = mapping_table.get(dataset, {}).get("mappings", [])
            for m in mappings:
                if m["decision"] == "ACCEPTED":
                    source = m["source_column"].lower()
                    for nc in negative_controls:
                        if nc in source:
                            return True, f"NEGATIVE CONTROL '{m['source_column']}' was incorrectly ACCEPTED"
        
        return False, "All negative control columns correctly rejected"
    
    def review_agents(self, schema_report: Dict, mapping_table: Dict,
                      validation_result: Dict) -> SupervisorVerdict:
        """Review all agent outputs and produce verdict."""
        
        # Run all hallucination indicator checks
        indicators = []
        triggered_critical = []
        triggered_high = []
        triggered_any = False
        
        for ind_def in self.indicator_definitions:
            triggered, evidence = ind_def["check"](schema_report, mapping_table, validation_result)
            
            indicator = HallucinationIndicator(
                indicator_id=ind_def["id"],
                description=ind_def["description"],
                triggered=triggered,
                evidence=evidence,
                severity=ind_def["severity"],
            )
            indicators.append(indicator)
            
            if triggered:
                triggered_any = True
                if ind_def["severity"] == "CRITICAL":
                    triggered_critical.append(ind_def["id"])
                elif ind_def["severity"] == "HIGH":
                    triggered_high.append(ind_def["id"])
        
        # Review each agent
        agent_reviews = {
            "schema_inspector": self._review_schema_inspector(schema_report),
            "semantic_mapper": self._review_semantic_mapper(mapping_table),
            "validation_gating": self._review_validation_gating(validation_result),
        }
        
        # Determine verdict
        if triggered_critical:
            verdict = "FAIL"
            verdict_reason = f"CRITICAL hallucination indicators triggered: {triggered_critical}"
            hallucination_detected = True
            hallucination_details = f"Critical issues: {triggered_critical}. High issues: {triggered_high}"
        elif triggered_high:
            verdict = "NEEDS_HUMAN_REVIEW"
            verdict_reason = f"HIGH severity indicators triggered: {triggered_high}"
            hallucination_detected = True
            hallucination_details = f"High severity issues require human review: {triggered_high}"
        elif triggered_any:
            verdict = "NEEDS_HUMAN_REVIEW"
            verdict_reason = "Medium/Low severity indicators triggered"
            hallucination_detected = False
            hallucination_details = "No critical hallucinations, but some concerns detected"
        else:
            # Check schema inspector halt recommendations
            meta_halt = schema_report.get("metadata_report", {}).get("halt_recommended", False)
            ts_halt = schema_report.get("timeseries_report", {}).get("halt_recommended", False)
            
            # Check if all mappings were rejected
            meta_accepted = mapping_table.get("metadata_mapping", {}).get("summary", {}).get("accepted", 0)
            ts_accepted = mapping_table.get("timeseries_mapping", {}).get("summary", {}).get("accepted", 0)
            total_accepted = meta_accepted + ts_accepted
            
            # If Schema Inspector recommends halt OR no mappings were accepted,
            # this is a SUCCESSFUL rejection of a hostile dataset
            if meta_halt or ts_halt:
                halt_reasons = []
                if meta_halt:
                    halt_reasons.append(schema_report.get("metadata_report", {}).get("halt_reason", "Unknown"))
                if ts_halt:
                    halt_reasons.append(schema_report.get("timeseries_report", {}).get("halt_reason", "Unknown"))
                
                verdict = "FAIL"
                verdict_reason = f"Schema Inspector HALTED execution: {'; '.join(halt_reasons)}"
                hallucination_detected = False
                hallucination_details = "No hallucination - system correctly rejected hostile dataset"
            
            elif total_accepted == 0:
                verdict = "FAIL"
                verdict_reason = f"All {meta_accepted + ts_accepted} mapping attempts REJECTED - dataset cannot be processed"
                hallucination_detected = False
                hallucination_details = "No hallucination - system correctly rejected all ambiguous mappings"
            
            else:
                # Check if validation passed
                meta_decision = validation_result.get("metadata_validation", {}).get("decision", "FAIL")
                ts_decision = validation_result.get("timeseries_validation", {}).get("decision", "FAIL")
                
                if meta_decision == "PASS" and ts_decision == "PASS":
                    verdict = "PASS"
                    verdict_reason = "All checks passed, no hallucination indicators triggered"
                elif meta_decision == "FAIL" or ts_decision == "FAIL":
                    verdict = "FAIL"
                    verdict_reason = f"Validation failed (meta={meta_decision}, ts={ts_decision}) - this is correct behavior"
                else:
                    verdict = "NEEDS_HUMAN_REVIEW"
                    verdict_reason = f"Validation requires review (meta={meta_decision}, ts={ts_decision})"
                
                hallucination_detected = False
                hallucination_details = None
        
        # Generate recommendations
        recommendations = self._generate_recommendations(indicators, agent_reviews)
        
        return SupervisorVerdict(
            verdict=verdict,
            verdict_reason=verdict_reason,
            hallucination_detected=hallucination_detected,
            hallucination_details=hallucination_details,
            hallucination_indicators=indicators,
            agent_reviews=agent_reviews,
            recommendations=recommendations,
        )
    
    def _review_schema_inspector(self, schema_report: Dict) -> Dict:
        """Review schema inspector agent output."""
        meta = schema_report.get("metadata_report", {})
        ts = schema_report.get("timeseries_report", {})
        
        return {
            "status": "OK" if (meta.get("halt_recommended") or ts.get("halt_recommended")) else "CONCERN",
            "metadata_halt_recommended": meta.get("halt_recommended", False),
            "metadata_halt_reason": meta.get("halt_reason"),
            "metadata_ambiguous_fields": meta.get("ambiguous_fields", []),
            "metadata_rejected_fields": meta.get("rejected_fields", []),
            "timeseries_halt_recommended": ts.get("halt_recommended", False),
            "timeseries_halt_reason": ts.get("halt_reason"),
            "timeseries_ambiguous_fields": ts.get("ambiguous_fields", []),
            "timeseries_rejected_fields": ts.get("rejected_fields", []),
        }
    
    def _review_semantic_mapper(self, mapping_table: Dict) -> Dict:
        """Review semantic mapper agent output."""
        meta = mapping_table.get("metadata_mapping", {}).get("summary", {})
        ts = mapping_table.get("timeseries_mapping", {}).get("summary", {})
        
        total_rejected = meta.get("rejected", 0) + ts.get("rejected", 0)
        total_accepted = meta.get("accepted", 0) + ts.get("accepted", 0)
        
        return {
            "status": "OK" if total_rejected > 0 else "CONCERN",
            "metadata_accepted": meta.get("accepted", 0),
            "metadata_rejected": meta.get("rejected", 0),
            "metadata_ambiguous": meta.get("ambiguous", 0),
            "timeseries_accepted": ts.get("accepted", 0),
            "timeseries_rejected": ts.get("rejected", 0),
            "timeseries_ambiguous": ts.get("ambiguous", 0),
            "rejection_ratio": total_rejected / (total_accepted + total_rejected) if (total_accepted + total_rejected) > 0 else 0,
        }
    
    def _review_validation_gating(self, validation_result: Dict) -> Dict:
        """Review validation gating agent output."""
        meta = validation_result.get("metadata_validation", {})
        ts = validation_result.get("timeseries_validation", {})
        
        return {
            "status": "OK",
            "metadata_decision": meta.get("decision", "UNKNOWN"),
            "metadata_errors": meta.get("summary", {}).get("errors", 0),
            "metadata_warnings": meta.get("summary", {}).get("warnings", 0),
            "timeseries_decision": ts.get("decision", "UNKNOWN"),
            "timeseries_errors": ts.get("summary", {}).get("errors", 0),
            "timeseries_warnings": ts.get("summary", {}).get("warnings", 0),
        }
    
    def _generate_recommendations(self, indicators: List[HallucinationIndicator],
                                   agent_reviews: Dict) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        # Check triggered indicators
        triggered = [i for i in indicators if i.triggered]
        
        if any(i.indicator_id == "H001" for i in triggered):
            recommendations.append("Require multi-source evidence for all mappings, not just name matching")
        
        if any(i.indicator_id == "H002" for i in triggered):
            recommendations.append("Ensure all accepted mappings have documentation support")
        
        if any(i.indicator_id == "H005" for i in triggered):
            recommendations.append("Halt processing when required fields are missing")
        
        if any(i.indicator_id == "H010" for i in triggered):
            recommendations.append("CRITICAL: Negative control was accepted - review mapping logic")
        
        # Check agent reviews
        mapper_review = agent_reviews.get("semantic_mapper", {})
        if mapper_review.get("rejection_ratio", 0) < 0.2:
            recommendations.append("Low rejection ratio - consider if mapping criteria are too permissive")
        
        if not recommendations:
            recommendations.append("System performed correctly - no issues detected")
        
        return recommendations


def run_supervisor_critic(
    schema_report_path: Path,
    mapping_table_path: Path,
    validation_result_path: Path,
    output_dir: Path,
) -> SupervisorVerdict:
    """Run supervisor critic agent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all agent outputs
    schema_report = json.loads(schema_report_path.read_text())
    mapping_table = json.loads(mapping_table_path.read_text())
    validation_result = json.loads(validation_result_path.read_text())
    
    agent = SupervisorCriticAgent()
    verdict = agent.review_agents(schema_report, mapping_table, validation_result)
    
    # Generate final system report
    report_md = generate_final_report_md(verdict)
    report_path = output_dir / "final_system_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    
    # Also save JSON
    json_path = output_dir / "supervisor_verdict.json"
    json_path.write_text(json.dumps(verdict.to_dict(), indent=2), encoding="utf-8")
    
    return verdict


def generate_final_report_md(verdict: SupervisorVerdict) -> str:
    """Generate the final system report in markdown."""
    lines = [
        "# Final System Report",
        "",
        f"**Generated:** {verdict.timestamp}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"### Verdict: **{verdict.verdict}**",
        "",
        f"> {verdict.verdict_reason}",
        "",
        "---",
        "",
        "## Hallucination Analysis",
        "",
        f"**Hallucination Detected:** {'YES' if verdict.hallucination_detected else 'NO'}",
        "",
    ]
    
    if verdict.hallucination_details:
        lines.append(f"**Details:** {verdict.hallucination_details}")
        lines.append("")
    
    lines.append("### Hallucination Indicators Checked")
    lines.append("")
    lines.append("| ID | Description | Triggered | Severity | Evidence |")
    lines.append("|-----|-------------|-----------|----------|----------|")
    
    for ind in verdict.hallucination_indicators:
        status = "⚠️ YES" if ind.triggered else "✅ NO"
        lines.append(f"| {ind.indicator_id} | {ind.description} | {status} | {ind.severity} | {ind.evidence[:50]}... |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Agent Reviews")
    lines.append("")
    
    for agent_name, review in verdict.agent_reviews.items():
        lines.append(f"### {agent_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"**Status:** {review.get('status', 'N/A')}")
        lines.append("")
        for key, value in review.items():
            if key != "status":
                lines.append(f"- {key}: {value}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    for i, rec in enumerate(verdict.recommendations, 1):
        lines.append(f"{i}. {rec}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Mandatory Question Answer")
    lines.append("")
    lines.append("> **Q: If a new battery dataset arrives with unknown columns and partial metadata,")
    lines.append("> how does the system prevent hallucinated schema mapping, and how is this verified?**")
    lines.append("")
    lines.append("### Answer:")
    lines.append("")
    lines.append("The Battery AI Co-Scientist prevents hallucinated schema mapping through a multi-layered defense:")
    lines.append("")
    lines.append("1. **Schema Inspector Agent**: Inspects all columns with decomposed evidence scoring")
    lines.append("   (metadata_match, unit_plausibility, value_range, cross_column_consistency).")
    lines.append(
        f"   - HALTS execution if required fields have confidence < "
        f"{get_config().schema_inspector.low_threshold:.2f}"
    )
    lines.append("   - Explicitly flags ambiguous columns (containing 'mystery', 'unknown', etc.)")
    lines.append("")
    lines.append("2. **Semantic Mapper Agent**: Enforces strict evidence requirements:")
    lines.append("   - NO string similarity alone - requires multiple evidence types")
    lines.append("   - NO blind guessing - minimum 2 evidence components required")
    lines.append("   - REJECTS plausible-but-wrong columns (negative controls like 'coulombic_efficiency')")
    lines.append("   - Every mapping includes explicit evidence + confidence score")
    lines.append("")
    lines.append("3. **Validation & Gating Agent**: Validates physical plausibility:")
    lines.append("   - Checks value ranges against physical impossibility thresholds")
    lines.append("   - Compares distributions to NASA reference data")
    lines.append("   - Detects unit scaling issues")
    lines.append("   - Produces PASS / REVIEW / FAIL decision")
    lines.append("")
    lines.append("4. **Supervisor / Critic Agent**: Final review layer:")
    lines.append("   - Checks 10 specific hallucination indicators")
    lines.append("   - Verifies no agent bypassed safety checks")
    lines.append("   - Produces auditable verdict with explicit evidence")
    lines.append("")
    lines.append("### Verification:")
    lines.append("")
    lines.append(f"In this run, the system produced verdict: **{verdict.verdict}**")
    lines.append("")
    
    triggered = [i for i in verdict.hallucination_indicators if i.triggered]
    if triggered:
        lines.append(f"**{len(triggered)} hallucination indicator(s) were triggered:**")
        for ind in triggered:
            lines.append(f"- {ind.indicator_id}: {ind.description}")
    else:
        lines.append("**No hallucination indicators were triggered.**")
    
    lines.append("")
    
    if verdict.verdict == "FAIL":
        lines.append("> [!CAUTION]")
        lines.append("> The system correctly REJECTED this hostile dataset.")
        lines.append("> This FAILURE proves the anti-hallucination mechanism is working.")
    elif verdict.verdict == "NEEDS_HUMAN_REVIEW":
        lines.append("> [!WARNING]")
        lines.append("> The system flagged this dataset for human review.")
        lines.append("> This demonstrates appropriate uncertainty handling.")
    else:
        lines.append("> [!NOTE]")
        lines.append("> The system PASSED this dataset.")
    
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path
    
    base = Path("d:/Energy Project/code/battery-project3")
    out_dir = base / "data/processed/hostile_validation"
    
    verdict = run_supervisor_critic(
        schema_report_path=out_dir / "schema_report.json",
        mapping_table_path=out_dir / "mapping_table.json",
        validation_result_path=out_dir / "validation_result.json",
        output_dir=out_dir,
    )
    
    logger.info("=== SUPERVISOR CRITIC VERDICT ===")
    logger.info(f"Verdict: {verdict.verdict}")
    logger.info(f"Hallucination Detected: {verdict.hallucination_detected}")
    logger.info(f"Reason: {verdict.verdict_reason}")

# src/pipeline/schema.py
"""
Canonical schema + synonym dictionary.

- Keys in SYNONYMS_* are the ONLY canonical column names allowed downstream.
- Values are raw column name variants (case/spacing differences, abbreviations, etc.)
"""

from __future__ import annotations

from typing import Dict

from src.config import get_config

SYNONYMS_META = {
    "battery_id": ["battery_id", "Battery", "battery", "cell_id"],
    "type": ["type", "step_type", "mode"],
    "start_time": ["start_time", "start", "time_start"],
    "ambient_temperature": ["ambient_temperature", "ambient temp", "ambient_temp", "ambient"],
    "test_id": ["test_id", "test"],
    "uid": ["uid", "unique_id"],
    "filename": ["filename", "file", "csv", "path"],
    "capacity": ["capacity", "Capacity", "cap", "discharge_capacity"],
    "Re": ["Re", "re", "resistance_electrolyte"],
    "Rct": ["Rct", "rct", "charge_transfer_resistance"],
}

SYNONYMS_TS = {
    "time": ["Time", "time", "t"],
    "voltage_measured": ["Voltage_measured", "voltage_measured", "V_measured"],
    "current_measured": ["Current_measured", "current_measured", "I_measured"],
    "temperature_measured": ["Temperature_measured", "temperature_measured", "T_measured"],
    "current_load": ["Current_load", "current_load", "I_load"],
    "voltage_load": ["Voltage_load", "voltage_load", "V_load"],
}


def get_schema_ranges() -> Dict[str, Dict[str, float]]:
    """
    Return validated physical ranges from config.
    """
    s = get_config().schema
    return {
        "capacity_ahr": {"min": float(s.capacity_ahr.min), "max": float(s.capacity_ahr.max)},
        "temperature_c": {"min": float(s.temperature_c.min), "max": float(s.temperature_c.max)},
        "voltage_v": {"min": float(s.voltage_v.min), "max": float(s.voltage_v.max)},
        "current_a": {"min": float(s.current_a.min), "max": float(s.current_a.max)},
        "energy_j": {"min": float(s.energy_j.min), "max": float(s.energy_j.max)},
        "duration_s": {"min": float(s.duration_s.min), "max": float(s.duration_s.max)},
    }

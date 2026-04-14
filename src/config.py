"""
Validated pipeline configuration loader.

Usage:
    from src.config import get_config
    cfg = get_config()
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigValidationError(ValueError):
    """Raised when config YAML fails validation."""


class SplitConfig(BaseModel):
    train_ratio: float = Field(0.80, ge=0.5, le=0.95)
    random_seed: int = Field(42, ge=0)
    n_cal_batteries: int = Field(6, ge=2, le=20)
    stratify_by: str = Field("temperature_group")

    @field_validator("stratify_by")
    @classmethod
    def _validate_stratify_by(cls, value: str) -> str:
        allowed = {"temperature_group", "none"}
        if value not in allowed:
            raise ValueError(f"stratify_by must be one of {allowed}, got '{value}'")
        return value


class DLModelConfig(BaseModel):
    sequence_length: int = Field(8, ge=2, le=50)
    hidden_channels: int = Field(32, ge=8, le=512)
    learning_rate: float = Field(1e-3, gt=0.0, lt=1.0)
    batch_size: int = Field(64, ge=8, le=1024)
    epochs: int = Field(60, ge=5, le=2000)
    patience: int = Field(10, ge=1, le=200)
    dropout: float = Field(0.10, ge=0.0, le=0.8)
    weight_decay: float = Field(1e-5, ge=0.0, lt=1.0)
    force_backend: str = Field("auto")

    @field_validator("force_backend")
    @classmethod
    def _validate_force_backend(cls, value: str) -> str:
        allowed = {"auto", "torch_tcn", "keras"}
        if value not in allowed:
            raise ValueError(f"force_backend must be one of {allowed}, got '{value}'")
        return value

    @model_validator(mode="after")
    def _validate_patience(self) -> "DLModelConfig":
        if self.patience >= self.epochs:
            raise ValueError(f"patience ({self.patience}) must be < epochs ({self.epochs})")
        return self


class XGBoostConfig(BaseModel):
    n_estimators: int = Field(400, ge=10, le=5000)
    max_depth: int = Field(6, ge=1, le=20)
    learning_rate: float = Field(0.05, gt=0.0, lt=1.0)
    subsample: float = Field(0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(0.8, gt=0.0, le=1.0)
    random_state: int = Field(42, ge=0)
    quantile_alphas: list[float] = Field(default_factory=lambda: [0.05, 0.50, 0.95])

    @field_validator("quantile_alphas")
    @classmethod
    def _validate_quantile_alphas(cls, values: list[float]) -> list[float]:
        if not values:
            raise ValueError("quantile_alphas cannot be empty")
        if not all(0.0 < q < 1.0 for q in values):
            raise ValueError("all quantile_alphas must be in (0,1)")
        sorted_vals = sorted(values)
        if 0.50 not in sorted_vals:
            raise ValueError("quantile_alphas must include 0.50")
        return sorted_vals


class ConformalConfig(BaseModel):
    coverage: float = Field(0.90, gt=0.5, lt=1.0)
    min_cal_batteries: int = Field(2, ge=1, le=10)
    safety_factor: float = Field(1.20, ge=1.0, le=2.0)
    lobo_groups: list[str] = Field(default_factory=lambda: ["cold"])

    @field_validator("lobo_groups")
    @classmethod
    def _validate_lobo_groups(cls, values: list[str]) -> list[str]:
        allowed = {"room", "hot", "cold"}
        bad = set(values) - allowed
        if bad:
            raise ValueError(f"lobo_groups contains unknown groups: {sorted(bad)}")
        return values


class UncertaintyConfig(BaseModel):
    q_low: float = Field(0.05, gt=0.0, lt=0.5)
    q_high: float = Field(0.95, gt=0.5, lt=1.0)
    min_sigma: float = Field(1.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_quantile_order(self) -> "UncertaintyConfig":
        if self.q_low >= self.q_high:
            raise ValueError(f"q_low ({self.q_low}) must be < q_high ({self.q_high})")
        return self


class RiskConfig(BaseModel):
    horizon_cycles: int = Field(20, ge=1, le=500)
    high_threshold: float = Field(0.70, gt=0.0, lt=1.0)
    medium_threshold: float = Field(0.30, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "RiskConfig":
        if self.medium_threshold >= self.high_threshold:
            raise ValueError(
                f"medium_threshold ({self.medium_threshold}) must be < high_threshold ({self.high_threshold})"
            )
        return self

    def risk_category(self, failure_prob: float) -> str:
        if failure_prob >= self.high_threshold:
            return "HIGH"
        if failure_prob >= self.medium_threshold:
            return "MEDIUM"
        return "LOW"


class DriftConfig(BaseModel):
    psi_amber_threshold: float = Field(0.10, gt=0.0, lt=1.0)
    psi_red_threshold: float = Field(0.20, gt=0.0, lt=2.0)
    n_bins: int = Field(10, ge=3, le=100)

    @model_validator(mode="after")
    def _validate_order(self) -> "DriftConfig":
        if self.psi_amber_threshold >= self.psi_red_threshold:
            raise ValueError(
                f"psi_amber_threshold ({self.psi_amber_threshold}) must be < "
                f"psi_red_threshold ({self.psi_red_threshold})"
            )
        return self


class RangeConfig(BaseModel):
    min: float
    max: float

    @model_validator(mode="after")
    def _validate_min_max(self) -> "RangeConfig":
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be < max ({self.max})")
        return self


class SchemaConfig(BaseModel):
    capacity_ahr: RangeConfig = Field(default_factory=lambda: RangeConfig(min=1.0, max=2.0))
    temperature_c: RangeConfig = Field(default_factory=lambda: RangeConfig(min=4.0, max=44.0))
    voltage_v: RangeConfig = Field(default_factory=lambda: RangeConfig(min=2.0, max=4.2))
    current_a: RangeConfig = Field(default_factory=lambda: RangeConfig(min=0.5, max=4.0))
    energy_j: RangeConfig = Field(default_factory=lambda: RangeConfig(min=5000, max=30000))
    duration_s: RangeConfig = Field(default_factory=lambda: RangeConfig(min=1000, max=7000))


class EolConfig(BaseModel):
    nominal_capacity_ahr: float = Field(2.0, gt=0.0)
    capacity_fade_fraction: float = Field(0.80, gt=0.0, lt=1.0)

    @property
    def eol_capacity_ahr(self) -> float:
        return self.nominal_capacity_ahr * self.capacity_fade_fraction


class AnomalyConfig(BaseModel):
    capacity_floor_factor: float = Field(0.50, gt=0.0, lt=1.0)
    flagged_battery_prefixes: list[str] = Field(default_factory=list)


class SurvivalConfig(BaseModel):
    horizon_cycles: int = Field(20, ge=1, le=500)
    method: str = Field("kaplan_meier")

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        allowed = {"kaplan_meier", "cox_ph"}
        if value not in allowed:
            raise ValueError(f"method must be one of {allowed}, got '{value}'")
        return value


class HostileConfig(BaseModel):
    low_conf_threshold: float = Field(0.50, gt=0.0, lt=1.0)
    high_conf_threshold: float = Field(0.75, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_order(self) -> "HostileConfig":
        if self.low_conf_threshold >= self.high_conf_threshold:
            raise ValueError(
                f"low_conf_threshold ({self.low_conf_threshold}) must be < "
                f"high_conf_threshold ({self.high_conf_threshold})"
            )
        return self


class GatingConfig(BaseModel):
    low_threshold: float = Field(0.60, gt=0.0, lt=1.0)
    review_threshold: float = Field(0.85, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_order(self) -> "GatingConfig":
        if self.low_threshold >= self.review_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be < review_threshold ({self.review_threshold})"
            )
        return self


class SchemaInspectorConfig(BaseModel):
    low_threshold: float = Field(0.60, gt=0.0, lt=1.0)
    review_threshold: float = Field(0.85, gt=0.0, lt=1.0)
    required_evidence_score: float = Field(0.70, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_order(self) -> "SchemaInspectorConfig":
        if self.low_threshold >= self.review_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be < review_threshold ({self.review_threshold})"
            )
        if self.required_evidence_score < self.low_threshold:
            raise ValueError(
                f"required_evidence_score ({self.required_evidence_score}) must be >= "
                f"low_threshold ({self.low_threshold})"
            )
        return self


class SupervisorConfig(BaseModel):
    low_confidence_accept_threshold: float = Field(0.70, gt=0.0, lt=1.0)
    rmse_pass_threshold: float = Field(100.0, gt=0.0)
    cv_rmse_ratio_warn_threshold: float = Field(1.5, gt=1.0)


class PathsConfig(BaseModel):
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    trained_models: str = "trained_models"
    outputs: str = "outputs"
    hostile_timeseries: str = "data/raw/hostile/timeseries"
    hostile_output: str = "data/processed/hostile_validation"

    def resolve(self, root: Path, attr: str) -> Path:
        return root / getattr(self, attr)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    split: SplitConfig = Field(default_factory=SplitConfig)
    dl_model: DLModelConfig = Field(default_factory=DLModelConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    schema_config: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    eol: EolConfig = Field(default_factory=EolConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    survival: SurvivalConfig = Field(default_factory=SurvivalConfig)
    hostile: HostileConfig = Field(default_factory=HostileConfig)
    gating: GatingConfig = Field(default_factory=GatingConfig)
    schema_inspector: SchemaInspectorConfig = Field(default_factory=SchemaInspectorConfig)
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @property
    def schema(self) -> SchemaConfig:
        return self.schema_config

    @model_validator(mode="after")
    def _validate_horizon_match(self) -> "PipelineConfig":
        if self.risk.horizon_cycles != self.survival.horizon_cycles:
            raise ValueError(
                f"risk.horizon_cycles ({self.risk.horizon_cycles}) must equal "
                f"survival.horizon_cycles ({self.survival.horizon_cycles})"
            )
        return self

    def summary(self) -> str:
        return (
            "PipelineConfig | "
            f"seed={self.split.random_seed} "
            f"train_ratio={self.split.train_ratio:.2f} "
            f"horizon={self.risk.horizon_cycles} "
            f"conformal_coverage={self.conformal.coverage:.0%} "
            f"eol_capacity={self.eol.eol_capacity_ahr:.3f}Ahr"
        )


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=4)
def get_config(config_path: Optional[str] = None) -> PipelineConfig:
    env_path = os.environ.get("BATTERY_CONFIG")
    resolved = Path(config_path or env_path or _DEFAULT_CONFIG_PATH)

    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}\n"
            f"Expected at: {_DEFAULT_CONFIG_PATH}\n"
            "Override with BATTERY_CONFIG or get_config(config_path=...)."
        )

    raw = _load_yaml(resolved)
    try:
        return PipelineConfig(**raw)
    except Exception as exc:
        raise ConfigValidationError(f"Configuration validation failed in '{resolved}':\n{exc}") from exc


def reload_config(config_path: Optional[str] = None) -> PipelineConfig:
    get_config.cache_clear()
    return get_config(config_path=config_path)

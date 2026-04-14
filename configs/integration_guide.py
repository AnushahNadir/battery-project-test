"""
HOW TO ADOPT get_config() IN EACH MODULE
========================================

This file shows the exact before/after for every module that has
magic numbers. Apply each block to its target file.
"""

# =============================================================================
# 1. run_full_pipeline.py
# =============================================================================
#
# ADD at the top (after existing imports):
#
#   from src.config import get_config
#
# Then replace hardcoded split/DL/anomaly/risk values with cfg.* lookups.


# =============================================================================
# 2. conformal.py
# =============================================================================
#
# Replace:
#   LOBO_GROUPS / MIN_CAL_BATTERIES / SAFETY_FACTOR
# with:
#   cfg.conformal.lobo_groups / cfg.conformal.min_cal_batteries / cfg.conformal.safety_factor


# =============================================================================
# 3. uncertainty.py
# =============================================================================
#
# Replace hardcoded quantile/risk thresholds:
#   Q_LOW=0.05, Q_HIGH=0.95, horizon=20, risk tiers 0.70/0.30
# with:
#   cfg.uncertainty + cfg.risk values.


# =============================================================================
# 4. run_hostile_validation.py
# =============================================================================
#
# Replace:
#   LOW_CONF_THRESHOLD=0.50 / HIGH_CONF_THRESHOLD=0.75
# with:
#   cfg.hostile.low_conf_threshold / cfg.hostile.high_conf_threshold
#
# Also read hostile input/output dirs from cfg.paths.


# =============================================================================
# 5. pipeline/schema.py + validation.py
# =============================================================================
#
# Physical limits must come from cfg.schema.* ranges rather than literals.


# =============================================================================
# 6. analysis/rul.py
# =============================================================================
#
# Replace hardcoded EOL alpha defaults with cfg.eol.capacity_fade_fraction.


# =============================================================================
# 7. startup logging
# =============================================================================
#
# Add to main():
#   cfg = get_config()
#   print(f"[Config] {cfg.summary()")

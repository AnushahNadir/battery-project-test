# Assumptions And Known Limits

## Known Limitation: B0055 Uncertainty Calibration

- Observation: Uncertainty intervals for battery `B0055` are under-calibrated (too narrow relative to observed prediction error).
- Impact: Reported confidence for `B0055` can be over-optimistic, even when point error is high.
- Current decision: Accepted as a known limitation for this release.
- Planned fix: Revisit during conformal prediction implementation to calibrate interval coverage at battery/test-distribution level.

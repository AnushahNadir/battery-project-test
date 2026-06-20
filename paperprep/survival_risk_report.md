# Survival Risk Report

**Model:** cox_ph_proxy_logistic_hazard
**Configured method:** cox_ph
**Horizon:** 30 cycles

## What this means (simple)

- `hazard_prob` = chance the battery fails *at this cycle* (given it survived before).
- `failure_prob_horizon` = chance the battery fails within the next **30 cycles**.
- This is a risk signal, not a causal explanation.

## Data summary

- Rows: 638
- Batteries: 6
- Events observed: 5
- Event rate: 0.7837%

## Risk distribution (by row)

| Category | Count | % |
|---|---:|---:|
| LOW | 351 | 55.0% |
| MEDIUM | 151 | 23.7% |
| HIGH | 136 | 21.3% |

## Top signals (model weights)

> Larger absolute weight = model uses that feature more for risk scoring.

| Feature | Weight |
|---|---:|
| `i_min` | -0.7767 |
| `v_mean` | 0.6616 |
| `capacity` | -0.4356 |
| `duration_s` | -0.2956 |
| `energy_j` | -0.2456 |
| `ah_est` | -0.1117 |
| `temp_mean` | -0.0957 |
| `i_mean` | 0.0682 |
| `temp_max` | -0.0185 |
| `v_min` | 0.0000 |

## Notes

- Configured method='cox_ph' is approximated with logistic discrete-time hazard in this version.
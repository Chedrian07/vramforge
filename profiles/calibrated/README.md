# Calibrated profiles

This registry is intentionally **empty**. Calibrated profiles come from optional GPU validation
(plan.md §17.4, milestone M5), which is not connected in this release.

A calibrated profile will key a correction on architecture, loading scope, objective, adapter,
dtype, kernel path, GPU, dependency lock digest, length range and batch composition, and must state
its measured domain. Outside that domain the result falls back to the analytic profile and reports
`CALIBRATION_OUT_OF_DOMAIN`.

Until profiles exist here, every estimate is graded `analytic` (plan.md §11.4) and is never shown as
calibrated or measured.

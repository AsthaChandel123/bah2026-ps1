"""urbanheat.interventions — cooling-scenario simulation and optimization.

* ``catalog`` — the intervention type registry (driver perturbations + cited degC
  ranges) backed by :data:`urbanheat.constants.INTERVENTION_PARAMS`. [R6 §2]
* ``simulate`` — counterfactual ΔLST/ΔT_air via the trained model (perturb
  drivers, re-predict) with distance decay and physical clipping. [R5 §6 / R6]
* ``invest_cooling`` — a GEE/numpy-portable port of the InVEST Urban Cooling
  Model (CC -> HM -> T_air) as an independent biophysical ΔT estimator. [R6 §5]
* ``optimize`` — lazy-greedy submodular + ILP + NSGA-II placement under
  budget/area/equity constraints, returning a ranked portfolio. [R6 §7]
"""

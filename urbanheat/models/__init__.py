"""urbanheat.models — the physics-informed ML core (LST <-> drivers).

* ``features`` — assemble the predictor matrix X and target y from a FeatureStack.
* ``train`` — fit the monotone-constrained gradient-boosting ensemble (+ optional
  MGWR spatial layer and PINN reconciler) producing a differentiable response
  surface. [R5 §4]
* ``attribution`` — SHAP / ALE driver attribution with physics-sign audit, plus
  variance partitioning and GWR coefficient maps. [R5 §5 / R9 §2]
* ``validation`` — spatial cross-validation, the metric panel, and physics-
  consistency checks. [R9 §1]
"""

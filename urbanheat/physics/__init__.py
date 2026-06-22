"""urbanheat.physics — surface-energy-balance physics and the PINN.

* ``energy_balance`` — SEB terms (Q*, Q_H, Q_E, dQ_S via OHM, Q_F), the radiative
  law L_up = eps*sigma*Ts^4, Bowen ratio, and a physics-only LST backbone. [R5 §1]
* ``pinn`` — the physics-informed neural network reconciler (heat-PDE + SEB-
  closure + monotonicity loss). torch is imported lazily; this module is optional. [R5 §3.1]
"""

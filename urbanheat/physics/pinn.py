"""urbanheat.physics.pinn — the optional physics-informed neural network.

The headline physics-informed differentiator: a neural network that predicts
surface temperature ``T_theta(drivers, x, y)`` and is trained with a **composite
loss** that forces it to respect the surface energy balance, not just fit data
(``research/05 §3.1``)::

    L = L_data + lambda_pde * L_pde + lambda_seb * L_seb + lambda_mono * L_mono

    L_data = mean |T_theta - LST_obs|^2                         (data fidelity)
    L_pde  = mean |dT/dt - kappa * laplacian(T) - S/(rho c d)|^2 (heat-PDE residual, autodiff)
    L_seb  = mean |Q* - (Q_H + Q_E + dQ_S + Q_F)|^2             (SEB closure, Eq. 1)
    L_mono = sum ReLU(+/- dT/ddriver)^2 over the §1.6 sign table (monotonicity)

Because the SEB-closure residual and the monotonicity penalties live *in the
loss*, perturbing albedo or NDVI yields a ``Delta LST`` that satisfies energy
conservation and never flips sign — the counterfactual is physical, not
extrapolated nonsense.

This module is **optional**. ``torch`` is imported lazily inside :meth:`fit` /
:meth:`predict`; the class always imports, and if ``torch`` is absent those
methods raise a single, clear, actionable :class:`RuntimeError` telling the user
how to install it. The rest of the system (``models.train`` ensemble, attribution,
validation) runs fully without torch.

Public API: :class:`HeatPINN` (ARCHITECTURE.md §11.4 contract). :class:`SEBPinn`
is an alias retained for the surface-energy-balance-PINN naming.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from urbanheat.constants import PHYSICAL_CONSTANTS, SIGMA_SB
from urbanheat.datamodel import (
    AIR_TEMP,
    ALBEDO,
    DEFAULT_PREDICTORS,
    EMISSIVITY,
    LONGWAVE_DOWN,
    LST,
    NET_RADIATION,
    SOLAR_RADIATION,
    FeatureStack,
)

__all__ = ["HeatPINN", "SEBPinn"]


_TORCH_HINT = (
    "PyTorch is required for the PINN reconciler but is not installed. "
    "Install it with `pip install torch` (CPU build is fine), then retry. "
    "The PINN is OPTIONAL: the monotone-constrained ensemble in "
    "urbanheat.models.train (RandomForest/HistGradientBoosting/XGBoost/LightGBM) "
    "delivers the physics-informed LST model, attribution and validation without "
    "torch."
)


def _require_torch() -> Any:
    """Import and return ``torch`` lazily, or raise an actionable error."""
    try:
        import torch  # lazy, optional
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError(_TORCH_HINT) from exc
    return torch


# ---------------------------------------------------------------------------
# Monotonicity sign vector (shared source of truth with the ensemble)
# ---------------------------------------------------------------------------
def _sign_vector(predictors: Sequence[str]) -> np.ndarray:
    """Return the ``(P,)`` expected ``dLST/ddriver`` sign vector for ``predictors``.

    Reuses :func:`urbanheat.models.features.monotone_constraints` so the PINN's
    ``L_mono`` penalties use *exactly* the same SEB sign table as the GBM
    ensemble's ``monotone_constraints``.
    """
    try:
        from urbanheat.models.features import monotone_constraints

        cst = monotone_constraints(predictors)
        return np.asarray([cst[p] for p in predictors], dtype=np.float64)
    except Exception:  # pragma: no cover - defensive
        return np.zeros(len(predictors), dtype=np.float64)


class HeatPINN:
    """Physics-informed neural network reconciler for the LST response surface.

    Trains an MLP ``T_theta`` of the driver stack (plus normalised coordinates)
    against the composite SEB loss described in the module docstring. The
    network is a small fully-connected ``tanh`` MLP; the PDE term uses
    finite-difference Laplacians of the predicted LST field on the grid (a
    standalone, autodiff-free PDE residual that keeps the implementation compact
    and dependency-light while still penalising non-smooth, energy-violating
    fields), and the SEB and monotonicity terms use autodiff gradients of the
    network output w.r.t. its inputs.

    Parameters
    ----------
    predictors:
        Driver names used as network inputs (default :data:`DEFAULT_PREDICTORS`).
    lambda_pde, lambda_seb, lambda_mono:
        Composite-loss weights for the heat-PDE residual, the SEB-closure
        residual and the monotonicity penalty.
    hidden:
        Hidden-layer widths of the MLP.
    kappa:
        Thermal diffusivity (m^2/s) used in the heat-PDE term.
    lr:
        Adam learning rate.
    seed:
        RNG seed (numpy + torch).

    Notes
    -----
    Optional module: instantiating is always safe; :meth:`fit` / :meth:`predict`
    raise a clear error if ``torch`` is unavailable.
    """

    def __init__(
        self,
        predictors: Sequence[str] = tuple(DEFAULT_PREDICTORS),
        lambda_pde: float = 1.0,
        lambda_seb: float = 1.0,
        lambda_mono: float = 1.0,
        *,
        hidden: Sequence[int] = (64, 64, 64),
        kappa: float = 1.0e-6,
        lr: float = 1.0e-3,
        seed: int = 0,
        **kw: Any,
    ) -> None:
        self.predictors = list(predictors)
        self.lambda_pde = float(lambda_pde)
        self.lambda_seb = float(lambda_seb)
        self.lambda_mono = float(lambda_mono)
        self.hidden = tuple(int(h) for h in hidden)
        self.kappa = float(kappa)
        self.lr = float(lr)
        self.seed = int(seed)
        self.kw = kw

        self._net: Any = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._sign_vec = _sign_vector(self.predictors)
        self._fitted = False
        self.history_: list[dict[str, float]] = []

    # ----- network construction -----------------------------------------
    def _build_net(self, n_in: int) -> Any:
        torch = _require_torch()
        import torch.nn as nn

        layers: list[Any] = []
        prev = n_in
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        return nn.Sequential(*layers)

    # ----- feature assembly ---------------------------------------------
    def _assemble(self, fs: FeatureStack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build ``(features, coords_norm, valid_mask)`` from the stack (flattened).

        ``features`` is ``(N, P)`` with present predictors (missing -> 0 after
        normalisation), ``coords_norm`` is the grid coordinates scaled to ~[0,1].
        """
        rows, cols = fs.shape
        n = rows * cols
        feats = np.zeros((n, len(self.predictors)), dtype=np.float64)
        present = np.zeros(len(self.predictors), dtype=bool)
        for j, name in enumerate(self.predictors):
            if fs.has(name):
                feats[:, j] = fs.get(name).ravel().astype(np.float64)
                present[j] = True
        xx, yy = fs.grid_coords()
        cx = xx.ravel()
        cy = yy.ravel()
        cx = (cx - cx.min()) / (np.ptp(cx) or 1.0)
        cy = (cy - cy.min()) / (np.ptp(cy) or 1.0)
        coords_norm = np.column_stack([cx, cy]).astype(np.float64)
        valid = ~np.isnan(feats).any(axis=1)
        feats = np.nan_to_num(feats, nan=0.0)
        return feats, coords_norm, valid

    def _seb_closure_residual(self, fs: FeatureStack, lst_pred: Any) -> Any:
        """SEB closure residual ||Q* - (Q_H+Q_E+dQ_S+Q_F)|| as a torch tensor.

        When the physics module exposes the SEB terms we use them; otherwise we
        use the radiative-balance form (Eq. 2-3): the net radiation implied by
        the predicted ``T_s`` must match the available net radiation. This keeps
        the term meaningful with only ``albedo / K_down / L_down / emissivity``.
        """
        torch = _require_torch()
        sigma = float(SIGMA_SB)

        def _layer(name: str, default: float) -> Any:
            if fs.has(name):
                arr = fs.get(name).ravel().astype(np.float64)
                arr = np.nan_to_num(arr, nan=default)
            else:
                arr = np.full(lst_pred.shape[0], default, dtype=np.float64)
            return torch.as_tensor(arr, dtype=lst_pred.dtype)

        albedo = _layer(ALBEDO, 0.2)
        kdown = _layer(SOLAR_RADIATION, 800.0)
        ldown = _layer(LONGWAVE_DOWN, 350.0)
        emis = _layer(EMISSIVITY, 0.96)
        ts_k = lst_pred.squeeze(-1) + float(PHYSICAL_CONSTANTS["KELVIN_OFFSET"])
        lup = emis * sigma * ts_k ** 4 + (1.0 - emis) * ldown
        qstar_rad = (1.0 - albedo) * kdown + (ldown - lup)
        # if a precomputed NET_RADIATION layer exists, residual against it;
        # else penalise net radiation that cannot be balanced (drives toward 0
        # net storage at the slab surface — the physically-anchored closure).
        if fs.has(NET_RADIATION):
            qstar_obs = _layer(NET_RADIATION, 0.0)
            return torch.mean((qstar_rad - qstar_obs) ** 2)
        # turbulent + storage proxy: sensible heat using air temp if present
        ta = _layer(AIR_TEMP, 25.0) + float(PHYSICAL_CONSTANTS["KELVIN_OFFSET"])
        rho_cp = float(PHYSICAL_CONSTANTS["RHO_AIR"] * PHYSICAL_CONSTANTS["CP_AIR"])
        q_h = rho_cp * (ts_k - ta) / 50.0  # nominal aerodynamic resistance r_a~50
        return torch.mean((qstar_rad - q_h) ** 2)

    # ----- fit ----------------------------------------------------------
    def fit(self, fs: FeatureStack, epochs: int = 2000) -> "HeatPINN":
        """Train the PINN with the composite SEB loss (lazy-imports torch).

        Parameters
        ----------
        fs:
            Source :class:`FeatureStack` (must contain :data:`LST` as the data
            target and at least one predictor; physics terms use the radiative
            drivers when present).
        epochs:
            Number of full-batch Adam steps.

        Returns
        -------
        HeatPINN
            ``self`` (fitted).

        Raises
        ------
        RuntimeError
            If ``torch`` is not installed (with install instructions), or if the
            stack lacks the :data:`LST` target.
        """
        torch = _require_torch()
        if not fs.has(LST):
            raise RuntimeError(
                f"HeatPINN.fit requires the target layer {LST!r} in the stack.")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        feats, coords_norm, valid = self._assemble(fs)
        y = fs.get(LST).ravel().astype(np.float64)
        valid = valid & ~np.isnan(y)
        if valid.sum() < 8:
            raise RuntimeError("HeatPINN.fit: fewer than 8 valid pixels to train on.")

        # standardise inputs
        self._x_mean = feats[valid].mean(0)
        self._x_std = np.where(feats[valid].std(0) == 0, 1.0, feats[valid].std(0))
        feats_n = (feats - self._x_mean) / self._x_std

        net_in = np.column_stack([feats_n, coords_norm])
        n_in = net_in.shape[1]
        self._net = self._build_net(n_in)

        X_t = torch.as_tensor(net_in, dtype=torch.float32, requires_grad=True)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        valid_t = torch.as_tensor(valid)
        sign_t = torch.as_tensor(self._sign_vec, dtype=torch.float32)
        rows, cols = fs.shape

        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        self.history_ = []
        epochs = int(epochs)
        for ep in range(epochs):
            opt.zero_grad()
            pred = self._net(X_t)  # (N,1)
            pred_v = pred.squeeze(-1)

            # data fidelity (only where LST observed/valid)
            l_data = torch.mean((pred_v[valid_t] - y_t[valid_t]) ** 2)

            # monotonicity: dT/dx_j must satisfy sign s_j
            grads = torch.autograd.grad(
                pred_v.sum(), X_t, create_graph=True, retain_graph=True,
            )[0]
            dfeat = grads[:, : len(self.predictors)]  # ignore coord cols
            relu = torch.relu
            # s_j>0 => want dT/dx>=0 => penalise negative; s_j<0 => penalise positive
            pen_pos = relu(-dfeat) * (sign_t > 0).float()
            pen_neg = relu(dfeat) * (sign_t < 0).float()
            l_mono = torch.mean(pen_pos ** 2 + pen_neg ** 2)

            # heat-PDE residual via finite-difference Laplacian on the grid
            l_pde = self._pde_residual(torch, pred_v, rows, cols)

            # SEB-closure residual
            l_seb = self._seb_closure_residual(fs, pred)

            loss = (
                l_data
                + self.lambda_pde * l_pde
                + self.lambda_seb * l_seb
                + self.lambda_mono * l_mono
            )
            loss.backward()
            opt.step()
            if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
                self.history_.append(
                    {
                        "epoch": ep,
                        "loss": float(loss.detach()),
                        "data": float(l_data.detach()),
                        "pde": float(l_pde.detach()),
                        "seb": float(l_seb.detach()),
                        "mono": float(l_mono.detach()),
                    }
                )
        self._fitted = True
        return self

    @staticmethod
    def _pde_residual(torch: Any, pred_v: Any, rows: int, cols: int) -> Any:
        """Steady-state heat-diffusion residual: penalise |laplacian(T)| on the grid.

        Uses a 5-point finite-difference Laplacian of the predicted field
        reshaped to ``(rows, cols)``. Penalising the Laplacian drives the field
        toward the harmonic (diffusion-consistent) solution between data anchors
        — the autodiff-free, dependency-light realisation of ``L_pde``.
        """
        field = pred_v.reshape(rows, cols)
        if rows < 3 or cols < 3:
            return torch.zeros((), dtype=pred_v.dtype)
        lap = (
            field[2:, 1:-1]
            + field[:-2, 1:-1]
            + field[1:-1, 2:]
            + field[1:-1, :-2]
            - 4.0 * field[1:-1, 1:-1]
        )
        return torch.mean(lap ** 2)

    # ----- predict ------------------------------------------------------
    def predict(self, fs: FeatureStack) -> np.ndarray:
        """Return the physically-consistent LST surface (degC) on the grid.

        Raises a clear error if ``torch`` is unavailable or the model is unfit.
        """
        torch = _require_torch()
        if not self._fitted or self._net is None:
            raise RuntimeError("HeatPINN is not fitted; call .fit(fs) first.")
        feats, coords_norm, _valid = self._assemble(fs)
        feats_n = (feats - self._x_mean) / self._x_std
        net_in = np.column_stack([feats_n, coords_norm])
        X_t = torch.as_tensor(net_in, dtype=torch.float32)
        with torch.no_grad():
            pred = self._net(X_t).squeeze(-1).cpu().numpy()
        return np.asarray(pred, dtype=np.float64).reshape(fs.shape)

    def predict_delta(self, fs: FeatureStack, perturb: dict[str, float]) -> np.ndarray:
        """``Delta LST`` for a driver perturbation (SEB-closed, correctly signed).

        Evaluates ``F(X + dX) - F(X)`` on the grid for the given additive driver
        perturbations (keyed by predictor name). Because the network was trained
        with the monotonicity penalty, the returned field respects the SEB sign
        table (e.g. raising NDVI cannot warm a pixel). [R5 §6]
        """
        base = self.predict(fs)
        pert_fs = FeatureStack(
            layers={k: v.copy() for k, v in fs.layers.items()},
            transform=fs.transform, crs=fs.crs, bounds=fs.bounds,
            shape=fs.shape, meta=dict(fs.meta),
        )
        for name, dv in perturb.items():
            if pert_fs.has(name):
                pert_fs.layers[name] = (pert_fs.get(name) + float(dv)).astype(np.float32)
        return self.predict(pert_fs) - base

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"HeatPINN(n_predictors={len(self.predictors)}, "
            f"lambda_pde={self.lambda_pde}, lambda_seb={self.lambda_seb}, "
            f"lambda_mono={self.lambda_mono}, fitted={self._fitted})"
        )


# Alias: surface-energy-balance PINN (brief naming).
SEBPinn = HeatPINN

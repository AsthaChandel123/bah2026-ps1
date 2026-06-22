"""urbanheat.synthetic — offline synthetic data backend (demo + tests).

Generates physically-plausible synthetic LST and driver fields on a grid so the
ENTIRE pipeline runs end-to-end without GEE credentials or network. The
synthetic LST is built to respect the surface-energy-balance signs (cooler over
vegetation/water/high-albedo, hotter over impervious/low-SVF), so attribution,
counterfactuals and the optimizer all behave sensibly in demo mode.

Public entry point: :class:`urbanheat.synthetic.source.SyntheticDataSource`.
"""

# Hands-On Quantum Mechanics

Numerical quantum mechanics from scratch. Each notebook sets up a physical system, solves the Schrödinger equation on a grid, and visualizes the results — no black-box libraries, no hand-waving.

## Notebooks

| # | Topic | Key ideas |
|---|-------|-----------|
| 01 | [Dynamics of a Quantum Harmonic Oscillator](01_harmonic_oscillator_dynamics.py) | Grid discretization, finite-difference Hamiltonian, ODE time evolution, expectation values, probability current, Husimi Q function |
| 02 | [Beyond the Harmonic Oscillator: Anharmonicity and Dirac Notation](02_anharmonic_oscillator_and_dirac_notation.py) | Dirac notation, inner product, basis independence, energy eigenstates as a basis, anharmonic potential, energy spectrum, autocorrelation function, quantum revival |
| 03 | [Finding Energy Eigenstates: The Shooting Method](03_determining_eigenstates.py) | Stationary states, spatial ODE, finding eigenvalues and eigenstates, completeness, symmetry and selection rules, time-reversal symmetry, linearity |

## Prerequisites

- Python 3.10+
- NumPy, SciPy, Matplotlib

```
pip install numpy scipy matplotlib
```

## Running

Each notebook exists in two formats:
- **`.py`** — the source of truth. Uses `#%%` cell delimiters. Open in VS Code and run cells with **Shift+Enter** (requires the Python extension). Also runs as a plain script, for example: `python 01_harmonic_oscillator_dynamics.py`
- **`.ipynb`** — exported Jupyter notebook with the same content. Plots and animations are stripped (matplotlib/Jupyter uses inefficient encodings that bloat file size). Re-run all cells to regenerate output.

## Approach

Every concept is introduced with prose, then implemented in code, then visualized.

No prior quantum mechanics experience required — just comfort with Python and basic linear algebra.

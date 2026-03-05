#%%
r"""
# Finding Energy Eigenstates: The Shooting Method

So far we've been solving the time-dependent Schrodinger equation:

$$
i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle
$$

We discretized space, built the Hamiltonian as a matrix, and handed the whole
thing to an ODE solver that marched $|\psi\rangle$ forward in time. Along the
way we discovered that the dynamics boil down to the energy eigenstates
$|n\rangle$ satisfying $\hat{H}|n\rangle = E_n|n\rangle$. A coherent state
in the harmonic oscillator stays together because the $E_n$ are evenly spaced.
Add a quartic perturbation and the unequal spacing causes the wave packet to
spread, fragment, and eventually revive.

The eigenstates were the key to understanding all of it — but we obtained them
as a black box, calling `eigsh` on a matrix. In this notebook we go after them
directly.

## From Time Evolution to Eigenstates

Any state can be expanded in the energy eigenbasis:

$$
|\psi(t)\rangle = \sum_n c_n e^{-iE_n t/\hbar} |n\rangle
$$

Each component just rotates in the complex plane at its own frequency
$E_n/\hbar$. Suppose a state has only *one* component — it's already an
eigenstate $|\psi(t)\rangle = e^{-iEt/\hbar}|E\rangle$. Then the time
dependence is nothing but a global phase that has no physical consequence.
Such a state is **stationary**: all its observable properties are constant
in time. This is what "time-independent" really means — not that we've
changed the equation, but that we're looking for the special states where
nothing happens.

Setting $|\psi(t)\rangle = e^{-iEt/\hbar}|E\rangle$ and substituting into
the Schrodinger equation, the time dependence cancels and we get the
**eigenvalue equation**:

$$
\hat{H}|E\rangle = E|E\rangle
$$

To turn this into something we can compute, we project onto position states
$\langle x|$. Writing $\phi(x) = \langle x|E\rangle$ and expanding
$\hat{H} = \hat{T} + \hat{V}$:

$$
-\frac{\hbar^2}{2m}\frac{d^2\phi}{dx^2} + V(x)\phi(x) = E\phi(x)
$$

Rearranging:

$$
\frac{d^2\phi}{dx^2} = \frac{2m}{\hbar^2}\bigl[V(x) - E\bigr]\phi(x)
$$

Look at what this is: a second-order ODE in *space*. The energy $E$ enters
as a parameter. Given any value of $E$, we can pick initial conditions and
integrate across $x$ — exactly the way `solve_ivp` integrated the
time-dependent equation across $t$ in Notebook 1.

The catch: for a bound state, $\langle x|E\rangle$ must vanish as
$x \to \pm\infty$. Deep in the classically forbidden region (where
$V(x) > E$), the ODE becomes approximately $\phi'' \approx \kappa^2 \phi$
with $\kappa > 0$, which has two solutions: a decaying exponential (physical)
and a growing one (unphysical). If $E$ is not exactly an eigenvalue, the
growing component inevitably takes over and the solution explodes.

Let's see this happen. We'll use the anharmonic potential from Notebook 2 —
$V(x) = \frac{1}{2}m\omega^2 x^2 + \lambda x^4$ — and integrate the ODE
from the left boundary rightward with an energy that is *not* an eigenvalue.
"""

#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Natural units: hbar = m = omega = 1
hbar = 1.0
m = 1.0
omega = 1.0
lam = 0.01          # quartic coupling (same as Notebook 2)

def V(x):
    return 0.5 * m * omega**2 * x**2 + lam * x**4

# The ODE: phi'' = (2m/hbar^2) * (V(x) - E) * phi
# As a first-order system: y = [phi, phi']
def schrodinger_ode(x, y, E):
    phi, dphi = y
    d2phi = (2 * m / hbar**2) * (V(x) - E) * phi
    return [dphi, d2phi]

# Try an energy between the ground state (~0.51) and first excited state (~1.54)
E_try = 4.2

x_start, x_end = -6.0, 6.0
x_eval = np.linspace(x_start, x_end, 2000)

# Start with a tiny value deep in the forbidden region (left side).
# The exact numbers don't matter — the ODE is linear, so any nonzero
# start will give the same shape up to an overall scale.
sol = solve_ivp(
    lambda x, y: schrodinger_ode(x, y, E_try),
    [x_start, x_end],
    [1e-5, 1e-3],
    t_eval=x_eval,
    rtol=1e-10, atol=1e-12,
)

phi = sol.y[0]

# Normalize so the peak in the classically allowed region equals 1.
# This way the wave-like shape is clearly visible before the divergence.
allowed = np.abs(x_eval) < 2.5
phi /= np.max(np.abs(phi[allowed]))

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2]})

ax1.plot(x_eval, V(x_eval), 'k-', linewidth=1.5, label='V(x)')
ax1.axhline(E_try, color='coral', linestyle='--', linewidth=1,
            label=f'E = {E_try}')
ax1.set_ylabel('Energy')
ax1.set_ylim(0, 10)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title(f'Shooting with E = {E_try} (not an eigenvalue)')

ax2.plot(x_eval, phi, 'steelblue', linewidth=1.5)
ax2.axhline(0, color='k', linewidth=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel(r'$\phi(x)$')
ax2.set_ylim(-5, 5)
ax2.grid(True, alpha=0.3)

# Shade classically forbidden regions
forbidden = V(x_eval) > E_try
ax2.fill_between(x_eval, -5, 5, where=forbidden,
                 alpha=0.06, color='red', label='Classically forbidden')
ax2.legend()

plt.tight_layout()
plt.show()

print(f"phi at x = {x_end}: {phi[-1]:.2e}  (should be ~0 for an eigenstate)")

#%%

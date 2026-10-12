"""Classical 2D Ising: temperature sweep with exact (Onsager) reference curves.

Generates plots/quantities_of_interest.png
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipk

from ising import Ising2D

TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


def onsager_energy(T, J=1.0):
    """Exact energy per spin of the infinite lattice (B = 0)."""
    K = J / T
    k = 2.0 * np.sinh(2 * K) / np.cosh(2 * K) ** 2
    return -J / np.tanh(2 * K) * (1 + 2 / np.pi * (2 * np.tanh(2 * K) ** 2 - 1) * ellipk(k**2))


def onsager_magnetization(T, J=1.0):
    """Exact spontaneous magnetization per spin (zero above T_c)."""
    with np.errstate(invalid="ignore"):
        m = (1 - np.sinh(2 * J / T) ** -4.0) ** 0.125
    return np.where(T < TC, m, 0.0)


def main(N=16, sweeps=1500, seed=0):
    np.random.seed(seed)
    temperatures = np.linspace(1.5, 3.3, 32)
    res = {k: [] for k in ("energy", "abs_magnetization", "specific_heat", "susceptibility")}

    t0 = time.time()
    for T in temperatures:
        model = Ising2D(N=N, mu=0.0, J=1.0, B=0.0)
        model.simulate(num_steps=sweeps, temperature=T)
        for k in res:
            res[k].append(getattr(model, k))
    print(f"simulated {len(temperatures)} temperatures in {time.time() - t0:.0f}s")

    Tf = np.linspace(1.5, 3.3, 600)
    Tf = Tf[np.abs(Tf - TC) > 1e-3]
    e_exact = onsager_energy(Tf)
    c_exact = np.gradient(e_exact, Tf)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        ("energy", "Energy per spin", "IndianRed", (Tf, e_exact)),
        ("abs_magnetization", r"$\langle |m| \rangle$", "RoyalBlue", (Tf, onsager_magnetization(Tf))),
        ("specific_heat", "Specific heat per spin", "IndianRed", (Tf, c_exact)),
        ("susceptibility", r"Susceptibility $\chi$", "RoyalBlue", None),
    ]
    for ax, (key, label, color, exact) in zip(axes.ravel(), panels):
        ax.scatter(temperatures, res[key], s=35, color=color, label=f"Monte Carlo ({N}x{N})", zorder=3)
        if exact is not None:
            ax.plot(*exact, "k-", lw=1, label="Onsager (infinite lattice)")
        ax.axvline(TC, color="gray", ls="--", lw=1, label=r"$T_c \approx 2.269$")
        ax.set_xlabel("Temperature $T$ ($J/k_B$)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"2D Ising model, Metropolis Monte Carlo ({sweeps} sweeps per point)")
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/quantities_of_interest.png", dpi=130)
    print("saved plots/quantities_of_interest.png")


if __name__ == "__main__":
    main()

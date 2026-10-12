"""Quantum 2D transverse-field Ising model: generate all quantum plots.

    python quantum_demo.py            # full run (sizes up to 5x4 = 20 spins, a few minutes)
    python quantum_demo.py --quick    # small sizes only

Writes figures to plots/quantum/ and the raw scan data to results/quantum_scan.npz.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quantum_ising import (H_CRITICAL, TFIM2D, exact_thermodynamics, ftlm, krylov_evolve,
                           lanczos_ground_energy)

OUT = "plots/quantum"
SIZES_FULL = [(3, 3), (4, 3), (4, 4), (5, 4)]
SIZES_QUICK = [(3, 3), (4, 3), (4, 4)]
COLORS = plt.cm.viridis(np.linspace(0.0, 0.85, 4))


def label(size):
    return f"{size[0]}x{size[1]} (N={size[0] * size[1]})"


# ------------------------------------------------------------------ ground-state scan
def scan_ground_state(sizes, h_values):
    """Lanczos ground state (+ lowest odd level) for every size and field; warm started from the
    previous field value so each solve needs only a handful of Lanczos iterations."""
    data = {}
    for size in sizes:
        t0 = time.time()
        keys = ("E0", "sx", "m2", "binder", "gap", "S", "chiF")
        d = {k: np.zeros(len(h_values)) for k in keys}
        psi_prev = odd_prev = None
        psi_last = None
        for n, h in enumerate(h_values):
            model = TFIM2D(*size, J=1.0, h=h)
            w, V = model.lowest_states(k=1, parity=+1, v0=psi_prev)
            E0, psi = float(w[0]), V[:, 0]
            wo, Vo = model.lowest_states(k=1, parity=-1, v0=odd_prev)
            d["E0"][n] = E0 / model.N
            d["sx"][n] = model.expectation_sigma_x(psi)
            d["m2"][n] = model.order_parameter_moments(psi)[0]
            d["binder"][n] = model.binder_cumulant(psi)
            d["gap"][n] = wo[0] - E0
            d["S"][n] = model.entanglement_entropy(psi)
            if psi_last is not None:  # fidelity susceptibility between neighbouring grid points
                dh = h - h_values[n - 1]
                overlap = abs(np.vdot(psi_last, psi))
                d["chiF"][n] = -2.0 * np.log(min(overlap, 1.0)) / dh**2 / model.N
            psi_prev, odd_prev, psi_last = psi, Vo[:, 0], psi
        data[size] = d
        print(f"  {label(size)}: {len(h_values)} fields in {time.time() - t0:.0f}s", flush=True)
    return data


def crossing(h, y1, y2, lo, hi):
    """Field at which two curves cross inside [lo, hi] (linear interpolation), or nan."""
    sel = (h >= lo) & (h <= hi)
    hs, diff = h[sel], (y1 - y2)[sel]
    for k in range(len(hs) - 1):
        if diff[k] == 0 or diff[k] * diff[k + 1] < 0:
            return hs[k] - diff[k] * (hs[k + 1] - hs[k]) / (diff[k + 1] - diff[k])
    return np.nan


def mark_hc(ax):
    ax.axvline(H_CRITICAL, color="gray", ls="--", lw=1, label=r"$h_c \approx 3.044$ (QMC)")


def plot_phase_transition(data, h, sizes):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [
        ("E0", r"Ground-state energy $E_0 / N$", None),
        ("sx", r"Transverse magnetisation $\langle\sigma^x\rangle$", None),
        ("m2", r"Order parameter $\langle m_z^2 \rangle$", None),
        ("binder", r"Binder cumulant $U = 1 - \langle m^4\rangle / 3\langle m^2\rangle^2$", None),
        ("gap", r"Gap $\Delta = E_{\rm odd} - E_0$", "log"),
        ("S", "Half-system entanglement entropy $S$ (nats)", None),
    ]
    for ax, (key, ylabel, scale) in zip(axes.ravel(), panels):
        for c, size in zip(COLORS, sizes):
            ax.plot(h, data[size][key], "o-", ms=3, lw=1.2, color=c, label=label(size))
        mark_hc(ax)
        ax.set_xlabel(r"Transverse field $h / J$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if scale:
            ax.set_yscale(scale)
            ax.set_ylim(1e-8, 30)   # below ~1e-8 the exponentially small gap is lost in round-off
    axes[1, 2].axhline(np.log(2), color="k", ls=":", lw=1)
    axes[1, 2].text(0.2, np.log(2) * 1.02, r"$\ln 2$ (cat state)", fontsize=8)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("2D transverse-field Ising model: ground-state properties from Lanczos "
                 r"($H = -J\sum\sigma^z\sigma^z - h\sum\sigma^x$)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/ground_state_phase_transition.png", dpi=130)
    plt.close(fig)


def plot_scaling(data, h, sizes):
    """Fidelity susceptibility, gap x L crossing, and Krylov convergence."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    h_mid = 0.5 * (h[1:] + h[:-1])
    for c, size in zip(COLORS, sizes):
        ax.plot(h_mid, data[size]["chiF"][1:], "o-", ms=3, color=c, label=label(size))
    mark_hc(ax)
    ax.set_xlabel(r"Transverse field $h / J$")
    ax.set_ylabel(r"Fidelity susceptibility $\chi_F / N$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for c, size in zip(COLORS, sizes):
        L = np.sqrt(size[0] * size[1])
        ax.plot(h, L * data[size]["gap"], "o-", ms=3, color=c, label=label(size))
    mark_hc(ax)
    ax.set_xlabel(r"Transverse field $h / J$")
    ax.set_ylabel(r"$L \cdot \Delta$,  $L=\sqrt{N}$")
    ax.set_xlim(1.5, 5)
    ax.set_ylim(0, 8)
    ax.grid(alpha=0.3)
    ax.set_title("Gap crossing ($z=1$) locates $h_c$", fontsize=10)

    ax = axes[2]
    model_args = (4, 4)
    for h_val, c in zip((1.0, H_CRITICAL, 6.0), ("tab:blue", "tab:red", "tab:green")):
        model = TFIM2D(*model_args, h=h_val)
        E0, _ = model.ground_state()
        v0 = model._start_vector(+1, None, seed=0)      # random start in the P-even sector
        Ek = lanczos_ground_energy(model.operator(+1).matvec, v0, 80)
        ax.semilogy(np.arange(1, len(Ek) + 1), np.maximum(np.abs(Ek - E0) / abs(E0), 1e-16), "o-", ms=3,
                    color=c, label=f"h = {h_val:.2f}")
    ax.set_xlabel("Lanczos iterations (= matrix-vector products)")
    ax.set_ylabel(r"$|E_{\rm Ritz} - E_0| / |E_0|$")
    ax.set_title(r"Krylov convergence (even sector), $4\times4$ ($2^{16}$ states)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/finite_size_scaling_and_convergence.png", dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------- dynamics
def plot_quench_dynamics(size=(4, 4), h_finals=(1.0, H_CRITICAL, 6.0), t_max=4.0):
    """Quench from the fully polarised |up...up> state (h = 0 ground state) to field h_f,
    propagated with the adaptive Krylov integrator."""
    times = np.linspace(0, t_max, 161)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for h_f, c in zip(h_finals, ("tab:blue", "tab:red", "tab:green")):
        model = TFIM2D(*size, J=1.0, h=h_f)
        psi0 = model.up_state()

        def observe(psi, t):
            return (model.expectation_sigma_z(psi), model.expectation_sigma_x(psi),
                    -np.log(max(abs(psi[0]) ** 2, 1e-300)) / model.N)

        t0 = time.time()
        out, n_mv = krylov_evolve(model.matvec, psi0, times, m=24, tol=1e-9, observer=observe)
        out = np.array(out)
        print(f"  quench h_f={h_f:.2f}: {n_mv} matvecs, {time.time() - t0:.1f}s", flush=True)
        for ax, col in zip(axes, range(3)):
            ax.plot(times, out[:, col], color=c, lw=1.6, label=f"$h_f$ = {h_f:.2f}")
    for ax, yl in zip(axes, (r"$\langle\sigma^z\rangle(t)$", r"$\langle\sigma^x\rangle(t)$",
                             r"Loschmidt rate $\lambda(t) = -\ln|\langle\psi_0|\psi(t)\rangle|^2 / N$")):
        ax.set_xlabel("time $t$ ($\\hbar/J$)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Quench dynamics from the ordered state, {size[0]}x{size[1]} lattice "
                 "(Krylov time evolution)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/quench_dynamics.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- thermodynamics
def plot_thermodynamics(size=(4, 4), h_values=(0.5, 2.0, H_CRITICAL, 6.0), n_samples=60, m=80, n_low=30):
    T = np.logspace(-0.8, 1.3, 60)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for h, c in zip(h_values, plt.cm.plasma(np.linspace(0.05, 0.8, len(h_values)))):
        model = TFIM2D(*size, J=1.0, h=h)
        t0 = time.time()
        res = ftlm(model, T, n_samples=n_samples, m=m, seed=1, n_low=n_low)
        print(f"  FTLM h={h:.2f}: {time.time() - t0:.0f}s", flush=True)
        axes[0].semilogx(T, res["c"], color=c, lw=1.6, label=f"h = {h:.2f}")
        axes[1].semilogx(T, res["s"], color=c, lw=1.6, label=f"h = {h:.2f}")
    axes[1].axhline(np.log(2), color="k", ls=":", lw=1)
    axes[0].set_ylabel("Specific heat per site $c(T)$")
    axes[1].set_ylabel("Entropy per site $s(T)$  (max $\\ln 2$)")
    for ax in axes[:2]:
        ax.set_xlabel("Temperature $T / J$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_title(f"{size[0]}x{size[1]} lattice: FTLM ({n_samples} random vectors, {m} Lanczos steps, {n_low} exact low states)",
                      fontsize=10)

    # validation against exact diagonalisation on a lattice small enough for dense linear algebra
    small = TFIM2D(4, 3, J=1.0, h=2.0)
    ref = exact_thermodynamics(small.dense(), small.N, T)
    approx = ftlm(small, T, n_samples=n_samples, m=m, seed=1, n_low=n_low)
    ax = axes[2]
    ax.semilogx(T, ref["c"], "k-", lw=1.6, label="exact diagonalisation")
    ax.semilogx(T, approx["c"], "r--", lw=1.4, label="FTLM")
    ax.set_xlabel("Temperature $T / J$")
    ax.set_ylabel("Specific heat per site $c(T)$")
    ax.set_title("Validation: 4x3 lattice, $h=2$ (dim 4096)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/thermodynamics_ftlm.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small lattices only")
    parser.add_argument("--replot", action="store_true",
                        help="only redraw the ground-state figures from results/quantum_scan.npz")
    args = parser.parse_args()
    sizes = SIZES_QUICK if args.quick else SIZES_FULL
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    h = np.unique(np.concatenate([np.linspace(0.1, 2.4, 12), np.linspace(2.5, 3.7, 25),
                                  np.linspace(3.9, 6.0, 8)]))
    if args.replot:
        cache = np.load("results/quantum_scan.npz")
        h = cache["h"]
        keys = ("E0", "sx", "m2", "binder", "gap", "S", "chiF")
        data = {s: {k: cache[f"{s[0]}x{s[1]}_{k}"] for k in keys} for s in sizes}
    else:
        print("Ground-state scan (Lanczos)")
        data = scan_ground_state(sizes, h)
        np.savez("results/quantum_scan.npz", h=h,
                 **{f"{s[0]}x{s[1]}_{k}": v for s, d in data.items() for k, v in d.items()})
    plot_phase_transition(data, h, sizes)
    plot_scaling(data, h, sizes)

    # estimate of h_c from the crossing of neighbouring system sizes
    print("Estimates of h_c (literature: %.3f)" % H_CRITICAL)
    for a, b in zip(sizes[:-1], sizes[1:]):
        La, Lb = np.sqrt(a[0] * a[1]), np.sqrt(b[0] * b[1])
        xg = crossing(h, La * data[a]["gap"], Lb * data[b]["gap"], 2.0, 4.5)
        xb = crossing(h, data[a]["binder"], data[b]["binder"], 2.0, 4.5)
        print(f"  {label(a)} / {label(b)}: gap crossing {xg:.3f}, Binder crossing {xb:.3f}")

    if args.replot:
        return
    print("Quench dynamics (Krylov propagation)")
    plot_quench_dynamics()
    print("Thermodynamics (FTLM)")
    plot_thermodynamics()
    print("done")


if __name__ == "__main__":
    main()


"""Quantum 2D transverse-field Ising model (TFIM) solved with Krylov-subspace methods.

    H = -J * sum_<ij> sigma^z_i sigma^z_j  -  h * sum_i sigma^x_i

on an Lx x Ly periodic square lattice (Pauli matrices, hbar = 1). For h = 0 this is the
classical Ising model; for h >> J the spins polarise along x. In 2D the ground state has a
quantum phase transition at h_c / J ~ 3.044 (quantum Monte Carlo), the (2+1)D Ising universality
class.

Everything is *matrix free*: a state is a vector of 2^N amplitudes, the Hamiltonian is never
stored, and one application of H costs O(N 2^N). This is what makes the Krylov methods below
efficient:

* ground state and low-lying levels  -> Lanczos (implicitly restarted, ``scipy.sparse.linalg.eigsh``)
* real-time dynamics exp(-iHt)|psi>  -> own Krylov/Lanczos propagator (`krylov_evolve`)
* thermodynamics at finite T         -> Finite-Temperature Lanczos Method (`ftlm`)

Conventions
-----------
Site i = x * Ly + y is tensor axis i of the state, i.e. bit (N - 1 - i) of the basis index
(C-ordered reshape). Spin up (sigma^z = +1) is bit 0.
"""
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

H_CRITICAL = 3.04438  # QMC value of h_c / J for the 2D square-lattice TFIM (Bloete & Deng 2002)


class TFIM2D(object):
    def __init__(self, Lx: int, Ly: int, J: float = 1.0, h: float = 1.0) -> None:
        if Lx < 1 or Ly < 1:
            raise ValueError("lattice dimensions must be positive")
        if Lx * Ly > 26:
            raise ValueError("lattice too large for a dense state vector")
        self.Lx, self.Ly, self.N = Lx, Ly, Lx * Ly
        self.J, self.h = J, h
        self.dim = 2 ** self.N
        self.bonds = self._build_bonds()

        # sigma^z of every site for every basis state, and the (h independent) diagonal.
        idx = np.arange(self.dim, dtype=np.int64)
        self._sz = [(1 - 2 * ((idx >> (self.N - 1 - i)) & 1)).astype(np.int8) for i in range(self.N)]
        self.mz = np.sum(self._sz, axis=0, dtype=np.int16).astype(np.float64)  # total sigma^z
        zz = np.zeros(self.dim)
        for i, j in self.bonds:
            zz += self._sz[i] * self._sz[j]
        self.zz = zz                      # sum_<ij> sz_i sz_j
        self.n_A = (Lx // 2) * Ly if Lx > 1 else self.N // 2  # sites in the entanglement cut

    def _build_bonds(self) -> List[Tuple[int, int]]:
        """Unique nearest-neighbour bonds (an L = 2 periodic direction gives a single bond)."""
        bonds = set()
        for x in range(self.Lx):
            for y in range(self.Ly):
                i = x * self.Ly + y
                for j in (((x + 1) % self.Lx) * self.Ly + y, x * self.Ly + (y + 1) % self.Ly):
                    if i != j:
                        bonds.add((min(i, j), max(i, j)))
        return sorted(bonds)

    # ------------------------------------------------------------------ operators
    def sigma_x_sum(self, v: np.ndarray) -> np.ndarray:
        """Apply sum_i sigma^x_i. Flipping spin i is a reversal of axis i of the state tensor."""
        v = np.ascontiguousarray(v).reshape(-1)
        out = np.zeros_like(v)
        for i in range(self.N):
            shape = (1 << i, 2, 1 << (self.N - 1 - i))
            out.reshape(shape)[...] += v.reshape(shape)[:, ::-1, :]
        return out

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """H |v> without ever forming H."""
        v = np.asarray(v).reshape(-1)
        return -self.J * self.zz * v - self.h * self.sigma_x_sum(v)

    def energy_bound(self) -> float:
        """Upper bound on ||H|| (used to push unwanted symmetry sectors out of the spectrum)."""
        return abs(self.J) * len(self.bonds) + abs(self.h) * self.N + 1.0

    def operator(self, parity: Optional[int] = None) -> spla.LinearOperator:
        """H as a ``LinearOperator``. With parity = +1 / -1 the operator acts on the even / odd
        sector of the global spin flip P = prod_i sigma^x_i (which commutes with H and is just
        a reversal of the vector). Vectors from the other sector are shifted above the spectrum."""
        if parity is None:
            return spla.LinearOperator((self.dim, self.dim), matvec=self.matvec, dtype=np.float64)
        shift = self.energy_bound()

        def sector_matvec(v):
            v = np.asarray(v).reshape(-1)
            p = 0.5 * (v + parity * v[::-1])
            return self.matvec(p) + shift * (v - p)

        return spla.LinearOperator((self.dim, self.dim), matvec=sector_matvec, dtype=np.float64)

    def dense(self) -> np.ndarray:
        """Full dense Hamiltonian (small N only; for testing and exact reference)."""
        if self.N > 13:
            raise ValueError("dense matrix only for N <= 13")
        eye = np.eye(self.dim)
        return np.column_stack([self.matvec(eye[:, k]) for k in range(self.dim)])

    # ------------------------------------------------------------- ground state (Lanczos)
    def _start_vector(self, parity: Optional[int], v0: Optional[np.ndarray], seed: int) -> np.ndarray:
        if v0 is None:
            v0 = np.random.default_rng(seed).standard_normal(self.dim)
        v0 = np.asarray(v0, dtype=np.float64).reshape(-1)
        if parity is not None:
            v0 = 0.5 * (v0 + parity * v0[::-1])
        return v0 / np.linalg.norm(v0)

    def lowest_states(self, k: int = 1, parity: Optional[int] = None, v0: Optional[np.ndarray] = None,
                      tol: float = 1e-10, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """k lowest eigenpairs (optionally within one parity sector) by Lanczos."""
        if self.dim <= 64 or k >= self.dim - 1:
            H = self.dense()
            if parity is not None:
                Q = 0.5 * (np.eye(self.dim) + parity * np.eye(self.dim)[::-1])  # sector projector
                H = Q @ H @ Q + self.energy_bound() * (np.eye(self.dim) - Q)
            w, V = la.eigh(H)
            return w[:k], V[:, :k]
        v0 = self._start_vector(parity, v0, seed)
        for attempt in range(6):
            # ARPACK can break down on (near-)degenerate spectra; retry with a larger Krylov
            # subspace and a different start vector.
            v_start = v0 if attempt == 0 else self._start_vector(parity, None, seed + attempt)
            try:
                w, V = spla.eigsh(self.operator(parity), k=k, which="SA", v0=v_start, tol=tol,
                                  ncv=min(self.dim - 1, max(2 * k + 1, 20) + 20 * attempt))
                break
            except spla.ArpackError:
                if attempt == 5:
                    raise
        order = np.argsort(w)
        return w[order], V[:, order]

    def ground_state(self, v0: Optional[np.ndarray] = None, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        """Ground state. It is always in the P-even sector (Perron-Frobenius for h > 0); working
        in that sector avoids returning an arbitrary mix of the two near-degenerate
        'cat-state' levels in the ordered phase."""
        w, V = self.lowest_states(k=1, parity=+1, v0=v0, tol=tol)
        return float(w[0]), V[:, 0]

    def gap(self, E0: Optional[float] = None, v0: Optional[np.ndarray] = None, tol: float = 1e-10) -> float:
        """Gap to the lowest P-odd state. This is the true gap (E1 - E0); it closes exponentially
        with N in the ordered phase and opens linearly with h in the paramagnet."""
        if E0 is None:
            E0, _ = self.ground_state()
        w, _ = self.lowest_states(k=1, parity=-1, v0=v0, tol=tol)
        return float(w[0] - E0)

    # ----------------------------------------------------------------- observables
    def expectation_sigma_x(self, psi: np.ndarray) -> float:
        """<sigma^x> per site."""
        return float(np.real(np.vdot(psi, self.sigma_x_sum(psi))) / self.N)

    def expectation_sigma_z(self, psi: np.ndarray) -> float:
        """<sigma^z> per site (zero in the symmetric ground state)."""
        return float(np.sum(np.abs(psi) ** 2 * self.mz) / self.N)

    def order_parameter_moments(self, psi: np.ndarray) -> Tuple[float, float, float]:
        """(<m^2>, <m^4>, <|m|>) of the magnetisation per site m = sum_i sigma^z_i / N."""
        p = np.abs(psi) ** 2
        m = self.mz / self.N
        return float(np.sum(p * m**2)), float(np.sum(p * m**4)), float(np.sum(p * np.abs(m)))

    def binder_cumulant(self, psi: np.ndarray) -> float:
        m2, m4, _ = self.order_parameter_moments(psi)
        return 1.0 - m4 / (3.0 * m2**2)

    def zz_correlation(self, psi: np.ndarray, i: int, j: int) -> float:
        return float(np.sum(np.abs(psi) ** 2 * self._sz[i] * self._sz[j]))

    def entanglement_entropy(self, psi: np.ndarray) -> float:
        """Von Neumann entropy (nats) of the first `n_A` sites (whole columns of the lattice)."""
        s = la.svd(np.reshape(psi, (2 ** self.n_A, -1)), compute_uv=False)
        lam = s**2
        lam = lam[lam > 1e-14]
        return float(-np.sum(lam * np.log(lam)))

    def up_state(self) -> np.ndarray:
        """Fully polarised |up up ... up> (a symmetry-broken ground state at h = 0)."""
        psi = np.zeros(self.dim)
        psi[0] = 1.0
        return psi

    def x_state(self) -> np.ndarray:
        """Fully x-polarised state (the ground state for J = 0)."""
        return np.full(self.dim, self.dim ** -0.5)


# ---------------------------------------------------------------------------------------
# Generic Krylov routines
# ---------------------------------------------------------------------------------------
def lanczos(matvec: Callable[[np.ndarray], np.ndarray], v0: np.ndarray, m: int,
            reorthogonalize: bool = True):
    """m-step Lanczos tridiagonalisation  H V = V T + beta_m v_{m+1} e_m^T.

    Returns (alpha, beta, V) with T = tridiag(beta, alpha, beta), V of shape (k, dim) holding the
    orthonormal Krylov basis, and beta[-1] the residual coupling (0 on a happy breakdown).
    Full reorthogonalisation keeps the basis orthogonal at the price of O(m^2 dim) work, which is
    negligible next to the m matrix-vector products for the Hamiltonians used here.
    """
    dtype = np.result_type(v0.dtype, np.float64)
    V = np.zeros((m, v0.size), dtype=dtype)
    V[0] = v0 / np.linalg.norm(v0)
    alpha, beta = [], []
    w_prev = None
    for k in range(m):
        w = matvec(V[k])
        a = float(np.real(np.vdot(V[k], w)))
        w = w - a * V[k]
        if k > 0:
            w = w - beta[-1] * V[k - 1]
        if reorthogonalize:
            w = w - V[: k + 1].T @ (V[: k + 1].conj() @ w)
        b = float(np.linalg.norm(w))
        alpha.append(a)
        beta.append(b)
        if b < 1e-12 or k == m - 1:
            break
        V[k + 1] = w / b
    n = len(alpha)
    return np.array(alpha), np.array(beta), V[:n]


def lanczos_ground_energy(matvec, v0: np.ndarray, m: int) -> np.ndarray:
    """Lowest Ritz value after each Lanczos step 1..m (for convergence studies)."""
    alpha, beta, _ = lanczos(matvec, v0, m)
    return np.array([la.eigvalsh_tridiagonal(alpha[:k], beta[: k - 1])[0] for k in range(1, len(alpha) + 1)])


def krylov_expm_step(matvec, psi: np.ndarray, dt: float, m: int = 24, tol: float = 1e-10):
    """One step psi -> exp(-i H dt) psi in an m-dimensional Krylov space.

    exp(-iH dt) psi ~= ||psi|| V exp(-i T dt) e_1. Returns (psi_new, error_estimate); the
    estimate is the standard a-posteriori bound beta_m |[exp(-iT dt)]_{m,1}|.
    """
    norm = np.linalg.norm(psi)
    alpha, beta, V = lanczos(matvec, psi.astype(np.complex128), m)
    k = len(alpha)
    if k == 1:
        return np.exp(-1j * alpha[0] * dt) * psi, 0.0
    w, U = la.eigh_tridiagonal(alpha, beta[: k - 1])
    coeff = U @ (np.exp(-1j * w * dt) * U[0])
    err = abs(beta[k - 1] * coeff[k - 1]) if beta[k - 1] > 1e-12 else 0.0
    return norm * (V.T @ coeff), err


def krylov_evolve(matvec, psi0: np.ndarray, times: Sequence[float], m: int = 24, tol: float = 1e-9,
                  observer: Optional[Callable[[np.ndarray, float], object]] = None,
                  max_dt: float = 0.5):
    """Propagate psi0 under exp(-iHt) and sample `observer(psi, t)` at every time in `times`
    (a sorted array starting at 0). Substeps are chosen adaptively so each Krylov step meets
    `tol`. Returns (list of observer values, number of matvecs used)."""
    psi = psi0.astype(np.complex128)
    t = 0.0
    out, n_mv = [], 0
    for t_target in times:
        while t < t_target - 1e-14:
            dt = min(t_target - t, max_dt)
            while True:
                new, err = krylov_expm_step(matvec, psi, dt, m)
                n_mv += m
                if err < tol * dt or dt < 1e-6:
                    break
                dt *= 0.5
            psi, t = new, t + dt
        out.append(observer(psi, t_target) if observer else psi.copy())
    return out, n_mv


def ftlm(model: TFIM2D, temperatures: np.ndarray, n_samples: int = 16, m: int = 60, seed: int = 0,
         n_low: int = 0):
    """Finite-Temperature Lanczos Method.

    Z(T) = Tr e^{-H/T} is estimated by stochastic trace evaluation, replacing each random vector
    |r> by an m-step Lanczos run:  Z ~ (1/R) sum_r |r|^2 sum_j |<r|phi_j>|^2 e^{-eps_j / T},
    with eps_j, phi_j the eigenpairs of the tridiagonal matrix. Returns a dict of per-site
    energy `e`, specific heat `c`, free energy `f` and entropy `s` (k_B = 1).

    The stochastic error is relative to the *whole* partition function, which is large when
    only a few states matter (low T). With `n_low > 0` the `n_low` lowest eigenpairs are computed
    exactly by Lanczos and summed explicitly; the random vectors are projected onto their
    orthogonal complement so only the high-energy part of the trace is sampled. This removes the
    low-temperature noise (e.g. slightly negative entropies) at negligible extra cost.
    """
    rng = np.random.default_rng(seed)
    temperatures = np.asarray(temperatures, dtype=float)

    if n_low > 0:
        low_w, low_V = model.lowest_states(k=n_low, tol=1e-9)
        # keep only converged eigenpairs; any state left out is simply sampled stochastically
        resid = np.array([np.linalg.norm(model.matvec(low_V[:, k]) - low_w[k] * low_V[:, k])
                          for k in range(len(low_w))])
        good = resid < 1e-6
        low_w, low_V = low_w[good], low_V[:, good].T      # (n_low, dim), orthonormal rows

        def project(v):
            return v - low_V.T @ (low_V @ v)

        def matvec(v):
            return project(model.matvec(project(v)))
    else:
        low_w, low_V = np.empty(0), None
        project = lambda v: v
        matvec = model.matvec

    ritz, weights = [], []
    for _ in range(n_samples):
        r = project(rng.choice([-1.0, 1.0], size=model.dim))  # Rademacher vector, E[r r^T] = 1
        norm2 = float(r @ r)
        alpha, beta, _ = lanczos(matvec, r, m)
        w, U = la.eigh_tridiagonal(alpha, beta[: len(alpha) - 1])
        ritz.append(w)
        weights.append(norm2 / n_samples * U[0] ** 2)      # weight of each Ritz value in Tr(...)
    N = model.N
    e, c, f, s = [], [], [], []
    e_min = min(min(w.min() for w in ritz), low_w.min() if n_low > 0 else np.inf)
    for T in temperatures:
        z = z1 = z2 = 0.0
        for w, wt in zip(ritz, weights):
            boltz = wt * np.exp(-(w - e_min) / T)
            z += boltz.sum()
            z1 += (boltz * w).sum()
            z2 += (boltz * w**2).sum()
        boltz = np.exp(-(low_w - e_min) / T)
        z += boltz.sum()
        z1 += (boltz * low_w).sum()
        z2 += (boltz * low_w**2).sum()
        E, E2 = z1 / z, z2 / z
        ln_z = np.log(z) - e_min / T
        e.append(E / N)
        c.append((E2 - E**2) / (N * T**2))
        f.append(-T * ln_z / N)
        s.append((E / T + ln_z) / N)
    return dict(T=temperatures, e=np.array(e), c=np.array(c), f=np.array(f), s=np.array(s))


def exact_thermodynamics(H: np.ndarray, N: int, temperatures: np.ndarray):
    """Reference thermodynamics from full diagonalisation (small N)."""
    w = la.eigvalsh(H)
    e, c, s = [], [], []
    for T in temperatures:
        p = np.exp(-(w - w[0]) / T)
        z = p.sum()
        p /= z
        E, E2 = (p * w).sum(), (p * w**2).sum()
        e.append(E / N)
        c.append((E2 - E**2) / (N * T**2))
        s.append((E / T + np.log(z) - w[0] / T) / N)
    return dict(T=np.asarray(temperatures), e=np.array(e), c=np.array(c), s=np.array(s))

import numpy as np
import pytest
import scipy.linalg as la

from quantum_ising import (TFIM2D, exact_thermodynamics, ftlm, krylov_evolve,
                           krylov_expm_step, lanczos, lanczos_ground_energy)


def test_bonds():
    assert len(TFIM2D(3, 3).bonds) == 18          # 2 N bonds on a periodic torus
    assert len(TFIM2D(4, 4).bonds) == 32
    assert len(TFIM2D(2, 2).bonds) == 4           # L = 2: wrap bond coincides with direct bond
    assert len(TFIM2D(6, 1).bonds) == 6           # a periodic chain


def test_matvec_matches_dense_and_is_symmetric():
    m = TFIM2D(3, 3, J=1.0, h=0.7)
    H = m.dense()
    assert np.allclose(H, H.T)
    v = np.random.default_rng(1).standard_normal(m.dim)
    assert np.allclose(H @ v, m.matvec(v))


def test_two_site_analytic():
    """Two spins, one bond: even sector [[-J, -2h], [-2h, J]] -> E0 = -sqrt(J^2 + 4 h^2)."""
    m = TFIM2D(2, 1, J=1.3, h=0.8)
    w, _ = m.lowest_states(k=1)
    assert w[0] == pytest.approx(-np.sqrt(1.3**2 + 4 * 0.8**2))


@pytest.mark.parametrize("h", [0.4, 1.0, 2.5])
def test_chain_matches_free_fermion_solution(h):
    """Ly = 1 gives the 1D transverse-field Ising ring, solvable by Jordan-Wigner:
    E0 = -sum_k sqrt(J^2 + h^2 - 2 J h cos k), k = (2n+1) pi / N (even-parity sector)."""
    N, J = 12, 1.0
    m = TFIM2D(N, 1, J=J, h=h)
    E0, _ = m.ground_state()
    k = (2 * np.arange(N) + 1) * np.pi / N
    assert E0 == pytest.approx(-np.sum(np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k))), abs=1e-8)


def test_lanczos_vs_dense_2d():
    m = TFIM2D(3, 3, h=2.0)
    ref = la.eigvalsh(m.dense())
    E0, psi = m.ground_state()
    assert E0 == pytest.approx(ref[0], abs=1e-9)
    # Hellmann-Feynman: <sigma^x> = -(1/N) dE0/dh
    d = 1e-4
    Ep, _ = TFIM2D(3, 3, h=2.0 + d).ground_state()
    Em, _ = TFIM2D(3, 3, h=2.0 - d).ground_state()
    assert m.expectation_sigma_x(psi) == pytest.approx(-(Ep - Em) / (2 * d) / m.N, abs=1e-6)
    # residual of the eigenvector
    assert np.linalg.norm(m.matvec(psi) - E0 * psi) < 1e-7


def test_odd_sector_gap_and_parity():
    m = TFIM2D(3, 3, h=1.5)
    ref = la.eigvalsh(m.dense())
    E0, psi = m.ground_state()
    assert np.allclose(psi, psi[::-1], atol=1e-8)           # ground state is P-even
    gap = m.gap(E0)
    # lowest odd-parity level must be one of the exact eigenvalues above E0
    assert np.min(np.abs(ref - (E0 + gap))) < 1e-8
    assert gap > 0


def test_limits_h0_and_J0():
    m0 = TFIM2D(4, 3, J=1.0, h=0.0)
    E0, psi = m0.ground_state()
    assert E0 / m0.N == pytest.approx(-2.0)                  # classical ground state energy
    mJ = TFIM2D(3, 3, J=0.0, h=1.7)
    E0, psi = mJ.ground_state()
    assert E0 / mJ.N == pytest.approx(-1.7)
    assert mJ.expectation_sigma_x(psi) == pytest.approx(1.0)
    # in the h = 0 ground-state manifold the cat state has <m^2> = 1, Binder U = 2/3
    _, psi0 = m0.ground_state()
    m2, m4, _ = m0.order_parameter_moments(psi0)
    assert m2 == pytest.approx(1.0, abs=1e-6)
    assert m0.binder_cumulant(psi0) == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_entanglement_limits():
    m = TFIM2D(4, 2, J=1.0, h=0.0)
    _, psi = m.ground_state()
    assert m.entanglement_entropy(psi) == pytest.approx(np.log(2.0), abs=1e-6)   # GHZ cat state
    m = TFIM2D(4, 2, J=0.0, h=1.0)
    _, psi = m.ground_state()
    assert m.entanglement_entropy(psi) == pytest.approx(0.0, abs=1e-8)           # product state


def test_lanczos_tridiagonal_relation():
    m = TFIM2D(3, 2, h=1.0)
    v0 = np.random.default_rng(0).standard_normal(m.dim)
    alpha, beta, V = lanczos(m.matvec, v0, 12)
    T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
    assert np.allclose(V @ V.T, np.eye(len(alpha)), atol=1e-10)
    assert np.allclose(V @ m.dense() @ V.T, T, atol=1e-9)


def test_lanczos_convergence():
    m = TFIM2D(3, 3, h=3.0)
    ref = la.eigvalsh(m.dense())[0]
    v0 = np.random.default_rng(0).standard_normal(m.dim)
    E = lanczos_ground_energy(m.matvec, v0, 60)
    assert np.all(np.diff(E) < 1e-9)                          # Ritz value decreases monotonically
    assert E[-1] == pytest.approx(ref, abs=1e-8)


def test_krylov_time_evolution_matches_expm():
    m = TFIM2D(3, 2, J=1.0, h=1.3)
    H = m.dense()
    psi0 = m.up_state()
    times = np.linspace(0, 3.0, 7)
    states, n_mv = krylov_evolve(m.matvec, psi0, times, m=20)
    for t, psi in zip(times, states):
        assert np.allclose(psi, la.expm(-1j * H * t) @ psi0, atol=1e-8)
        assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-10)
        assert np.real(np.vdot(psi, H @ psi)) == pytest.approx(np.real(np.vdot(psi0, H @ psi0)), abs=1e-8)


def test_krylov_step_error_estimate_is_reliable():
    m = TFIM2D(3, 3, h=1.0)
    psi0 = m.up_state()
    exact = la.expm(-1j * m.dense() * 0.4) @ psi0
    new, err = krylov_expm_step(m.matvec, psi0, 0.4, m=8)
    assert np.linalg.norm(new - exact) < 10 * err + 1e-12


def test_ftlm_matches_exact_diagonalisation():
    m = TFIM2D(4, 3, h=2.0)                                  # dim 4096
    T = np.array([0.5, 1.0, 2.0, 5.0, 20.0])
    ref = exact_thermodynamics(m.dense(), m.N, T)
    res = ftlm(m, T, n_samples=300, m=50, seed=3)   # stochastic trace: error ~ 1/sqrt(n_samples)
    assert np.allclose(res["e"], ref["e"], atol=0.02)
    assert np.allclose(res["c"], ref["c"], atol=0.03)
    assert np.allclose(res["s"], ref["s"], atol=0.02)


def test_ftlm_low_lying_deflation_fixes_low_temperature():
    """Exact treatment of the lowest states removes the low-T stochastic noise."""
    m = TFIM2D(4, 3, h=2.0)
    T = np.array([0.15, 0.3, 0.6, 1.0, 3.0])
    ref = exact_thermodynamics(m.dense(), m.N, T)
    res = ftlm(m, T, n_samples=40, m=50, seed=5, n_low=30)
    assert np.allclose(res["e"], ref["e"], atol=0.01)
    assert np.allclose(res["c"], ref["c"], atol=0.02)
    assert np.allclose(res["s"], ref["s"], atol=0.01)
    assert np.all(res["s"] > -1e-3)                 # entropy cannot be negative

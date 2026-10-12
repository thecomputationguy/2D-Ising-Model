import itertools

import numpy as np
import pytest

from ising import Ising2D


def exact_averages(N, T, J=1.0, field=0.0):
    """Brute-force canonical averages for an N x N periodic lattice (field = mu*B)."""
    S = np.array(list(itertools.product([-1, 1], repeat=N * N))).reshape(-1, N, N)
    nb = np.roll(S, 1, 1) + np.roll(S, -1, 1) + np.roll(S, 1, 2) + np.roll(S, -1, 2)
    E = -0.5 * J * (S * nb).sum((1, 2)) - field * S.sum((1, 2))
    M = S.sum((1, 2))
    w = np.exp(-(E - E.min()) / T)
    w /= w.sum()
    n = N * N
    Em, E2 = (w * E).sum(), (w * E**2).sum()
    Ma, M2 = (w * np.abs(M)).sum(), (w * M**2).sum()
    return dict(
        e=Em / n,
        C=(E2 - Em**2) / (n * T**2),
        m=Ma / n,
        chi=(M2 - Ma**2) / (n * T),
    )


def test_ground_state_and_checkerboard_energy():
    m = Ising2D(N=6, mu=0.0, J=1.5, B=0.0, initial="up")
    assert m._calculate_energy() / 36 == pytest.approx(-2 * 1.5)
    m.system[:] = np.indices((6, 6)).sum(0) % 2 * 2 - 1
    assert m._calculate_energy() / 36 == pytest.approx(+2 * 1.5)


def test_field_term_counted_once():
    m = Ising2D(N=4, mu=0.5, J=1.0, B=2.0, initial="up")
    assert m._calculate_energy() / 16 == pytest.approx(-2.0 - 0.5 * 2.0)


@pytest.mark.parametrize("N", [2, 3, 6])
def test_delta_energy_matches_energy_functional(N):
    """The Metropolis flip energy must equal the difference of the total energy."""
    np.random.seed(N)
    m = Ising2D(N=N, mu=0.3, J=1.0, B=0.7)
    for _ in range(25):
        x, y = np.random.randint(N, size=2)
        dE = m._delta_energy(x, y)
        before = m._calculate_energy()
        m.system[x, y] *= -1
        assert m._calculate_energy() - before == pytest.approx(dE)


def test_cold_start_low_temperature():
    np.random.seed(0)
    m = Ising2D(N=8, mu=0.0, J=1.0, B=0.0, initial="up")
    m.simulate(num_steps=200, temperature=0.3)
    assert m.abs_magnetization > 0.99
    assert m.energy == pytest.approx(-2.0, abs=0.05)


def test_matches_exact_enumeration_4x4():
    np.random.seed(1)
    T = 2.5
    ref = exact_averages(4, T)
    m = Ising2D(N=4, mu=0.0, J=1.0, B=0.0)
    m.simulate(num_steps=30000, temperature=T)
    assert m.energy == pytest.approx(ref["e"], abs=0.02)
    assert m.abs_magnetization == pytest.approx(ref["m"], abs=0.02)
    assert m.specific_heat == pytest.approx(ref["C"], rel=0.07)
    assert m.susceptibility == pytest.approx(ref["chi"], rel=0.07)


def test_matches_exact_enumeration_with_field():
    np.random.seed(2)
    T, mu, B = 3.0, 0.5, 1.0
    ref = exact_averages(3, T, field=mu * B)
    m = Ising2D(N=3, mu=mu, J=1.0, B=B)
    m.simulate(num_steps=40000, temperature=T)
    assert m.energy == pytest.approx(ref["e"], abs=0.03)
    assert m.specific_heat == pytest.approx(ref["C"], rel=0.1)


def test_invalid_arguments():
    m = Ising2D(N=4)
    with pytest.raises(ValueError):
        m.simulate(num_steps=10, temperature=0.0)
    with pytest.raises(ValueError):
        m.simulate(num_steps=0, temperature=1.0)
    with pytest.raises(ValueError):
        Ising2D(N=4, initial="bogus")

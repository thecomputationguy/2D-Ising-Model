from typing import Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

class Ising2D(object):
    """Classical 2D Ising model on an N x N periodic lattice (k_B = 1).

    Hamiltonian:  H = -J * sum_<ij> s_i s_j  -  mu * B * sum_i s_i,   s_i = +/-1

    All reported quantities are *per spin*. Note that the default couplings
    (J=0.2, mu=0.33, B=1) include a magnetic field, which smears out the phase
    transition; use ``J=1, mu=0, B=0`` for the textbook model with
    T_c = 2 / ln(1 + sqrt(2)) ~ 2.269.

    ``magnetization`` is the signed mean <M>/N^2. For B = 0 the two ordered
    states are degenerate, so the signed mean is ~0 over a long run and
    ``abs_magnetization`` = <|M|>/N^2 is the meaningful order parameter.
    """
    def __init__(self, N, mu = 0.33, J = 0.2, B = 1.0, initial = "random") -> None:
        self.N = N
        self.T = 0.0
        self.step = 0
        self.mu = mu
        self.J = J
        self.B = B
        if initial == "random":
            self.system = 2 * np.random.randint(2, size=(self.N, self.N)) - 1
        elif initial == "up":
            self.system = np.ones((self.N, self.N), dtype=int)
        else:
            raise ValueError("initial must be 'random' or 'up'")
        self.energy = 0.0
        self.magnetization = 0.0
        self.abs_magnetization = 0.0
        self.specific_heat = 0.0
        self.susceptibility = 0.0

    def _neighbor_sum(self, x, y) -> int:
        N = self.N
        return (self.system[(x+1)%N, y] + self.system[x, (y+1)%N]
                + self.system[(x-1)%N, y] + self.system[x, (y-1)%N])

    def _delta_energy(self, x, y) -> float:
        """Energy change of flipping spin (x, y):  dE = 2 s (J * sum_nn + mu * B)."""
        site = self.system[x, y]
        return 2 * ((self.B * self.mu * site) + (self.J * site * self._neighbor_sum(x, y)))

    def _metropolis_hastings(self, temperature) -> None:
        xs = np.random.randint(0, self.N, size=self.N**2)
        ys = np.random.randint(0, self.N, size=self.N**2)
        rs = np.random.rand(self.N**2)

        for x, y, r in zip(xs, ys, rs):
            energy = self._delta_energy(x, y)

            if energy <= 0 or r < np.exp(-energy / temperature):
                self.system[x, y] *= -1

    def plot_system(self) -> None:
        os.makedirs("plots", exist_ok=True)
        plt.figure()
        plt.imshow(self.system, interpolation='nearest')
        plt.title(f"Time={self.step}, Temperature={self.T}K")
        plt.savefig(f"plots/time_{self.step}.png")
        plt.show()

    def _calculate_energy(self) -> float:
        """Total energy. Summing s_i * (sum of its 4 neighbours) over all sites counts every
        bond twice, hence the factor 1/2 on the exchange term (the field term is counted once)."""
        s = self.system
        neighbors = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
        return float(-0.5 * self.J * np.sum(s * neighbors) - self.B * self.mu * np.sum(s))

    def _calculate_magnetization(self) -> np.ndarray:
        return np.sum(self.system)

    def _equilibrate_system(self, num_steps, temperature) -> None:
        for _ in range(num_steps):
            self._metropolis_hastings(temperature=temperature)

    def simulate(self, num_steps, temperature, plot = False) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        self._equilibrate_system(num_steps= int(0.2 * num_steps), temperature=temperature)
        E = M = AbsM = E_squared = M_squared = 0.0
        self.T = temperature

        for i in range(num_steps):
            self._metropolis_hastings(temperature=temperature)
            energy = self._calculate_energy()
            magnetization = self._calculate_magnetization()
            E += energy
            M += magnetization
            AbsM += abs(magnetization)
            E_squared += energy**2
            M_squared += magnetization**2 

            if plot == True and (i % max(1, num_steps // 4) == 0):
                self.plot_system()

            self.step += 1

        # Sample means <X> = (sum of X) / num_steps, then everything per spin (divide by N^2).
        n_spins = self.N**2
        E, M, AbsM = E / num_steps, M / num_steps, AbsM / num_steps
        E_squared, M_squared = E_squared / num_steps, M_squared / num_steps

        self.energy = E / n_spins
        self.magnetization = M / n_spins
        self.abs_magnetization = AbsM / n_spins
        # C = (<E^2> - <E>^2) / (N^2 T^2)  and  chi = (<M^2> - <|M|>^2) / (N^2 T)
        self.specific_heat = (E_squared - E**2) / (n_spins * temperature**2)
        self.susceptibility = (M_squared - AbsM**2) / (n_spins * temperature)

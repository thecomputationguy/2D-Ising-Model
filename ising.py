from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

class Ising2D(object):
    def __init__(self, N, mu = 0.33, J = 0.2, B = 1.0):
        self.N = N
        self.T = 0.0
        self.step = 0
        self.mu = mu
        self.J = J
        self.B = B
        self.system = 2 * np.random.randint(2, size=(self.N, self.N)) - 1
        self.energy = 0.0
        self.magnetization = 0.0
        self.specific_heat = 0.0
        self.susceptibility = 0.0
    
    def _metropolis_hastings(self, temperature):
        for _ in range(self.N**2):
            x = np.random.randint(0,self.N)
            y = np.random.randint(0,self.N)

            site = self.system[x,y]
            neighbors = self.system[(x+1)%self.N, y] + self.system[x, (y+1)%self.N] + self.system[(x-1)%self.N, y] + self.system[x, (y-1)%self.N]
            energy = 2 * ((self.B * self.mu * site) + (self.J * site * neighbors))

            if energy < 0:
                site *= -1
            elif np.random.rand() < np.exp(-energy * 1.0/temperature):
                site *= -1

            self.system[x,y] = site

    def plot_system(self):
        plt.figure()
        plt.imshow(self.system, interpolation='nearest')
        plt.title(f"Time={self.step}, Temperature={self.T}K")
        plt.savefig(f"plots/time_{self.step}.png")
        plt.show()

    def _calculate_energy(self):
        energy = 0.0
        for i in range(self.N):
            for j in range(self.N):
                site = self.system[i,j]
                neighbors = self.system[(i+1)%self.N, j] + self.system[i, (j+1)%self.N] + self.system[(i-1)%self.N, j] + self.system[i, (j-1)%self.N]
                energy += -((self.B * self.mu * site) + (self.J * site * neighbors))

        return energy / 4

    def _calculate_magnetization(self):
        return np.sum(self.system)

    def _equilibrate_system(self, num_steps, temperature):
        for _ in range(num_steps):
            self._metropolis_hastings(temperature=temperature)

    def simulate(self, num_steps, temperature, plot = False):
        self._equilibrate_system(num_steps= int(0.2 * num_steps), temperature=temperature)
        E = M = E_squared = M_squared = 0.0
        self.T = temperature

        for i in range(num_steps):
            self._metropolis_hastings(temperature=temperature)
            energy = self._calculate_energy()
            magnetization = self._calculate_magnetization()
            E += energy
            M += magnetization
            E_squared += energy**2
            M_squared += magnetization**2 

            if (i % (num_steps // 4) == 0) and plot == True:
                self.plot_system()

            self.step += 1

        normalization_factor_1 = (num_steps * self.N**2)
        normalization_factor_2 = (num_steps**2 * self.N**2)

        self.energy = E / normalization_factor_1
        self.magnetization = M / normalization_factor_1
        self.specific_heat = (E_squared / normalization_factor_1 - E**2 / normalization_factor_2) / (temperature**2)
        self.susceptibility = (M_squared / normalization_factor_1 - M**2 / normalization_factor_2) / (temperature)

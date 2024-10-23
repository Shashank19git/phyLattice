from phyLattice.xy_package.utils import np
from torch import rand, pi, sum, cos, roll, exp, tensor
'''
def compute_energy(spins, L, T, J):
    energy = 0.0
    for i in range(L):
        for j in range(L):
            S = np.array([np.cos(spins[i, j]), np.sin(spins[i, j])])
            neighbors = [
                spins[(i + 1) % L, j],
                spins[i, (j + 1) % L],
                spins[(i - 1) % L, j],
                spins[i, (j - 1) % L]
            ]
            for neighbor in neighbors:
                S_neighbor = np.array([np.cos(neighbor), np.sin(neighbor)])
                energy -= J * np.dot(S, S_neighbor)
    return energy / 2.0  # Each pair is counted twice
'''

def compute_energy(spins, L, T, J):
    return -(J/2)*sum(cos(spins-roll(spins, 1, 0)) + cos(spins-roll(spins, 1, 1)) + cos(spins-roll(spins, -1, 0)) + cos(spins-roll(spins, -1, 1))) #energy

# Function to compute specific heat
def specific_heat(energies, L, T, J):
    import numpy as np
    E_mean = np.mean(energies)
    E_squared_mean = np.mean(np.array(energies)**2)
    C_v = (E_squared_mean - E_mean**2) / (T**2)
    return C_v

# auto correlation
def correlation(configs):
    mean = np.mean(configs, axis=0)
    return (np.mean((configs[0]-mean)*(configs[len(configs)-1]-mean)))/(np.mean(np.std(configs[0],axis=0)*np.std(configs[len(configs)-1],axis=0)))


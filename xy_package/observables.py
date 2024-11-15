from phyLattice.xy_package.utils import np
from torch import rand, pi, sum, cos, roll, exp, tensor

def compute_energy(spins, L, T, J):
    """
    This function computes energy for a give spin configuration.
    
    Arg:
        spins: spin configuration as a torch tensor

        L: Lattice size

        T: Temperature
         
        J: coupling constant for nearest neighbour intrection
    """
    return -(J/2)*sum(cos(spins-roll(spins, 1, 0)) + cos(spins-roll(spins, 1, 1)) + cos(spins-roll(spins, -1, 0)) + cos(spins-roll(spins, -1, 1))) #energy


# auto correlation
def correlation(configs):
    """
    This function computes Auto-correlation between two configuration for two different monte carlo time
    
    
    Arg:
        configs: list of configuration of a fixrd length
    """
    mean = np.mean(configs, axis=0)
    return (np.mean((configs[0]-mean)*(configs[len(configs)-1]-mean)))/(np.mean(np.std(configs[0],axis=0)*np.std(configs[len(configs)-1],axis=0)))


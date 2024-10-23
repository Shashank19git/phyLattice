from os import path, makedirs
from phyLattice.xy_package.utils import np
from phyLattice.xy_package.utils import plt
from torch import randn, pi, sum, cos, roll, exp
from phyLattice.xy_package.MC_updates import monte_carlo_step
from phyLattice.xy_package.observables import compute_energy, correlation, specific_heat


def energy(L ,T, J, n_steps, plot=False, MC_step_size=1):
    spins = 2 * pi * randn(L, L) - pi
    energies = []
    for step in range(n_steps):
        #if step%100==0:
        #    print(step)
        spins = monte_carlo_step(spins, L, T, J, MC_step_size)# Collect data
        #print(compute_energy(spins, L, T, J))
        energies.append(compute_energy(spins, L, T, J))
    if plot:

# Plotting the results
        plt.figure(figsize=(8, 6))
        plt.plot(np.linspace(0,1,n_steps),energies)
        plt.xlabel('MC_time')
        plt.ylabel('Energy')
        plt.title('Energy vs MC_time')
        plt.show()
    return compute_energy(spins, L, T, J)
def specific_heat(L, T, J, n_steps, MC_step_size=1):
    spins = 2 * pi * randn(L, L) - pi
    energies = []
    for step in range(n_steps):
        spins = monte_carlo_step(spins, L, T, J, MC_step_size)
        energies.append(compute_energy(spins, L, T, J))
    E_mean = np.mean(energies)
    E_squared_mean = np.mean(np.array(energies)**2)
    C_v = (E_squared_mean - E_mean**2) / (T**2)
    return C_v

def autocorrelation(L, T, J, n_steps, lag, MC_step_size=1):
    spins = 2 * pi * randn(L, L) - pi
    configs=[]
    Autocorrelations=[]
    for steps in range(n_steps):
        spins = monte_carlo_step(spins,L, T, J, MC_step_size)
        configs.append(spins)
        if (steps+1)>=lag:
            Autocorrelations.append(correlation(np.array(configs)))
            configs.pop(0)
    #print(np.array(Autocorrelations).view)
# Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0,1,len(Autocorrelations)),Autocorrelations)
    plt.xlabel('MC_time')
    plt.ylabel('autocorrelation')
    plt.title('autocorrelation vs MC_time')
    plt.show()

def generate_configurations(L , T, J , n_steps, thermalisation_time, lag, MC_step_size=1):
    spins = 2 * pi * randn(L,L) - pi
    makedirs("configurations",exist_ok=True)
    count = 0
    for n in range(n_steps):
        spins = monte_carlo_step(spins, L, T, J, MC_step_size)
        if n > thermalisation_time and (n)%lag==0:
                count += 1
                np.savetxt(path.join("configurations",f"congfig_{count}.csv"),spins.numpy(),delimiter=',')

from os import path, makedirs
from utils import np
from utils import plt
from torch import randn, pi, sum, cos, roll, exp
from MC_updates import monte_carlo_step
from observables import compute_energy , correlation


def energy(L ,T, J, n_steps):
    spins = 2 * pi * randn(L, L) - pi
    energies = []
    for step in range(n_steps):
        if step%10==0:
            print(step)
        spins = monte_carlo_step(spins,L, T, J)# Collect data
        #print(compute_energy(spins, L, T, J))
        energies.append(compute_energy(spins, L, T, J))

# Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0,1,n_steps),energies)
    plt.xlabel('MC_time')
    plt.ylabel('Energy')
    plt.title('Energy vs MC_time')
    plt.show()


def autocorrelation(L, T, J, n_steps, lag):
    spins = 2 * pi * randn(L, L) - pi
    configs=[]
    Autocorrelations=[]
    for steps in range(n_steps):
        spins = monte_carlo_step(spins,L, T, J)
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

def generate_configurations(L , T, J , n_steps, thermalisation_time, lag):
    spins = 2 * pi * randn(L,L) - pi
    makedirs("configurations",exist_ok=True)
    count = 0
    for n in range(n_steps):
        spins = monte_carlo_step(spins, L, T, J)
        if n > thermalisation_time and (n)%lag==0:
                count += 1
                np.savetxt(path.join("configurations",f"congfig_{count}.csv"),spins.numpy(),delimiter=',')
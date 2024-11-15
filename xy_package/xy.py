from os import path, makedirs
from phyLattice.xy_package.utils import np
from phyLattice.xy_package.utils import plt
from torch import randn, pi, sum, cos, roll, exp
from phyLattice.xy_package.MC_updates import monte_carlo_step
from phyLattice.xy_package.observables import compute_energy, correlation


def energy(L ,T, J, n_steps, plot=False, MC_step_size=1):
    """
    This function does monte carlo updates for given number of steps and computes energy of the final updated configuration
    
    
    Arg:
        spins: spin configuration as a torch tensor

        L: Lattice size

        T: Temperature

        J: coupling constant for nearest neighbour intrection

        n_steps: number of monte carlo steps

        plot: "True" if plot required else default is "False"
         
        MC_step_size: step size for monte carlo update
    """
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


def autocorrelation(L, T, J, n_steps, lag, MC_step_size=1):
    """
    This function plots Auto-correlation between to configuration seperated by some time lag, w.r.t monte carlo time. Which is useful to see after how many updates the Auto-coorelation is near zero to start sampling.
    
    
    Arg:
        spins: spin configuration as a torch tensor

        L: Lattice size

        T: Temperature

        J: coupling constant for nearest neighbour intrection

        n_steps: number of monte carlo updates

        lag: monte carlo time between two configurations 

        MC_step_size: step size for monte carlo update
         
    """
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
    """
    This function generates configuration sfter thermalisation and save them in a folder.
    
    Arg:
        spins: spin configuration as a torch tensor

        L: Lattice size

        T: Temperature

        J: coupling constant for nearest neighbour intrection

        n_steps: number of monte carlo updates

        thermalisation_time: monte carlo step after which configuration are to be sampled, which is to be gussed uning Auto-correlation plot

        lag: monte carlo time between two configurations 

        MC_step_size: step size for monte carlo update
         
    """
    spins = 2 * pi * randn(L,L) - pi
    makedirs("configurations",exist_ok=True)
    count = 0
    for n in range(n_steps):
        spins = monte_carlo_step(spins, L, T, J, MC_step_size)
        if n > thermalisation_time and (n)%lag==0:
                count += 1
                np.savetxt(path.join("configurations",f"congfig_{count}.csv"),spins.numpy(),delimiter=',')

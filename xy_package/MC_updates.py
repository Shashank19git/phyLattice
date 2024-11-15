from torch import rand, pi, mean, cos, roll, exp, where, logical_or


def monte_carlo_step(spins, L, T, J, MC_step_size):
    """
    function to impliment monte carlo update on given spin configuration
    
    Arg:
     spins: spin configuration as a torch tensor

     L: Lattice size

     T: Temperature

     J: coupling constant for nearest neighbour intrection
         
     MC_step_size: step size for monte carlo update
    """
    theta_new = spins + MC_step_size*(2*pi*rand(L, L) - pi)
    delta_E = -J*((cos(theta_new-roll(spins, 1, 0)) + cos(theta_new-roll(spins, 1, 1)) + cos(theta_new-roll(spins, -1, 0)) + cos(theta_new-roll(spins, -1, 1))) - (cos(spins-roll(spins, 1, 0)) + cos(spins-roll(spins, 1, 1)) + cos(spins-roll(spins, -1, 0)) + cos(spins-roll(spins, -1, 1))))
    random_values = rand(L,L)
    return where(logical_or(delta_E < 0, random_values < exp(-delta_E/T)), theta_new, spins)

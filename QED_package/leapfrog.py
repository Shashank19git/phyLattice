from phyLattice.QED_package.force_term import calculate_fermion_force, gauge_force
from phyLattice.QED_package.action import staggered_dirac_operator
from torch import exp

#leapfrog update

def leapfrog_update(U, P, chi, chi_bar, p_chi, p_chi_bar, beta, m, epsilon_U, epsilon_F, L, Lt, dimension):
    """Perform a leapfrog update step"""

# update fermion fields with momenta
    
    # update fermion momenta p_chi and p_chi_bar
    grad_chi,_ = staggered_dirac_operator(chi, U, L, Lt, m, dimension)
    grad_chi_bar,_ = staggered_dirac_operator(chi_bar, U, L, Lt, m, dimension)
    p_chi -= epsilon_F * grad_chi
    p_chi_bar -= epsilon_F * grad_chi_bar
    # update fermion fields with momenta
    chi += epsilon_F * p_chi
    chi_bar += epsilon_F * p_chi_bar


    # Update momentum P 
    F = gauge_force(U, beta, dimension)+(calculate_fermion_force(chi, U, L, Lt, m, dimension))

    P += (epsilon_U/2) * F

    # Update link variables U with the updated momentum P
    
    U *= exp(1j * epsilon_U * P)
    
    

    
    
    #print('yo')

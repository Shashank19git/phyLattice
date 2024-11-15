from torch import sum, exp, rand, randn, float64, abs
import phyLattice.QED_package.action as fm_action
from phyLattice.QED_package import leapfrog


#initiate link variables and fermion fields
def init_fields(L, Lt, dimension):
    '''initialise fermion fields and link variables on Lattice.
    
    Arg:
        L: Spatial Lattice size

        Lt: Temporal Lattice size

        dimension: dimension of Lattice
    '''
    #fermion fields
    global psi
    global chi
    global chi_bar
    global U
    if dimension == 2:
         psi = randn((L, Lt, 2), dtype=float64)  # Real and imaginary parts
         chi = psi[..., 0] + 1j * psi[..., 1]  # Combine real and imaginary parts
         chi_bar = chi.conj()
         # Create gauge fields (complex numbers on the links) initialized as U(1) links
         U = randn(L, Lt, dimension, dtype=float64) + 1j*randn(L, Lt, dimension, dtype=float64)
         U = U/abs(U)
    if dimension == 3:
         psi = randn((L, L, Lt, 2), dtype=float64)  # Real and imaginary parts
         chi = psi[..., 0] + 1j * psi[..., 1]  # Combine real and imaginary parts
         chi_bar = chi.conj()
         # Create gauge fields (complex numbers on the links) initialized as U(1) links
         U = randn(L, L, Lt, dimension, dtype=float64) + 1j*randn(L, L, Lt, dimension, dtype=float64)
         U = U/abs(U)
    if dimension == 4:
         psi = randn((L, L, L, Lt, 2), dtype=float64)  # Real and imaginary parts
         chi = psi[..., 0] + 1j * psi[..., 1]  # Combine real and imaginary parts
         chi_bar = chi.conj()
         # Create gauge fields (complex numbers on the links) initialized as U(1) links
         U = randn(L, L, L, Lt, dimension, dtype=float64) + 1j*randn(L, L, L, Lt, dimension, dtype=float64)
         U = U/abs(U)


def hmc_update(chi, chi_bar, U, L, Lt, a, beta, m, epsilon_U, epsilon_F, dimension):
    """Perform a single HMC update.
    
    Arg:
        chi: staggered fermion field

        chi_bar: conj. fermion field

        U: link variables

        m: fermion mass

        a: Lattice constant

        L: Spetial Lattice size

        Lt: Temporal Lattice size

        beta: coupling 

        epsilon_U: step size to update link variables

        epsilon_F: step size to update fermions
         
        dimension: dimension of Lattice"""

    #initiate momenta fields gaussian distributed
    if dimension == 2:
        p_psi = randn((L, Lt, 2), dtype=float64)
        P = randn(L, Lt, 2, dtype=float64) + 1j*randn(L, Lt, 2, dtype=float64)
        p_chi = p_psi[..., 0] + 1j * p_psi[..., 1]  # Combine real and imaginary parts
        p_chi_bar = p_chi.conj()
    if dimension == 3:
        p_psi = randn((L, L, Lt, 2), dtype=float64)
        P = randn(L, L, Lt, 3, dtype=float64) + 1j*randn(L, L, Lt, 3, dtype=float64)
        p_chi = p_psi[..., 0] + 1j * p_psi[..., 1]  # Combine real and imaginary parts
        p_chi_bar = p_chi.conj()
    if dimension == 4:
        p_psi = randn((L, L, L, Lt, 2), dtype=float64)
        P = randn(L, L, L, Lt, 4, dtype=float64) + 1j*randn(L, L, L, Lt, 4, dtype=float64)
        p_chi = p_psi[..., 0] + 1j * p_psi[..., 1]  # Combine real and imaginary parts
        p_chi_bar = p_chi.conj()

    chi_old = (chi).clone()
    p_chi_old = p_chi.clone()
    chi_bar_old = (chi_bar).clone()
    p_chi_bar_old = (p_chi_bar).clone()
    U_old = U.clone()
    P_old = P.clone() 

    #old Hamiltonian

    H_old = (sum(p_chi * (p_chi).conj() + p_chi_bar * (p_chi_bar).conj())).real + fm_action.staggered_fermion(chi, chi_bar, U, m, a, L, Lt, dimension) + fm_action.gauge_action(U, beta, dimension)
    
    #leapfrog update
    leapfrog.leapfrog_update(U, P, chi, chi_bar, p_chi, p_chi_bar, beta, m, epsilon_U, epsilon_F, L, Lt, dimension)

    #new Hamiltonian

    H_new = (sum(p_chi * (p_chi).conj() + p_chi_bar * (p_chi_bar).conj())).real + fm_action.staggered_fermion(chi, chi_bar, U, m, a, L, Lt, dimension) + fm_action.gauge_action(U, beta, dimension)
    
    dH = H_new - H_old
    if rand(1) < exp(-dH):
        chi = chi_old
        chi_bar = chi_bar_old
        U = U_old

from phyLattice.QED_package.action import staggered_dirac_operator, conjugate_gradient
from torch import sum, zeros_like, complex128, tensor, roll


# Calculate fermion force
def calculate_fermion_force(chi, U, L, Lt, m, dimension):
    """calculate fermion force"""
    D, derivative = staggered_dirac_operator(chi, U, L, Lt, m, dimension)
    force = zeros_like(U,dtype=complex128) + 1j*zeros_like(U,dtype=complex128)

    for mu in range(dimension):

        # Compute the force
        force[...,mu] -= ((tensor(conjugate_gradient(D.conj()*D,((D).conj()) * chi, tol=1e-8, maxiter=1000)) * derivative[...,mu] ))
        #force[..., mu] -= (sum(conjugate_gradient(((D).conj()) @ chi, chi, U, L, Lt, dimension, tol=1e-8, maxiter=1000)* derivative[...,mu], axis=-1)).to(cdouble)

    return force

# calculate gauge force

def shift_link(U, shift, direction):
    """Shift the lattice in a given direction"""
    return roll(U, shift, direction)

#gauge force
def compute_staple(U, mu, nu):
    """Compute the staple for a given direction (mu, nu)"""
    # Forward staple contribution
    forward_staple = U[..., nu] * shift_link(U[..., mu], 1, nu) * (shift_link(U[..., nu], 1, mu).conj())

    # Backward staple contribution (shift backwards in nu direction)
    backward_staple = (shift_link(U[..., mu], -1, nu).conj()) * shift_link(U[..., mu], -1, nu) * shift_link(shift_link(U[..., nu], -1, nu),1,mu)

    return forward_staple + backward_staple

def gauge_force(U, beta, dimension):
    """Calculate the gauge force F_mu(x) for the given direction mu"""
    force = zeros_like(U, dtype=complex128)
    for mu in range(dimension):
        for nu in range(dimension):
          if nu != mu:
            # Compute the staple for this mu, nu combination
            staple = compute_staple(U, mu, nu)
            # Sum the staples (plaquette contributions) over all directions except mu
        force[...,mu] += staple
    return -1j * beta * force

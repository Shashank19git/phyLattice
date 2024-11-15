from torch import roll, sum, pow, stack, meshgrid, arange, zeros_like, sqrt, complex128
from numpy import vdot, sqrt, array
from numpy import zeros_like as zeros

# Function to compute eta_mu(x)
def eta_mu(mu, x):
    # Calculate the product of coordinates up to mu-1
    eta = pow(-1, sum(x[..., :mu], dim=-1))
    return eta


def staggered_fermion(chi, chi_bar, U, m, a, L, Lt, dimension):
    """
    Function to calculate staggered fermion action.
    
    return:
        staggered fermion action
    
    Arg:
        chi: staggered fermion field

        chi_bar: conjugate fermion feild

        U: link variables

        m: fermion mass

        a: Lattice constant

        L: Spetial Lattice size

        Lt: Temporal Lattice size

        dimension: dimension of Lattice
         
    """
    action = 0
    if dimension == 2:
        indices = stack(meshgrid(arange(L), arange(Lt)), dim=-1)
    if dimension == 3:
        indices = stack(meshgrid(arange(L), arange(L), arange(Lt)), dim=-1)
    if dimension == 4:
        indices = stack(meshgrid(arange(L), arange(L), arange(L), arange(Lt)), dim=-1)
    for mu in range(dimension):
        action += 0.5*(1/a)*sum(chi_bar*eta_mu(mu, indices)*(U[...,mu]*roll(chi, 1, mu)-roll(U[...,mu], -1, mu).conj()*roll(chi, -1, mu))).real 
    action += m*sum(chi_bar*chi).real
    return action
    

def staggered_dirac_operator(chi, U, L, Lt, m, dimension):
    """
    Function to calculate staggered dirac operator and its derivative w.r.t link variables.
    
    return: 
        staggered dirac operator, derivative w.r.t link variables
            
    

    Arg:
        chi: staggered fermion field

        U: link variables

        m: fermion mass

        L: Spetial Lattice size

        Lt: Temporal Lattice size

        dimension: dimension of Lattice
    """
    operator = zeros_like(chi,dtype=complex128)
    derivative_U = []
    if dimension == 2:
        indices = stack(meshgrid(arange(L), arange(Lt)), dim=-1)
    if dimension == 3:
        indices = stack(meshgrid(arange(L), arange(L), arange(Lt)), dim=-1)
    if dimension == 4:
        indices = stack(meshgrid(arange(L), arange(L), arange(L), arange(Lt)), dim=-1)
    
    for mu in range(dimension):
        u = U[...,mu]
        operator += 0.5*(eta_mu(mu, indices)*(u*roll(chi, 1, mu)-roll(u, -1, mu).conj()*roll(chi, -1, mu)))
        u.requires_grad_(True)
        u.retain_grad()
        ((u*roll(chi, 1, mu)-roll(u, -1, mu).conj()*roll(chi, -1, mu))).backward(u)
        derivative_U.append(u.grad)
    operator += m * chi
    
    return operator, stack(derivative_U,-1)
# Conjugate Gradient Algorithm (assuming already implemented)
def conjugate_gradient(a, B, tol=1e-6, maxiter=None):
    """Function to impliment conjugate gradient method to find inverse of dicar operator.
    return: 
        inverse(a)@B
    
    Arg:
        a: Symmetric positive definet matrix

         b: vector"""
    A = array(a,dtype='complex128')
    b = array(B,dtype='complex128')
    x = zeros(b)
    r = b - A * x
    p = r.copy()
    rsold = vdot(r, r)

    for i in range(maxiter or len(b)):
        Ap = A * p
        App= vdot(Ap, p)
        alpha = rsold / App
        x += alpha * p
        r -= alpha * Ap
        rsnew = vdot(r, r)
        if sqrt(rsnew) < tol:
            #print('yo')
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def gauge_action(U, beta, dimension):
    """
    Function to calculate gauge action.
    
    return:
        gauge action

    Arg:
        U: link variables

        beta: coupling 

        dimension: dimension of Lattice"""

    action = 0
    for mu in range(dimension):
        for nu in range(mu+1, dimension):
            action += beta*sum(1-U[...,mu]*roll(U[...,nu], nu, 1)*roll(U[...,mu], mu, 1).conj()*U[...,nu].conj())
    return action.real


def QED_action(chi, chi_bar, U, beta, m, a, L, Lt, dimension):
    """
    Function to calculate Total action on Lattice
    
    return:
        action
    
    
    Arg:
        chi: staggered fermion field

        chi_bar: conj. fermion field

        U: link variables

        m: fermion mass

        a: Lattice constant
         
        L: Spetial Lattice size

        Lt: Temporal Lattice size

        beta: coupling 

        dimension: dimension of Lattice
    """
    return staggered_fermion(chi, chi_bar, U, m, a, L, Lt, dimension) + gauge_action(U, beta, dimension)
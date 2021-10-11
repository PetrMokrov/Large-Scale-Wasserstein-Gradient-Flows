import numpy as np
from .tridiagonal_solver import TridiagonalSolver

class CCTarget:
    '''
    Base class to use for diffusion numerical 
    iterations
    '''

    def __init__(self):
        pass
    
    def potential(self, x):
        raise NotImplementedError
    
    def grad_potential(self, x):
        raise NotImplementedError
    
    def equil_solution(self, x, beta=1.):
        return np.exp(- beta * self.potential(x))
    

def get_dx(xs):
    if isinstance(xs, list):
        return xs[1] - xs[0]
    return xs[1].item() - xs[0].item()

def compute_delta(target, xs, beta=1.):
    dx = get_dx(xs)
    xs_half = xs + dx/2.
    xs_expd = np.insert(xs, len(xs), dx + xs[-1])
    B_j_half = target.grad_potential(xs_half)
    u = target.equil_solution(xs_expd, beta=beta)
    u_diff = u[1:] - u[:-1]
    c = 1./(beta*dx)
    delta = (B_j_half * u[1:] + u_diff*c)/(B_j_half * u_diff)
    return delta

def iterate_diffusion_cc(target, pdfs, xs, dt, beta=1.):
    '''
    Performs one iteration of diffusion process (w.r. time) using 
    Chang&Cooper scheme (see Chang and G Cooper. A practical difference scheme for fokker-planck equations.)
    :Parameters:
    target : class : implements `equil_solution` and `grad_potential` functions
    pdfs : np.array of (n,) : pdf on the grid
    xs : np.array of (n,) : the grid
    dt : float : iteration interval
    '''
    dx = get_dx(xs)
    xs_min_expd = np.insert(xs, 0, xs[0]-dx)
    delta_min_expd = compute_delta(target, xs_min_expd, beta=beta)
    delta = delta_min_expd[1:]
    delta_min = delta_min_expd[:-1]
    xs_half_expd = xs_min_expd + dx/2.
    B_j_half_expd = target.grad_potential(xs_half_expd)
    B_j_half = B_j_half_expd[1:]
    B_j_min_half = B_j_half_expd[:-1]
    dtdx = dt/dx
    c = 1./(beta*dx)
    upp = -((1. - delta)*B_j_half + c) * dtdx # x_j+1
    diag = 1. + dtdx * (2*c + (1. - delta_min) * B_j_min_half - delta * B_j_half) # x_j
    under = dtdx *(delta_min*B_j_min_half - c) # x_j-1
    td = TridiagonalSolver(under, diag, upp)
    res = td.solve(pdfs)
    _int = np.sum(res)*dx # numerical integration
    assert np.abs(_int - 1.) < 1e-2, 'error is {}'.format(_int)
    return res/_int

##########################################
# naive method to propagate the diffusion

def first_ord_derivative(xs, vals):
    '''
    Estimates the derivatives of the function, defined by vals
    on the grid xs
    :Parameters:
    xs : np.array of (n,): grid
    vals : np.array of (n,) : function values on the grid
    '''
    assert len(xs) == len(vals)
    dx = get_dx(xs)
    diff = vals[1:] - vals[:-1]
    f_d = np.insert(diff, len(diff), diff[-1])
    b_d = np.insert(diff, 0, diff[0])
    return (f_d + b_d) / (2.*dx)

def second_ord_derivative(xs, vals):
    '''
    Estimates the second derivatives of the function, defined by vals
    on the grid xs
    :Parameters:
    xs : np.array of (n,) : grid
    vals : np.array of (n,) : function values on the grid 
    '''
    dx = get_dx(xs)
    diff = vals[1:] - vals[:-1]
    f_x_p_h = vals[2:]
    f_x = vals[1:-1]
    f_x_m_h = vals[:-2]
    dd_f = (f_x_p_h + f_x_m_h)- 2*f_x 
    dd_f = np.insert(dd_f, len(dd_f), 0)
    dd_f = np.insert(dd_f, 0, 0)
    dd_f[0] = vals[2] - 2. * vals[1] + vals[0]
    dd_f[-1] = vals[-3] - 2.*vals[-2] + vals[-1]
    return dd_f/(dx**2)

def iterate_diffusion_simple(target, pdfs, xs, dt, beta=1.):
    '''
    Performs one iteration of diffusion process (w.r. to time) 
    by naive estimating of the derivatives 
    Warning: it's unstable!
    :Parameters:
    target : class : should implemeng grad_potential
    pdfs : np.array of (n,) : pdf on the grid
    xs : np.array : equidistant grid
    dt : float : iteration time
    '''
    g_psi_p = target.grad_potential(xs) * pdfs
    g_g_psi_p = first_ord_derivative(xs, g_psi_p)
    g_g_px = second_ord_derivative(xs, pdfs)
    dp_dt = g_g_psi_p + g_g_px / beta
    return pdfs + dt*dp_dt


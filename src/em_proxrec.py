import torch
import numpy as np
from torch.autograd.functional import jacobian
from multipledispatch import dispatch
from collections.abc import Callable
from scipy.stats import gaussian_kde

def torchProxRecur(pdf_prev, pot_prev, x_prev, x_curr, dt, reg, tol=1e-5, maxiter=300, beta=1., verbose=True):
    '''
    Proximal recursion to compute `pdf_curr` from `pdf_prev`
    :Parameters:
    pdf_prev: torch.Tensor (N,) : pdf estimated on x_prev
    pot_prev : torch.Tensor (N,) : potential drift vector estimated at x_prev
    x_prev : torch.Tensor (N, d) : samples from the process distribution at previous time moment t_{k - 1}
    x_curr : torch.Tensor (N, d) : samples from the process distribution at current time moment t_k
    dt : float : time segment t_k - t_{k - 1}
    reg : float : proximal operator entropic regularization parameter
    tol : float : tolerance for early stopping the iterations
    maxiter : float : maximal number of iterations
    beta : float : inverse temperature
    '''
    assert(len(pdf_prev.shape) == 1)
    assert(len(pot_prev.shape) == 1)
    assert(len(x_prev.shape) == 2)
    assert(len(x_curr.shape) == 2)
    assert(x_prev.size(0) == pdf_prev.size(0))
    N = len(pdf_prev)
    C = torch.cdist(x_prev, x_curr, p=2.) ** 2
    G = torch.exp(- C / (2. * reg))
    xi = torch.exp( - beta * pot_prev - 1.)
    # taken from original code on matlab
    lambda1 = torch.randn(N, dtype=pdf_prev.dtype)
    z = torch.exp((dt * lambda1)/ reg)
    # z = torch.randn(N)
    y = pdf_prev / torch.matmul(G, z)
    y_diff_norm = 0.
    z_diff_norm = 0.
    for l in range(maxiter):
        # print('y', torch.any(torch.isinf(y)))
        # print('y max min', y.max(), y.min())
        # print('G.Ty', torch.any(torch.isinf(torch.matmul(G.T, y))))
        # print('G.Ty max min', torch.matmul(G.T, y).max(), torch.matmul(G.T, y).min())
        z_new = torch.pow(xi / torch.matmul(G.T, y), 1./(1. + beta * reg / dt))
        y_new = pdf_prev / torch.matmul(G, z_new)
        # print('z_new', torch.any(torch.isinf(z_new)))
        # print('z_new max min', z_new.max(), z_new.min())
        # print('y_new', torch.any(torch.isinf(y_new)))
        # print('y_new max min', y_new.max(), y_new.min())
        y_diff_norm = torch.linalg.norm(y - y_new, ord=2)
        z_diff_norm = torch.linalg.norm(z - z_new, ord=2)
        if torch.isnan(y_diff_norm) or torch.isnan(z_diff_norm):
            # start new attempt:(
            # print(f'[torchProxRecur]: new attempt:(')
            # return torchProxRecur(
            #     pdf_prev, pot_prev, x_prev, x_curr, dt, reg, tol=tol, 
            #     maxiter=maxiter, beta=beta, verbose=verbose)
            raise Exception('too severe conditions!')
        if (y_diff_norm < tol) and (z_diff_norm < tol):
            y, z = y_new, z_new
            if verbose:
                print(f'[torchProxRecur] early stop at {l} iterations')
            break
        y, z = y_new, z_new
    if verbose:
        print(f'[torchProxRecur] y_diff: {y_diff_norm}, z_diff: {z_diff_norm}') 
    pdf_curr = z * torch.matmul(G.T, y)
    return pdf_curr

def EM_step_generator(x0, dt, minus_grad_pot, beta=1., dtype=torch.float32):
    device = x0.device
    _mean = torch.tensor(0.0, dtype=dtype).to(device)
    _std = torch.tensor(np.sqrt(dt), dtype=dtype).to(device)

    def deltaW(n, m):
        return torch.normal(_mean, _std, (n, m), device=device)
    
    n_pts = x0.shape[0]
    dim = x0.shape[1]

    def EM_step(x):
        dWn = deltaW(n_pts, dim) # (n, d)
        x_new = x + minus_grad_pot(x)*dt + np.sqrt(2./beta) * dWn
        return x_new
    return EM_step

def torchBatchItoEulerProxrec(
    pot_func, x0, pdf0, dt, t_fin, reg=0.05, tol=1e-5, 
    maxiter=300, beta=1., verbose=True, grad_pot_func=None):
    '''
    The function implements the overall scheme described in https://arxiv.org/pdf/1809.10844.pdf
    :Parameters:
    pot_func : callable : potential function of a Fokker-Planck process
    x0 : torch.Tensor (N, d) : points sample from initial distribution
    pdf0 : torch.Tensor (N,) : pdf of initial distribution estimated at x0
    dt : float : time segment of EM and ProxRecur operations
    t_fin : float : final diffusion timestep
    grad_pot_func : callable or None : gradient of potential function
    For other parameters see `torchProxRecur` signature reference
    '''

    sum_pot_func = lambda x: torch.sum(pot_func(x))
    if grad_pot_func is None:
        minus_grad_pot = lambda x: -jacobian(sum_pot_func, x)
    else:
        minus_grad_pot = lambda x: - grad_pot_func(x)
    dtype = x0.dtype
    
    N_iters = round(float(t_fin)/dt)
    assert N_iters * dt == t_fin
    # normalize pdf
    # pdf = pdf0/torch.sum(pdf0)
    pdf = pdf0.clone()
    x = x0.clone()
    EM_step = EM_step_generator(x0, dt, minus_grad_pot, beta=beta, dtype=dtype)
    for i_iter in range(N_iters):
        # perform one step of EM
        if verbose:
            print(f'[torchBatchItoEulerProxrec]: iteration {i_iter}')
        x_new = EM_step(x)
        # print(x_new.shape)
        pot = pot_func(x)
        # print(pot.shape)
        # perform ProxRecur
        pdf_new = torchProxRecur(
            pdf, pot, x, x_new, dt, reg, tol=tol, 
            maxiter=maxiter, beta=beta, verbose=verbose)
        x, pdf = x_new, pdf_new
    return x, pdf


def _normalize_pdf_reference(x, pdf, true_pdf_callable, method='importance'):
    '''
    :Parameters:
    x : torch.Tensor : sample from the distribution under consideration
    pdf : torch.Tensor : unnormalized pdf of the distrib under consideration at pts x
    true_pdf_callable: callable : reference distribution pdf callable
    '''
    assert method in ['importance', 'leastsq']
    true_pdf = true_pdf_callable(x)
    if isinstance(true_pdf, torch.Tensor):
        true_pdf = true_pdf.detach().cpu().numpy()
    true_pdf = np.maximum(true_pdf, 1e-15)
    pdf = pdf.cpu().numpy()
    if method == 'importance':
        return np.mean(pdf / true_pdf)
    if method == 'leastsq':
        qc = np.sum(true_pdf * pdf) / np.sum(pdf * pdf)
        return 1./qc

def normalize_pdf_reference(x, pdf, true_distrib, method='importance'):

    def true_pdf_callable(X):
        return np.exp(true_distrib.log_prob(X).cpu().numpy())
    
    return _normalize_pdf_reference(x, pdf, true_pdf_callable, method=method)

def normalize_pdf_kde(x, pdf, method='importance'):

    curr_em_distrib = gaussian_kde(x.cpu().numpy().transpose())
    def pdf_callable(x):
        return curr_em_distrib(x.cpu().numpy().T)
    return _normalize_pdf_reference(x, pdf, pdf_callable, method=method)


@dispatch(torch.Tensor, torch.Tensor, object)
def KL_targ_distrib_importance(x, pdf, targ):
    pdf_exact = np.exp(targ.log_prob(x).detach().cpu().numpy())
    pdf = pdf.detach().cpu().numpy()
    diff = pdf_exact / np.maximum(pdf, 1e-15)
    res = np.nanmean(np.log(diff) * diff)
    return res.item()

@dispatch(torch.Tensor, torch.Tensor, object)
def KL_train_distrib(x, pdf, targ):
    log_pdf_exact = targ.log_prob(x).detach().cpu().numpy()
    log_pdf = np.log(np.maximum(pdf.detach().cpu().numpy(), 1e-15))
    res = np.mean(log_pdf - log_pdf_exact)
    return res.item()
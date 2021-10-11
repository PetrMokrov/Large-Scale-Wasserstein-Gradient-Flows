import torch
import torch.nn as nn
import numpy as np
import torch.distributions as TD
import scipy
import scipy.linalg
from copy import deepcopy
from multipledispatch import dispatch
from collections import Iterable
import sdepy
from .em import batchItoEuler
from .em_proxrec import torchBatchItoEulerProxrec

class OU_distrib_modeler:
    '''
    This class models distribution X(t) of OrnsteinUhlenbeck process
    dX(t) = - \grad \frac{1}{2}(x - b)^T A (x - b) dt + \sqrt{2 \beta^{-1}} d W(t)
    b is n-dim vector
    A is (n \cross n) invertible symmetric matrix
    \beta is a positive scalar parameters
    W(t) is standart n-dim Wiener process
    '''

    def _U_rotate(self, M):
        if len(M.shape) == 1:
            return self.U @ np.diag(M) @ self.U.conj().T
        return self.U @ M @ self.U.conj().T

    def __init__(self, A, b, beta):
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu()
        self.A = np.asarray(A)
        self.b = np.asarray(b) 
        self.beta = beta
        assert self.A.shape[0] == self.A.shape[1], 'matrix A must be square'
        # assert np.allclose(self.A.T, self.A, 1e-13), 'matrix A must be symmetric'
        self.A = 0.5*(self.A + self.A.T)
        assert self.b.shape[0] == self.A.shape[0], 'b an A dimensions must coincide'
        assert np.linalg.matrix_rank(self.A, tol=1e-6) == self.A.shape[0], 'matrix A must have full rank'
        self.theta = self.A
        T, U = scipy.linalg.schur(self.theta)
        # assert np.allclose(T, np.diag(np.diagonal(T)), 1e-13)
        self.T = np.diagonal(T)
        self.U = U
    
    def _get_add_params(self, t):
        _scale_param = self._U_rotate(np.exp(-self.T * t))
        _add_param = (np.eye(self.b.shape[0]) - _scale_param).dot(self.b)
        return _scale_param, _add_param
    
    def _get_var_param(self, t):
        return 2. * (1./self.beta) * self._U_rotate((1. - np.exp(- 2.*self.T * t))/(2. * self.T))
    
    def get_distrib_params(self, X_0, t, dtype=torch.float32, device='cpu'):
        X_0 = np.asarray(X_0)
        _scale, _add = self._get_add_params(t)
        mean = _scale.dot(X_0) + _add
        # e_min_th_t = self._U_rotate(np.exp(-self.T * t))
        # mean = e_min_th_t.dot(X_0) + (np.eye(self.b.shape[0]) - e_min_th_t).dot(self.b)
        # var = 2. * (1./self.beta) * self._U_rotate((1. - np.exp(- 2.*self.T * t))/(2. * self.T))
        var = self._get_var_param(t)
        trc_mean = torch.tensor(mean, dtype=dtype).to(device)
        trc_var = torch.tensor(var, dtype=dtype).to(device)
        return trc_mean, trc_var
    
    def get_distrib(self, X_0, t, dtype=torch.float32, device='cpu'):
        mean, var = self.get_distrib_params(X_0, t, dtype=dtype, device=device)
        return TD.MultivariateNormal(mean, var)

def get_normal_distrib_params(mvnormal_distribution):
    assert isinstance(mvnormal_distribution, (TD.Normal, TD.MultivariateNormal))
    mean = mvnormal_distribution.mean
    var = 0.
    if isinstance(mvnormal_distribution, TD.MultivariateNormal):
        var = mvnormal_distribution.covariance_matrix
    else:
        var = mvnormal_distribution.scale ** 2
        if var.size(0) == 1:
            var = var.view(1, 1)
        else:
            var = torch.diag(var)
    return mean, var

def create_ou_distrib_modeler(mvnormal_distribution, beta=1.0):
    mean, var = get_normal_distrib_params(mvnormal_distribution)
    var *= beta
    return OU_distrib_modeler(torch.inverse(var), mean, beta)


class OU_tDeterministic(TD.MultivariateNormal):

    @staticmethod
    def _get_params(X_0, ou_distrib_modeler, t):
        return ou_distrib_modeler.get_distrib_params(
            X_0, t, dtype=X_0.dtype, device=X_0.device)

    def __init__(self, X_0, ou_distrib_modeler, t):
        super().__init__(*self._get_params(X_0, ou_distrib_modeler, t))

class OU_tNormal(TD.MultivariateNormal):

    @staticmethod
    def _get_params(init_distrib, ou_distrib_modeler, t):
        assert isinstance(init_distrib, (TD.Normal, TD.MultivariateNormal))
        b, A = get_normal_distrib_params(init_distrib)
        i_A = torch.inverse(A)
        dtype = b.dtype
        device = b.device
        F, g = ou_distrib_modeler._get_add_params(t)
        F, g = torch.tensor(F, dtype=dtype).to(device), torch.tensor(g, dtype=dtype).to(device)
        Sigma = torch.tensor(ou_distrib_modeler._get_var_param(t), dtype=dtype).to(device)
        i_Sigma = torch.inverse(Sigma)
        i_Xi = F.T @ i_Sigma @ F + i_A
        Xi = torch.inverse(i_Xi)
        Psi = i_Sigma - i_Sigma @ F @ Xi @ F.T @ i_Sigma
        i_Psi = torch.inverse(Psi)
        phi = i_Sigma @ F @ Xi @ i_A @ b
        _mean = g + i_Psi @ phi
        return _mean, i_Psi

    def __init__(self, init_distrib, ou_distrib_modeler, t):
        super().__init__(*self._get_params(init_distrib, ou_distrib_modeler, t))

class OU_tMixtureNormal(TD.MixtureSameFamily):

    @dispatch(TD.Distribution, OU_distrib_modeler, object)
    def __init__(self, init_distrib, ou_distrib_modeler, t):
        assert isinstance(init_distrib, TD.MixtureSameFamily)
        mixture = init_distrib.mixture_distribution
        comp = init_distrib.component_distribution
        assert isinstance(comp, TD.MultivariateNormal)
        means = comp.loc
        vars = comp.covariance_matrix
        return self.__init__(mixture, means, vars, ou_distrib_modeler, t)
    
    @dispatch(TD.Distribution, Iterable, Iterable, OU_distrib_modeler, object)
    def __init__(self, mixture_distrib, means, vars, ou_distrib_modeler, t):
        assert len(means) == len(vars)
        f_means = []
        f_vars = []
        for i in range(len(means)):
            distrib = TD.MultivariateNormal(means[i], vars[i])
            f_mean, f_var = OU_tNormal._get_params(distrib, ou_distrib_modeler, t)
            f_means.append(f_mean)
            f_vars.append(f_var)
        f_distrib = TD.MultivariateNormal(torch.stack(f_means), torch.stack(f_vars))
        super().__init__(mixture_distrib, f_distrib)

def create_em_proxrec_samples(
    x0, pdf0, final_distrib, t_fin, t_stp, 
    beta=1., verbose=False, **proxrec_params):
    '''
    creates diffusion samples along with pdf estimate using https://arxiv.org/pdf/1809.10844.pdf
    '''
    assert isinstance(final_distrib, (TD.Normal, TD.MultivariateNormal))
    fin_mean, fin_var = get_normal_distrib_params(final_distrib)
    device = x0.device
    dtype = 'float32' if x0.dtype == torch.float32 else 'float64'
    targ_grad_potential = get_ou_potential_func(
        fin_mean, fin_var, dim_first=False, beta=beta, 
        grad=True, _type='torch', dtype=dtype, device=device)
    targ_potential = get_ou_potential_func(
        fin_mean, fin_var, dim_first=False, beta=beta, 
        grad=False, _type='torch', dtype=dtype, device=device)
    assert len(x0.shape) == 2
    x_fin, pdf_fin = torchBatchItoEulerProxrec(
        targ_potential, x0, pdf0, t_stp, t_fin, beta=beta, verbose=verbose, 
        grad_pot_func=targ_grad_potential, **proxrec_params)
    return x_fin, pdf_fin
    

@dispatch(np.ndarray, TD.Distribution, float, float, int)
def create_em_samples(
    x0, final_distrib, 
    t_fin, t_stp, n_samples, 
    beta=1., return_init_spls=False):
    '''
    creates diffusion samples using Euler-Maruyama iterations
    :Parameters:
    x0 : np.ndarray : particles distributed according to initial distribution
    init_distrib: torch.Distribution like : particles initial distribution
    final_distrib: torch.Distribution like : final MultivariateNormal distribution
    t_fin : float : particles observation time (start time is 0)
    t_stp : float : time step of EM iterations
    n_samples : int :count of particles to propagate
    beta : float : diffusion magnitude
    '''
    assert isinstance(final_distrib, (TD.Normal, TD.MultivariateNormal))
    fin_mean, fin_var = get_normal_distrib_params(final_distrib)
    targ_grad_potential = get_ou_potential_func(
        fin_mean, fin_var, dim_first=False, beta=beta, grad=True)
    np_fin_var_inv = torch.inverse(fin_var).cpu().numpy()
    np_fin_mean = fin_mean.cpu().numpy()

    def minus_targ_grad_potential(x):
        return - targ_grad_potential(x)

    assert x0.shape[0] == n_samples
    assert len(x0.shape) == 2
    x_fin = batchItoEuler(minus_targ_grad_potential, x0, t_stp, t_fin, beta=beta)
    if not return_init_spls:
        return x_fin
    return x0, x_fin
    
@dispatch(TD.Distribution, TD.Distribution, float, float, int)
def create_em_samples(
    init_distrib, final_distrib, 
    t_fin, t_stp, n_samples, beta=1., 
    return_init_spls=False):
    '''
    creates diffusion samples using Euler-Maruyama iterations
    :Parameters:
    init_distrib: torch.Distribution like : particles initial distribution
    '''
    x0 = init_distrib.sample((n_samples,)).cpu().numpy()
    return create_em_samples(
        x0, final_distrib, t_fin, t_stp, n_samples, 
        beta=beta, return_init_spls=return_init_spls)

def generate_ou_target(dim, mean_scale=1., dtype=torch.float32, device='cpu'):
    var = make_spd_matrix(dim)
    mean = np.random.randn(dim) * mean_scale
    trc_var = torch.tensor(var, dtype=dtype).to(device)
    trc_mean = torch.tensor(mean, dtype=dtype).to(device)
    targ_distrib = TD.MultivariateNormal(trc_mean, trc_var)
    init = np.random.randn(dim) * mean_scale
    return targ_distrib, mean, var

def get_ou_potential_func(
    mean, var, dim_first=True, beta=1., grad=False, 
    _type='numpy', device='cpu', dtype='float32'):

    assert _type in ['numpy', 'torch']
    assert dtype in ['float32', 'float64']

    if isinstance(var, torch.Tensor):
        var = var.detach().cpu().numpy()
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if isinstance(var, list):
        var = np.array(var)
    if isinstance(mean, list):
        mean = np.array(mean)
    if isinstance(var, float):
        var = np.array(var).reshape(1, 1)
    if isinstance(mean, float):
        mean = np.array(mean).reshape(1)
    assert len(var.shape) == 2
    assert len(mean.shape) == 1
    assert var.shape[0] == var.shape[1]
    assert var.shape[0] == mean.shape[0]
    var_inv = np.linalg.inv(var)
    dim = mean.shape[0]

    if _type == 'numpy':
        def ou_potential_func_dim_first(x):
            assert x.shape[0] == dim
            x_norm = x - mean.reshape((-1, 1))
            w = (1. / (2. * beta)) * np.sum(x_norm * np.dot(var_inv, x_norm), axis=0)
            return w
        
        def ou_grad_potential_func_dim_first(x):
            x_norm = x - mean.reshape((-1, 1))
            return np.dot(var_inv, x_norm) / beta
    else:
        dtype = torch.float32 if dtype == 'float32' else torch.float64
        mean = torch.tensor(mean, dtype=dtype, device=device)
        var_inv = torch.tensor(var_inv, dtype=dtype, device=device)
        def ou_potential_func_dim_first(x):
            assert x.size(0) == dim
            x_norm = x - mean.view((-1, 1))
            w = (1. / (2. * beta)) * torch.sum(x_norm * torch.matmul(var_inv, x_norm), dim=0)
            return w
        
        def ou_grad_potential_func_dim_first(x):
            x_norm = x - mean.view((-1, 1))
            return torch.matmul(var_inv, x_norm) / beta
    
    if dim_first:
        if grad:
            return ou_grad_potential_func_dim_first
        else:
            return ou_potential_func_dim_first
    if grad:
        return lambda x : ou_grad_potential_func_dim_first(x.T).T
    else:
        return lambda x : ou_potential_func_dim_first(x.T).T

if __name__ == "__main__":

    ##############

    A = np.array([[1., 0.5], [0.5, 2.]])
    b = np.array([0.1, 0.7])
    nm = TD.MultivariateNormal(torch.tensor(b), torch.tensor(A))
    beta=2.0
    ou_d_m = create_ou_distrib_modeler(nm, beta)
    X_0 = np.array([0.4, 2.3])
    mean, var = ou_d_m.get_distrib_params(X_0, 70.)
    assert np.allclose(mean, b, 1e-5)
    assert np.allclose(var, A, 1e-5)

    ##############

    d = OU_tDeterministic(X_0, ou_d_m, 0.01)
    print(d.distrib.mean)
    print(d.distrib.covariance_matrix)
    
import torch
import torch.nn as nn
from multipledispatch import dispatch
import numpy as np
import sdeint
import torch.distributions as TD
from scipy.stats import gaussian_kde
import scipy.stats as sps
import scipy
from collections.abc import Callable

from .diffusion import Diffusion
from .ou import get_normal_distrib_params
from .utils import discretize_distrib, normalize_grid
from .changcooper import iterate_diffusion_cc
from .em import batchItoEuler
from .frogner import run_diffusion, get_mv_uniform_loc_scale_params

class DiffusionFilteringMCMC:

    @staticmethod
    def _compute_term_factor(xs, term_params):
        assert(len(xs) == len(term_params) + 1)
        assert isinstance(xs, torch.Tensor)
        assert isinstance(term_params, list)
        assert len(xs.shape) == 3 # (n_diffs + 1, bs, dim)
        if len(xs) == 1:
            return torch.zeros(xs.size(1), dtype=xs.dtype, device=xs.device)
        dim = xs.size(-1)
        assert dim == 1
        ys = torch.tensor([
            term_params[i][1] for i in range(len(term_params))]).to(xs).view(-1, dim) # (n_diffs, dim)
        vars = torch.tensor([
            term_params[i][0] for i in range(len(term_params))]).to(xs).view(-1, dim) # (n_diffs, dim)

        res = torch.sum(((xs[1:] - ys.unsqueeze(1))**2)/vars.unsqueeze(1), -1).sum(0) # (bs)
        return res

    def __init__(self, init_distrib, method='zero', n_decorrelate=0):
        self.init_distrib = init_distrib
        self.init_is_trch_distrib = isinstance(self.init_distrib, torch.distributions.Distribution)
        self.diffs = []
        self.prev_Xs = None
        self.prev_term_params = []
        self.n_warm_up = 0
        self.warmed_up = True

        assert(method in ['zero', 'sqrt_c'])
        self.method = method
        self.n_decorrelate = n_decorrelate


    def sample_init(self, n):
        X = self.init_distrib.sample((n,) if self.init_is_trch_distrib else n)
        if len(X.shape) == 1:
            return X.view(-1, 1)
        return X
    
    def add_terminated_diff(self, diff, y_term, noise_distrib, n_warm_up):
        '''
        Adds new diffusion model, terminated by `y_term` sampled from the
        diffusion process with noise defined by `noise_distrib`
        :Parameters:
        diff : Diffusion : terminated diffusion process
        y_term : torch.tensor : sample - terminator
        noise_distrib : Normal distrib : observations noise
        n_warm_up : int : number of free-runninb sumples from MCMC before starting the sampling
        '''
        assert isinstance(noise_distrib, TD.Normal)
        mean = noise_distrib.mean
        var = noise_distrib.scale ** 2
        assert mean.item() == 0.0
        self.prev_term_params.append((var.item(), y_term))
        self.diffs.append(diff)
        self.n_warm_up = n_warm_up
        self.warmed_up = False
        self.prev_Xs = None
    
    def _mcmc_sample(self, Xs, acc_rate):
        _rnd = torch.rand(acc_rate.size(0)).to(acc_rate)
        pos = _rnd < acc_rate
        self.prev_Xs[:, pos, :] = Xs[:, pos, :]
        return self.prev_Xs[-1]
    
    def _propagate_sample(self, X):
        curr_Xs = []
        curr_Xs.append(X.clone())
        for i_diff in range(len(self.diffs)):
            X = self.diffs[i_diff].propagate(X)
            curr_Xs.append(X.clone())
        curr_Xs = torch.stack(curr_Xs)
        return curr_Xs
    
    def _sample_zero_mean(self, batch_size):
        X = self.sample_init(batch_size)
        curr_Xs = self._propagate_sample(X)
        if self.prev_Xs is None:
            self.prev_Xs = curr_Xs
            return self.prev_Xs[-1]
        c_prev = self._compute_term_factor(self.prev_Xs, self.prev_term_params)
        c_curr = self._compute_term_factor(curr_Xs, self.prev_term_params)
        acc_rate = torch.exp((c_prev - c_curr)/2.).clamp(max=1.)
        return self._mcmc_sample(curr_Xs, acc_rate)
    
    def _sample_sqrt_c_mean(self, batch_size):
        X = self.sample_init(batch_size) # (bs, dim)
        if self.prev_Xs is None:
            curr_Xs = self._propagate_sample(X)
            self.prev_Xs = curr_Xs
            return self.prev_Xs[-1]
        c_prev = self._compute_term_factor(self.prev_Xs, self.prev_term_params)
        X = X + torch.sqrt(c_prev).unsqueeze(-1)
        curr_Xs = self._propagate_sample(X)
        c_curr = self._compute_term_factor(curr_Xs, self.prev_term_params)
        curr_X_hat = curr_Xs[0].squeeze()
        prev_X_hat = self.prev_Xs[0].squeeze()
        acc_rate = torch.exp(
            (c_prev - c_curr) + \
            (prev_X_hat * torch.sqrt(c_curr) - curr_X_hat * torch.sqrt(c_prev)))
        return self._mcmc_sample(curr_Xs, acc_rate)
    
    def sample(self, batch_size):
        '''
        Samples from the process
        :Parameters:
        batch size : int : batch size
        '''
        sample_methods = {
            'zero': self._sample_zero_mean,
            'sqrt_c': self._sample_sqrt_c_mean
        }
        sample_method = sample_methods[self.method]

        if not self.warmed_up:
            for i_warm in range(self.n_warm_up):
                sample_method(batch_size)
            self.warmed_up = True
            return sample_method(batch_size)
        for i_decor in range(self.n_decorrelate):
            sample_method(batch_size)
        return sample_method(batch_size)
    
    def sample_n(self, batch_size):
        return self.sample(batch_size)
    
    def unnorm_log_prob(self, X, diff=None, **diff_log_prob_kwargs):
        diff_log_prob_kwargs['ignore_init'] = True
        diff_log_prob_kwargs['return_X_proto'] = True
        _log_prob = 0.
        if diff is not None:
            X, curr_log_prob = diff.log_prob(X, **diff_log_prob_kwargs)
            _log_prob += curr_log_prob
        for i in range(len(self.diffs)-1, -1, -1):
            var, y_term = self.prev_term_params[i]
            curr_noise_distrib = TD.Normal(
                torch.tensor(y_term).to(X), 
                torch.tensor(np.sqrt(var)).to(X))
            _log_prob += curr_noise_distrib.log_prob(X).view(-1)
            curr_diff = self.diffs[i]
            X, curr_log_prob = curr_diff.log_prob(X, **diff_log_prob_kwargs)
            _log_prob += curr_log_prob
        _log_prob += self.init_distrib.log_prob(X).view(-1)
        return _log_prob.detach()

@dispatch(torch.Tensor, DiffusionFilteringMCMC, Diffusion)
def get_normalized_filtering_log_pdf(xs, df_mcmc, diff, **kwargs):
    nn_log_px = df_mcmc.unnorm_log_prob(xs.view(-1, 1), diff=diff, **kwargs)
    assert(len(xs) > 1)
    dx = xs[1].item() - xs[0].item() # suppose xs to be equidistant grid
    log_px = nn_log_px - torch.logsumexp(nn_log_px + np.log(dx), 0)
    return log_px

@dispatch(torch.Tensor, np.ndarray, DiffusionFilteringMCMC, Diffusion)
def KL_filtering_train_distrib(xs, true_px, df_mcmc, diff, ret_pred_px=False, **kwargs):
    '''
    Estimates KL divergence \int p_train log p_train/p_ref on the grid
    :Parameters:
    xs : torch.Tensor (on appropriate device) : grid (must be equidistant)
    true_px : np.ndarray : reference pdf values on the grid
    df_mcmc : DiffusionFilteringMCMC : 
    diff : Diffusion : last diffusion process 
    '''
    log_pred_px = get_normalized_filtering_log_pdf(xs, df_mcmc, diff, **kwargs).cpu().numpy()
    pxs_log_diff = log_pred_px - np.log(true_px)
    res = np.sum(pxs_log_diff * np.exp(log_pred_px)) * (xs[1] - xs[0]).item()
    if ret_pred_px:
        return res, np.exp(log_pred_px)
    return res

@dispatch(torch.Tensor, np.ndarray, DiffusionFilteringMCMC, Diffusion)
def KL_filtering_targ_distrib(xs, true_px, df_mcmc, diff, **kwargs):
    '''
    Estimates KL divergence \int p_ref log p_ref/p_train.
    See docs for `Kl_filtering_train_distrib` for parameters descr.
    '''
    log_pred_px = get_normalized_filtering_log_pdf(xs, df_mcmc, diff, **kwargs).cpu().numpy()
    pxs_log_diff = np.log(true_px) - log_pred_px
    res = np.sum(pxs_log_diff * true_px) * (xs[1] - xs[0]).item()
    return res

@dispatch(np.ndarray, np.ndarray, Callable)
def KL_filtering_train_distrib(xs, true_px, pdf_callable):
    log_pred_px = np.log(pdf_callable(xs))
    kl = np.sum(true_px * (np.log(true_px) - log_pred_px)) * (xs[1] - xs[0]).item()
    return kl

@dispatch(np.ndarray, np.ndarray, Callable)
def KL_filtering_targ_distrib(xs, true_px, pdf_callable):
    pred_px = pdf_callable(xs)
    kl = np.sum(pred_px * (np.log(pred_px) - np.log(true_px))) * (xs[1] - xs[0]).item()
    return kl

def create_observations_history(t_fin, t_observs, y_observs):
    def _check_strict_ascending(_list):
        for i in range(1, len(_list)):
            assert _list[i] > _list[i - 1]
    def _get_obs(observs, i):
        curr_obs = observs[i]
        if isinstance(curr_obs, (list, np.ndarray)):
            return curr_obs[-1]
        return curr_obs
    _check_strict_ascending(t_observs)
    assert len(t_observs) >= 1.
    assert len(t_observs) == len(y_observs)
    assert t_observs[0] >= 0.
    assert t_observs[-1] <= t_fin
    it_list = [(t_observs[0], True, _get_obs(y_observs, 0))]
    for i in range(1, len(t_observs)):
        it_list.append((
            t_observs[i] - t_observs[i - 1], 
            True, _get_obs(y_observs, i)))
    if t_fin > t_observs[-1]:
        it_list.append((t_fin - t_observs[-1], False, None))
    return it_list

def make_np_normal(trc_normal):
    assert isinstance(trc_normal, TD.Normal)
    _mean = trc_normal.mean.item()
    _scale = trc_normal.stddev.item()
    sps_normal = sps.norm(_mean, _scale)
    return sps_normal

def model_filtering_frogner(
    init_distrib, target, dt, t_fin, t_observs, y_observs, noise_distrib,
    beta=1., n_basis=400, umean=0., uvar=25./3, gamma=1e-6, n_normalize_grid=10000,
    n_zero_spls=10000, verbose=False, opt_method='l-bfgs-b', options={'gtol':1e-8, 'maxiter':5000}):
    '''
    Models nonlinear filtering using CFrogner's dual JKO method 
    (see http://proceedings.mlr.press/v108/frogner20a.html)
    1D case only!
    :Parameters:
    init_distrib : torch.Distribution : initial distribution of the points
    target : class : models target potential (see `changcooper.py` for more details)
    dt : float : time interval of JKO method
    t_observs : list : list of observation times
    y_observs : list : observations out of the process
    noise_distrib : normal (torch or scipy.stats) distribution  : normal noise of providing the observations
    Regarding the other parameters see `src.frogner.py`
    '''
    # default parameters for CFrogner's method
    assert beta == 1.
    kernel_type='gaussian'
    kernel_params = {'sigmasq': 1e-1}
    do_uniform=True
    do_hess=False
    nufunc_norm_constant=None
    # target potential function
    def targ_pot(x):
        return target.potential(x)
    # create observation-time history
    obs_hist = create_observations_history(t_fin, t_observs, y_observs)
    dim = 1
    # ensure the noise distrib to be sps
    sps_noise = noise_distrib if isinstance(noise_distrib, type(sps.norm())) \
        else make_np_normal(noise_distrib)
    # frogner's distribution parameters
    meanx_z_spl = np.zeros(dim) + umean
    meany_z_spl = np.zeros(dim) + umean
    covx_z_spl = np.eye(dim)*uvar
    covy_z_spl = np.eye(dim)*uvar
    # initial distribution function
    init_pdf = lambda x : np.exp(init_distrib.log_prob(torch.tensor(x)).cpu().numpy()).reshape(-1)
    # list with initial distributions at observation times
    obs_times_pdfs = [init_pdf,]
    # extract parameters of uniform region where optimization take place
    loc, scale = get_mv_uniform_loc_scale_params(umean, uvar)
    lft_b, rght_b = loc.item(), (loc + scale).item()
    if verbose:
        print(f"uniform region params: left bound: {lft_b}, right_bound: {rght_b}")
    norm_grid = np.linspace(lft_b, rght_b, n_normalize_grid, endpoint=True).reshape(1, -1)
    test_vals = np.linspace(-1, 1, 10).reshape(1, -1)
    # pdf normalizer
    def normalize_pdf(pdf0):
        diff_vals = pdf0(norm_grid)
        diff_int = normalize_grid(diff_vals, norm_grid.reshape(-1), ret_int=True)
        def normalized_pdf(x):
            return pdf0(x) / diff_int
        return normalized_pdf
    
    # make marginal posterior distributions at observation time
    def make_marginal_posterior(pdf0, y_obs, normalize=False):
        def obs_pdf(x):
            return sps_noise.pdf(x - y_obs).reshape(-1) * pdf0(x)
        if normalize:
            return normalize_pdf(obs_pdf)
        return obs_pdf

    for diff_t, to_sample, y_obs in obs_hist:

        # initial pdf used on current step
        curr_init_pdf = obs_times_pdfs[-1]

        # _normalize_pdf(curr_init_pdf)

        if diff_t > 0:
            # frogner's iteration
            curr_n_iters = int(np.ceil(diff_t/dt))
            # frogner's tau
            curr_tau = diff_t/float(curr_n_iters)
            if verbose:
                print(f">>>>>> start diffusion, diff_t: {diff_t}, n_iters: {curr_n_iters}, curr_tau: {curr_tau}")
            # obtain marginal prior of the next observation time
            curr_next_pdf0 = run_diffusion(
                curr_init_pdf, targ_pot, diff_t, curr_n_iters, dim, n_basis, 
                meanx=meanx_z_spl, covx=covx_z_spl, meany=meany_z_spl, covy=covy_z_spl, 
                beta=beta, gamma=gamma, kernel_type=kernel_type, kernel_params=kernel_params, 
                n_zero_spls=n_zero_spls, verbose=verbose, do_uniform=do_uniform, do_hess=do_hess, 
                nufunc_norm_constant=nufunc_norm_constant, opt_method=opt_method, options=options)
        else:
            curr_next_pdf0 = curr_init_pdf
        
        if to_sample:
            # make normalized marginal posterior
            curr_next_pdf = make_marginal_posterior(curr_next_pdf0, y_obs, normalize=True)
        else:
            curr_next_pdf = normalize_pdf(curr_next_pdf0)

        # update the array with new pdf
        obs_times_pdfs.append(curr_next_pdf)
    _final_func = obs_times_pdfs[-1]
    def final_func(x):
        return _final_func(x.reshape(1, -1))
    return final_func

def model_filtering_bbf(
    init_distrib, target, n_samples, dt, t_fin, 
    t_observs, y_observs, noise_distrib, beta=1.):
    '''
    Models nonlinear filtering using Bayesian bootstrap filter 
    (see `Novel approach to nonlinear/non-Gaussian Bayesian state estimation` by Gordon et.al.)
    :Parameters:
    init_distrib : torch.Distribution : initial distribution of the points
    target : class : models target potential (see `changcooper.py` for more details)
    n_samples : int : count of samples to propagate through the diffusion process
    dt : float : time interval used in Euler-Maruyaama iterations 
    t_fin : float : final time of the diffusion
    t_observs : list : list of observation times
    y_observs : list : observations out of the process
    noise_distrib : normal (torch or scipy.stats) distribution  : normal noise of providing the observations
    beta : float : process temperature
    '''
    # function to use in EM iterations
    def minus_grad_potential(x):
        return - target.grad_potential(x)
    # create observation-time history
    obs_hist = create_observations_history(t_fin, t_observs, y_observs)
    # ensure the noise distrib to be sps
    sps_noise = noise_distrib if isinstance(noise_distrib, type(sps.norm())) \
        else make_np_normal(noise_distrib)
    # sample from the init_distrib 
    #TODO: consider changing init_distrib to be sps
    x = init_distrib.sample((n_samples, 1)).cpu().numpy()
    for diff_t, to_sample, y_obs in obs_hist:
        # propagate through the diffusion process
        x = batchItoEuler(minus_grad_potential, x, dt, diff_t, beta=beta) # (n, 1)
        if to_sample:
            # create the probas to resample
            log_cond_x = sps_noise.logpdf(x - y_obs)
            denom = scipy.special.logsumexp(log_cond_x)
            qs = np.exp(log_cond_x - denom).reshape(-1)
            # resample with qs:
            new_ids = np.random.choice(n_samples, n_samples, p=qs)
            x = x[new_ids]
    return x

def model_filtering_cc(
    init_distrib, target, xs, dt, t_fin, 
    t_observs, y_observs, noise_distrib, beta=1.):
    '''
    Models nonlinear filtering using Chang&Cooper iterations
    :Parameters:
    init_distrib : torch.Distribution : initial distribution of the points
    target : class : models target potential (see `changcooper.py` for more details)
    xs : torch.Tensor or np.ndarray : grid to evaluate the pdf (on the grid)
    dt : float : time interval used in the numerical propagating method
    t_fin : float : final time fo the diffusion (start time is 0)
    sample_times : iterable : ascending non-negative moments of time 
    of the process observation (with the noise)
    using Euler-Maruyama iterations
    noise_distrib : torch.Distribution : noise of providing the observations
    beta : float : process temperature 
    '''
    # estimate init distrib on the grid
    px = discretize_distrib(xs, init_distrib).astype(np.float64)
    if isinstance(xs, torch.Tensor):
        xs = xs.cpu().numpy()
    xs = xs.astype(np.float64)
    # create observation-time history
    obs_hist = create_observations_history(t_fin, t_observs, y_observs)

    for diff_t, to_sample, y_obs in obs_hist:

        if diff_t > 0:
            n_pdf_prop_iters = int(np.ceil(diff_t/dt))
            curr_dt = diff_t/float(n_pdf_prop_iters)
            for i_prop in range(n_pdf_prop_iters):
                px = iterate_diffusion_cc(target, px, xs, curr_dt, beta=beta)
        
        # sample from the process with noise
        if to_sample:

            # update px with new pdf 
            noise_stddev = noise_distrib.scale.item()
            cond_distrib = sps.norm(y_obs, noise_stddev)
            add_px = np.exp(cond_distrib.logpdf(xs)).astype(np.float64)
            px = normalize_grid(px * add_px, xs)
    
    return px

def model_observations(
    init_distrib, target, sample_times, em_dt, 
    noise_distrib, beta=1., np_seed=None, trc_seed=None):
    '''
    Models the observations from the process utilizing Euler Maruyama iterations
    :Parameters:
    init_distrib : torch.Distribution : initial distribution of the points
    target : class : models target potential (see `changcooper.py` for more details)
    sample_times : iterable : ascending non-negative moments of time 
    of the process observation (with the noise)
    em_dt : float : time interval used in EM iterations
    noise_distrib : torch.Distribution : noise of providing the observations
    beta : float : process temperature
    '''
    # set random seeds if any:
    assert type(np_seed) == type(trc_seed)
    if np_seed is not None:
        np.random.seed(np_seed)
        torch.random.manual_seed(trc_seed)
    # sample from the init distribution
    x_p = init_distrib.sample((1,)).item()
    prev_t = 0. # time from which start the diffusion
    noise_sampled = []
    for spl_time in sample_times:
        diff_t = spl_time - prev_t

        def f(x, t):
            return - target.grad_potential(x)

        def g(x, t):
            return np.ones_like(x) * np.sqrt(2./beta)
        
        if diff_t > 0:
            n_em_prop_iters = int(diff_t/em_dt)
            t_span = np.linspace(0.0, diff_t, n_em_prop_iters+1, endpoint=True)
            # propagate through the process
            result = sdeint.itoEuler(f, g, x_p, t_span)
            x_p = result[-1].item()
        z_p = noise_distrib.sample((1,)).item()
        obs_p = z_p + x_p
        # sample from the process, observation
        noise_sampled.append([x_p, obs_p]) 
        prev_t = spl_time
    return noise_sampled


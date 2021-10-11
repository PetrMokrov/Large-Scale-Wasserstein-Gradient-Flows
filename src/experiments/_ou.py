import numpy as np
import torch.distributions as TD
import torch
from scipy.stats import gaussian_kde
from sklearn.datasets import make_spd_matrix
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import os, sys
import time

from ..ou import get_normal_distrib_params, create_em_samples
from ..ou import OU_tNormal, create_ou_distrib_modeler
from ..ou import get_normal_distrib_params, create_em_samples
from ..ou import get_ou_potential_func
from ..ou import create_em_proxrec_samples
from ..em_proxrec import normalize_pdf_reference, normalize_pdf_kde
from ..em_proxrec import KL_targ_distrib_importance as em_proxrec_KL_targ
from ..em_proxrec import  KL_train_distrib as em_proxrec_KL_train
from ..utils import id_pretrain_model, train_diffusion_model
from ..utils import KL_train_distrib, KL_targ_distrib, energy_based_distance
from ..icnn import DenseICNN
from ..diffusion import Diffusion
from ..frogner import run_diffusion
from ..frogner import normalize_unnorm_pdf, KL_train_distrib_importance, KL_targ_distrib
from ..frogner import accept_reject_sample

from .manual_random import get_random_manager
from .exp_file_manager import OU_fixed_dim_EFM, OU_vary_dim_EFM

def generate_target(dim, mean_scale=1., dtype=torch.float32, device='cpu'):
    # this function generates target normal distribution
    var = make_spd_matrix(dim)
    mean = np.random.randn(dim) * mean_scale
    trc_var = torch.tensor(var, dtype=dtype).to(device)
    trc_mean = torch.tensor(mean, dtype=dtype).to(device)
    targ_distrib = TD.MultivariateNormal(trc_mean, trc_var)
    init = np.random.randn(dim) * mean_scale
    return targ_distrib, mean, var

def ou_dual_jko_fixed_dimension_experiment(config):
    dim = config['dim']
    exp_numbers = config['exp_numbers']
    t_fin = config['t_fin']
    jko_dt = config['dt']
    jko_dt_estimation = config['dt_estimation']
    lbfgs_maxiter = config['lbfgs_maxiter']
    lbfgs_gtol = config['lbfgs_gtol']
    lbfgs_options = {'gtol': lbfgs_gtol, 'maxiter': lbfgs_maxiter}
    n_basis = config['n_basis']
    kernel_type = config['kernel_type']
    supp_var = config['supp_variance']
    jko_n_steps = round(t_fin/jko_dt)
    assert jko_n_steps * jko_dt == t_fin
    jko_steps_estimation = round(jko_dt_estimation/ jko_dt)
    assert jko_steps_estimation * jko_dt == jko_dt_estimation
    verbose = config['verbose']
    exp_name = config['experiment_name']
    init_variance = config['init_variance']
    n_eval_spls = config['n_eval_samples']
    file_manager = config['file_manager'] if 'file_manager' in config else OU_fixed_dim_EFM.fromconfig(config)
    
    # diffusion estimation intervals
    diff_est_params = []
    for i_est in range(jko_n_steps // jko_steps_estimation):
        diff_est_params.append((
            i_est, # iteration number
            jko_steps_estimation, # number of diffusion steps to perform
            (i_est + 1) * jko_dt_estimation # diffusion timestep 
            ))
    
    for exp_number in exp_numbers:
        exp_results = defaultdict(list)
        if verbose:
            print(f'start experiment: {exp_name}, dim: {dim}, number: {exp_number}')
        # seed all random sources
        r_m = get_random_manager(config['random_key'], dim, exp_number)
        r_m.seed()
        # create init sampler
        sampler = TD.MultivariateNormal(
            torch.zeros(dim, dtype=torch.float64), 
            init_variance * torch.eye(dim, dtype=torch.float64))
        
        # create target (stationary) distribution
        target, targ_mean, targ_var = generate_target(dim, dtype=torch.float64)
        
        # Ornstein-Uhlenbeck process modeller
        ou_d_m = create_ou_distrib_modeler(target)
        
        # pdf functions
        log_init_pdf = lambda x : sampler.log_prob(torch.tensor(x).T).numpy()
        init_pdf = lambda x : np.exp(log_init_pdf(x))
        targ_pot = get_ou_potential_func(targ_mean, targ_var)
        curr_init_pdf = init_pdf
        
        for i_est, curr_jko_steps, diff_tstp in diff_est_params:
            if verbose:
                print(f'Start diffusion iteration: {i_est}, curr_jko_steps: {curr_jko_steps}')
            
            t_tr_strt = time.perf_counter()
            # run diffusion on current time interval
            curr_final_unnorm_pdf = run_diffusion(
                curr_init_pdf, targ_pot, jko_dt_estimation, curr_jko_steps, dim, n_basis, kernel_type=kernel_type, 
                covx = np.eye(dim)*supp_var, covy = np.eye(dim)*supp_var, options=lbfgs_options, 
                verbose=verbose)
            t_tr_el = time.perf_counter() - t_tr_strt
            
            true_distrib = OU_tNormal(sampler, ou_d_m, diff_tstp)
            true_sample = true_distrib.sample((n_eval_spls,))
            norm_constant = normalize_unnorm_pdf(true_sample, curr_final_unnorm_pdf, true_distrib)
            if verbose:
                print(f'Start diffusion evaluation')
            
            t_ev_strt = time.perf_counter()
            exp_results['kl_train'].append((diff_tstp, KL_train_distrib_importance(
                true_sample, curr_final_unnorm_pdf, true_distrib, norm_constant=norm_constant)))
            exp_results['kl_target'].append((diff_tstp, KL_targ_distrib(
                true_sample, curr_final_unnorm_pdf, true_distrib, norm_constant=norm_constant)))
            train_sample = accept_reject_sample(
                n_eval_spls, curr_final_unnorm_pdf, true_distrib, norm_constant=norm_constant)
            exp_results['energy_based'].append((
                diff_tstp, energy_based_distance(
                    train_sample, true_sample, dim=dim)))
            t_ev_el = time.perf_counter() - t_ev_strt
            
            stat_sample = target.sample((n_eval_spls,))
            exp_results['kl_stationary_train'].append((diff_tstp, KL_train_distrib_importance(
                stat_sample, curr_final_unnorm_pdf, target, norm_constant=norm_constant)))
            exp_results['kl_stationary_target'].append((diff_tstp, KL_targ_distrib(
                stat_sample, curr_final_unnorm_pdf, target, norm_constant=norm_constant)))
            exp_results['time_train'].append(
                (diff_tstp, t_tr_el))
            exp_results['time_est'].append(
                (diff_tstp, t_ev_el))
            file_manager.save(exp_results, exp_number)

            curr_init_pdf = curr_final_unnorm_pdf

def ou_em_fixed_dimension_experiment(config):
    # unpack the config
    dim = config['dim']
    exp_numbers = config['exp_numbers']
    t_fin = config['t_fin']
    em_dt = config['dt']
    n_particles = config['n_particles']
    n_eval_samples = config['n_eval_samples']
    if n_eval_samples == -1:
        n_eval_samples = n_particles
    dt_estimation = config['dt_estimation']
    verbose = config['verbose']
    exp_name = config['experiment_name']
    init_variance = config['init_variance']
    file_manager = config['file_manager'] if 'file_manager' in config else OU_fixed_dim_EFM.fromconfig(config)
    
    for exp_number in exp_numbers:
        exp_results = defaultdict(list)
        if verbose:
            print(f'start experiment: {exp_name}, dim: {dim}, number: {exp_number}')
        # seed all random sources
        r_m = get_random_manager(config['random_key'], dim, exp_number)
        r_m.seed()
        # create init sampler
        sampler = TD.MultivariateNormal(
            torch.zeros(dim), 
            init_variance * torch.eye(dim))
        
        # create target (stationary) distribution
        target, mean, var = generate_target(dim)
        
        # Ornstein-Uhlenbeck process modeller
        ou_d_m = create_ou_distrib_modeler(target)
        
        # sample particles from the initial distribution
        x0 = sampler.sample((n_particles,)).cpu().numpy()
        prev_em_samples = x0.copy()
        for i_tstp in range(int(t_fin / dt_estimation)):
            curr_tstp = (i_tstp + 1) * dt_estimation
            if verbose:
                print(f"start step {i_tstp}, timestamp: {curr_tstp}")
            t_tr_strt = time.perf_counter() 
            curr_em_samples = create_em_samples(
                prev_em_samples, target, dt_estimation, em_dt, n_particles)
            curr_em_distrib = gaussian_kde(curr_em_samples.transpose())
            t_tr_elaps = time.perf_counter() - t_tr_strt
            
            true_distrib = OU_tNormal(sampler, ou_d_m, curr_tstp)
            # KL divergence evaluation
            t_ev_strt = time.perf_counter()
            # KL wrt train est.
            exp_results['kl_train'].append(
                (curr_tstp, KL_train_distrib(
                     curr_em_samples[:n_eval_samples, :], 
                     curr_em_distrib, true_distrib, device='cpu')))
            # KL wrt target est.
            curr_kl_target, X_targ = KL_targ_distrib(
                n_eval_samples, curr_em_distrib, true_distrib, ret_targ_sample=True)
            exp_results['kl_target'].append(
                (curr_tstp, curr_kl_target))
            # Energy-based dist. est.
            exp_results['energy_based'].append((
                curr_tstp, energy_based_distance(
                    curr_em_samples[:n_eval_samples, :], X_targ, dim=dim)))
            t_ev_elaps = time.perf_counter() - t_ev_strt
            exp_results['time_train'].append(
                (curr_tstp, t_tr_elaps))
            exp_results['time_est'].append(
                (curr_tstp, t_ev_elaps))
            prev_em_samples = curr_em_samples
        
        file_manager.save(exp_results, exp_number)

def ou_em_proxrec_fixed_dimension_experiment(config):

    dim = config['dim']
    exp_numbers = config['exp_numbers']
    t_fin = config['t_fin']
    em_dt = config['dt']
    n_particles = config['n_particles']
    n_eval_samples = config['n_eval_samples']
    if n_eval_samples == -1:
        n_eval_samples = n_particles
    dt_estimation = config['dt_estimation']
    verbose = config['verbose']
    exp_name = config['experiment_name']
    init_variance = config['init_variance']
    file_manager = config['file_manager'] if 'file_manager' in config else OU_fixed_dim_EFM.fromconfig(config)
    dtype = config['dtype']
    device = config['device']
    dtype = torch.float32 if dtype == 'float32' else torch.float64

    for exp_number in exp_numbers:
        exp_results = defaultdict(list)
        if verbose:
            print(f'start experiment: {exp_name}, dim: {dim}, number: {exp_number}')
        # seed all random sources
        r_m = get_random_manager(config['random_key'], dim, exp_number)
        r_m.seed()
        # create init sampler
        sampler = TD.MultivariateNormal(
            torch.zeros(dim, dtype=dtype, device=device), 
            init_variance * torch.eye(dim, dtype=dtype, device=device))
        
        # create target (stationary) distribution
        target, mean, var = generate_target(dim, device=device, dtype=dtype)
        
        # Ornstein-Uhlenbeck process modeller
        ou_d_m = create_ou_distrib_modeler(target)
        
        # sample particles from the initial distribution
        prev_em_samples = sampler.sample((n_particles,))
        prev_pdf = torch.exp(sampler.log_prob(prev_em_samples))

        for i_tstp in range(int(t_fin / dt_estimation)):
            curr_tstp = (i_tstp + 1) * dt_estimation
            if verbose:
                print(f"start step {i_tstp}, timestamp: {curr_tstp}")
            t_tr_strt = time.perf_counter() 

            curr_em_samples, curr_pdf = create_em_proxrec_samples(
                prev_em_samples, prev_pdf, target, dt_estimation, em_dt, verbose=verbose, 
                reg=config['proxrec_reg'], tol=config['proxrec_tol'], maxiter=config['proxrec_maxiter'])

            t_tr_elaps = time.perf_counter() - t_tr_strt
            
            true_distrib = OU_tNormal(sampler, ou_d_m, curr_tstp)

            # normalize distribution
            curr_norm_constant = normalize_pdf_kde(curr_em_samples, curr_pdf)
            curr_pdf = curr_pdf / curr_norm_constant

            # KL divergence evaluation
            t_ev_strt = time.perf_counter()
            # KL wrt train est.
            exp_results['kl_train'].append(
                (curr_tstp, 
                em_proxrec_KL_train(
                    curr_em_samples[:n_eval_samples, :], curr_pdf[:n_eval_samples], true_distrib)))
            # KL wrt target est.
            exp_results['kl_target'].append(
                (curr_tstp, em_proxrec_KL_targ(
                    curr_em_samples[:n_eval_samples], curr_pdf[:n_eval_samples], true_distrib)))
            t_ev_elaps = time.perf_counter() - t_ev_strt
            exp_results['time_train'].append(
                (curr_tstp, t_tr_elaps))
            exp_results['time_est'].append(
                (curr_tstp, t_ev_elaps))
            prev_em_samples = curr_em_samples
            prev_pdf = curr_pdf
        
        # print(exp_results)
        file_manager.save(exp_results, exp_number)

def ou_icnn_jko_fixed_dimension_experiment(config):

    # experiment parameters extraction
    batch_size = config['batch_size']
    device = config['device']
    dim = config['dim']
    exp_numbers = config['exp_numbers']
    t_fin = config['t_fin']
    jko_dt = config['dt']
    jko_dt_estimation = config['dt_estimation']
    jko_step_iteration = config['n_step_iterations']
    jko_n_eval_samples = config['n_eval_samples']
    jko_n_steps = round(t_fin/jko_dt)
    assert jko_n_steps * jko_dt == t_fin
    jko_steps_estimation = round(jko_dt_estimation/ jko_dt)
    assert jko_steps_estimation * jko_dt == jko_dt_estimation
    verbose = config['verbose']
    exp_name = config['experiment_name']
    init_variance = config['init_variance']
    n_layers = config['n_ICNN_layers']
    layer_width = config['ICNN_width']
    n_max_prop = config['ICNN_n_max_prop']
    lr = config['learning_rate']
    pretrain_lr = config['pretrain_learning_rate']
    file_manager = config['file_manager'] if 'file_manager' in config else OU_fixed_dim_EFM.fromconfig(config)

    # diffusion estimation intervals
    diff_est_params = []
    for i_est in range(jko_n_steps // jko_steps_estimation):
        diff_est_params.append((
            i_est, # iteration number
            jko_steps_estimation, # number of diffusion steps to perform
            (i_est + 1) * jko_dt_estimation # diffusion timestep 
            ))

    # starting the experiments
    for exp_number in exp_numbers:
        exp_results = defaultdict(list)
        if verbose:
            print(f'start experiment: {exp_name}, dim: {dim}, number: {exp_number}')
        # seed all random sources
        r_m = get_random_manager(config['random_key'], dim, exp_number)
        r_m.seed()
        # create init sampler
        sampler = TD.MultivariateNormal(
            torch.zeros(dim, device=device), 
            init_variance * torch.eye(dim, device=device))
        
        # create target (stationary) distribution
        target, mean, var = generate_target(dim, device=device)

        # Ornstein-Uhlenbeck process modeller
        ou_d_m = create_ou_distrib_modeler(target)

        # create ICNN base model
        model_args = [dim, [layer_width,]*n_layers]
        model_kwargs = {
            'rank':5, 
            'activation':'softplus', 
            'batch_size':batch_size} #TODO: consider additional config parameters here
        D0 = DenseICNN(*model_args, **model_kwargs).to(device)

        # initialize the model
        for p in D0.parameters():
            p.data = torch.randn(
                p.shape, device=device, 
                dtype=torch.float32) / np.sqrt(float(layer_width))
        
        # pretrain the model (to be identity function)
        D0 = id_pretrain_model(
            D0, sampler, lr=pretrain_lr, 
            n_max_iterations=4000, batch_size=batch_size, verbose=verbose)
        
        # start_diffusion_process
        diff = Diffusion(sampler, n_max_prop=batch_size)
        if verbose:
            X_test = sampler.sample_n(batch_size).view(-1, dim) #TODO: consider additional config parameter
        else:
            X_test = None
        
        for i_est, curr_jko_steps, diff_tstp in diff_est_params:
            if verbose:
                print(f'Start diffusion iteration: {i_est}, curr_jko_steps: {curr_jko_steps}')
            t_tr_strt = time.perf_counter() #TODO: make context manager instead
            diff = train_diffusion_model(
                diff, D0 if i_est == 0 else None, 
                (model_args, model_kwargs), 
                target, n_steps=curr_jko_steps, 
                step_iterations=jko_step_iteration,
                n_max_prop=n_max_prop,
                step_size=jko_dt, batch_size=batch_size, 
                X_test=X_test, lr=lr, device=device, 
                plot_loss=False, verbose=verbose)
            t_tr_elaps = time.perf_counter() - t_tr_strt
            if isinstance(diff, tuple):
                diff = diff[0]
            if verbose:
                print(f'Start diffusion estimation at timestep {diff_tstp}')
            # true distribution at diff_tstp timestep
            true_distrib = OU_tNormal(sampler, ou_d_m, diff_tstp)
            # KL wrt train and target distributions estimation
            t_est_strt = time.perf_counter()
            curr_kl_train, diff_X = KL_train_distrib(
                jko_n_eval_samples, diff, true_distrib, ret_diff_sample=True)
            curr_kl_target, targ_X = KL_targ_distrib(
                jko_n_eval_samples, diff, true_distrib, ret_targ_sample=True)
            # KL wrt train
            exp_results['kl_train'].append((diff_tstp, curr_kl_train.item()))
            # KL wrt target
            exp_results['kl_target'].append((diff_tstp, curr_kl_target.item()))
            # Energy-based dist.
            exp_results['energy_based'].append((
                diff_tstp, energy_based_distance(diff_X, targ_X, dim=dim).item()))
            t_est_elaps = time.perf_counter() - t_est_strt
            exp_results['time_train'].append(
                (diff_tstp, t_tr_elaps))
            exp_results['time_est'].append(
                (diff_tstp, t_est_elaps))
            file_manager.save(exp_results, exp_number)
        
        # saving results of the experiment
        file_manager.save(exp_results, exp_number)

def vary_dimension_experiment(config, fix_dim_func):
    dim_min = config['dim_min']
    dim_max = config['dim_max']
    verbose = config['verbose']
    exp_name = config['experiment_name']
    file_manager = config['file_manager'] if 'file_manager' in config else OU_vary_dim_EFM.fromconfig(config)
    if verbose:
        print(f'Start experiments: {exp_name}, dim_min: {dim_min}, dim_max: {dim_max}')
    fixed_dim_file_managers = {}
    for dim in range(dim_min, dim_max + 1):
        if verbose:
            print(f'Start dimension {dim}')
        curr_config = deepcopy(config)
        if 'dim_specs' in curr_config:
            for spec in curr_config['dim_specs'].values():
                if spec['dim'] == dim:
                    if verbose:
                        print(f'Specification detected, dim={dim}')
                    for key, value in spec.items():
                        if verbose:
                            print(f'Specification: key: {key}, value: {value}')
                        curr_config[key] = value
        curr_config['dim'] = dim
        fixed_dim_file_managers[dim] = OU_fixed_dim_EFM.fromconfig(curr_config, temporary=True)
        curr_config['file_manager'] = fixed_dim_file_managers[dim]
        fix_dim_func(curr_config)
    # collect the results
    exp_numbers = config['exp_numbers']
    for n_exp in exp_numbers:
        exp_results = {}
        for dim in range(dim_min, dim_max + 1):
            exp_results[dim] = fixed_dim_file_managers[dim].load(n_exp)
        file_manager.save(exp_results, n_exp)
    # remove temporary dirs
    for dim in range(dim_min, dim_max + 1):
        fixed_dim_file_managers[dim].rm_dir()

def ou_icnn_jko_vary_dimensions_experiment(config):
    return vary_dimension_experiment(config, ou_icnn_jko_fixed_dimension_experiment)

def ou_em_vary_dimensions_experiment(config):
    return vary_dimension_experiment(config, ou_em_fixed_dimension_experiment)

def ou_dual_jko_vary_dimensions_experiment(config):
    return vary_dimension_experiment(config, ou_dual_jko_fixed_dimension_experiment)

def ou_em_proxrec_vary_dimensions_experiment(config):
    return vary_dimension_experiment(config, ou_em_proxrec_fixed_dimension_experiment)

import torch
import numpy as np
import torch.distributions as TD
from ..utils import id_pretrain_model, train_diffusion_model
from ..utils import KL_train_distrib, KL_targ_distrib, energy_based_distance
from ..em import torchBatchItoEulerDistrib
from .manual_random import get_random_manager
from .exp_file_manager import Convergence_EFM, ConvergenceComparison_EFM
from ..icnn import DenseICNN
from ..diffusion import TargetedDiffusion
import time
import math
from scipy.stats import gaussian_kde
from ..em_proxrec import torchBatchItoEulerProxrec, normalize_pdf_kde, normalize_pdf_reference
from ..em_proxrec import KL_targ_distrib_importance as em_proxrec_KL_targ
from ..em_proxrec import  KL_train_distrib as em_proxrec_KL_train

def random_centers_distrib_generator(n, n_rand, centers_sample_distrib, std, device='cpu', dtype=torch.float32):
    centers = centers_sample_distrib.sample((n_rand, n)).to(device, dtype)
    comp = TD.Independent(TD.Normal(centers, torch.tensor([std,]).to(device, dtype)), 1)
    mix = TD.Categorical(torch.ones(n_rand,).to(device, dtype))
    target = TD.MixtureSameFamily(mix, comp)
    return target

def special_distrib_generator(dim, n_centers, dist, std, standardized=True, device='cpu'):
    centers = np.zeros((n_centers, dim), dtype=np.float32)
    for d in range(dim):
        idx = np.random.choice(list(range(n_centers)), n_centers, replace=False)
        centers[:, d] += dist * idx
    centers -= dist * (n_centers - 1) / 2

    maps = np.random.normal(size=(n_centers, dim, dim)).astype(np.float32)
    maps /= np.sqrt((maps ** 2).sum(axis=2, keepdims=True))
    
    if standardized:
        mult = np.sqrt((centers ** 2).sum(axis=1).mean() + dim * std ** 2) / np.sqrt(dim)
        centers /= mult
        maps /= mult
    covars = np.matmul(maps, maps.transpose((0, 2, 1)))* (std ** 2)
    trc_centers = torch.tensor(centers, device=device, dtype=torch.float32)
    trc_covars = torch.tensor(covars, device=device, dtype=torch.float32)
    mv_normals = TD.MultivariateNormal(trc_centers, trc_covars)
    mix = TD.Categorical(torch.ones(n_centers,).to(device))
    target = TD.MixtureSameFamily(mix, mv_normals)
    return target

def create_stationary_distrib(config, device, dtype=torch.float32):
    if config['stationary_type'] == 'cube':
        stationary_distrib = random_centers_distrib_generator(
            config['dim'], config['n_centers'], 
            TD.Uniform(-config['target_span'], config['target_span']), 
            config['target_std'], device, dtype)
    elif config['stationary_type'] == 'normalized':
        if dtype != torch.float32:
            raise Exception('only float32 supported for "normalized" stationary')
        stationary_distrib = special_distrib_generator(
            config['dim'], config['n_centers'], config['target_span'], 
            config['target_std'], device=device)
    return stationary_distrib
    

def convergence_mix_gauss_targ_experiment(config):
    dim = config['dim']
    device = config['device']
    verbose = config['verbose']
    file_manager = config['file_manager'] if 'file_manager' in config else Convergence_EFM.fromconfig(config)
    r_m = get_random_manager(config['random_key'], dim)
    r_m.seed()
    # creates stationary distribution
    stationary_distrib = create_stationary_distrib(config, device)
    
    init_distrib = TD.MultivariateNormal(
        torch.zeros(dim).to(device), config['init_variance'] * torch.eye(dim).to(device))
    
    # create ICNN base model
    batch_size= config['batch_size']
    model_args = [dim, [config['layer_width'],]*config['n_layers']]
    model_kwargs = {
        'rank':5, 
        'activation':'softplus', 
        'batch_size':batch_size} #TODO: consider additional config parameters here
    D0 = DenseICNN(*model_args, **model_kwargs).to(device)

    # initialize the model
    for p in D0.parameters():
        p.data = torch.randn(
            p.shape, device=device, 
            dtype=torch.float32) / np.sqrt(float(config['layer_width']))

    # pretrain the model (to be identity function)
    D0 = id_pretrain_model(
        D0, init_distrib, lr=config['pretrain_lr'], 
        n_max_iterations=4000, batch_size=batch_size, verbose=verbose)
    
    diff = TargetedDiffusion(init_distrib, stationary_distrib, n_max_prop=config['n_max_prop'])
    X_test = init_distrib.sample((batch_size,)).view(-1, dim)
    kl_train = []
    
    for n_iters, lr in config['lr']:
        diff, curr_kl_train = train_diffusion_model(
            diff, D0 if len(diff.Ds) == 0 else None, 
            (model_args, model_kwargs), 
            stationary_distrib, n_steps=n_iters, 
            step_iterations=config['n_step_iterations'],
            n_max_prop=config['n_max_prop'],
            step_size=config['dt'], batch_size=batch_size, 
            X_test=X_test, lr=lr, device=device, 
            plot_loss=False, verbose=verbose)
        kl_train.extend(curr_kl_train)
    file_manager.save_model(diff)
    dict_res = {'kl_train': kl_train}
    file_manager.save(dict_res)

def conv_comp_icnn_jko_mix_gauss_targ_experiment(config):
    dim = config['dim']
    device = config['device']
    verbose = config['verbose']
    file_manager = config['file_manager'] if 'file_manager' in config else ConvergenceComparison_EFM.fromconfig(config)
    t_train = config['dt'] * float(np.sum([n_iters for n_iters, _ in config['lr']]))
    assert math.isclose(t_train, config['t_fin'], abs_tol=1e-5)
    
    for exp_number in config['exp_numbers']:
        if verbose:
            print(f'[ICNN jko convergence № {exp_number}] starts: dim: {dim}')
        # seed all random sources
        r_m = get_random_manager(config['random_key'], dim, exp_number)
        r_m.seed()
        # creates stationary distribution
        stationary_distrib = create_stationary_distrib(config, device)
        
        # create init sampler
        init_distrib = TD.MultivariateNormal(
            torch.zeros(dim).to(device), config['init_variance'] * torch.eye(dim).to(device))
        
        # create base ICNN model
        batch_size= config['batch_size']
        model_args = [dim, [config['layer_width'],]*config['n_layers']]
        model_kwargs = {
            'rank':5, 
            'activation':'softplus', 
            'batch_size':batch_size}
        D0 = DenseICNN(*model_args, **model_kwargs).to(device)

        # initialize the model
        for p in D0.parameters():
            p.data = torch.randn(
                p.shape, device=device, 
                dtype=torch.float32) / np.sqrt(float(config['layer_width']))

        # pretrain the model (to be identity function)
        D0 = id_pretrain_model(
            D0, init_distrib, lr=config['pretrain_lr'], 
            n_max_iterations=4000, batch_size=batch_size, verbose=verbose)

        diff = TargetedDiffusion(init_distrib, stationary_distrib, n_max_prop=config['n_max_prop'])
        X_test = init_distrib.sample((batch_size,)).view(-1, dim)
        
        t_tr_strt = time.perf_counter()
        for n_iters, lr in config['lr']:
            diff, curr_kl_train = train_diffusion_model(
                diff, D0 if len(diff.Ds) == 0 else None, 
                (model_args, model_kwargs), 
                stationary_distrib, n_steps=n_iters, 
                step_iterations=config['n_step_iterations'],
                n_max_prop=config['n_max_prop'],
                step_size=config['dt'], batch_size=batch_size, 
                X_test=X_test, lr=lr, device=device, 
                plot_loss=False, verbose=verbose)
        t_tr_elaps = time.perf_counter() - t_tr_strt
        
        if verbose:
            print(f'[ICNN jko convergence № {exp_number}]: diffusion estimation')
            # true distribution at diff_tstp timestep
        exp_results = {}
        t_est_strt = time.perf_counter()
        curr_kl_train, diff_X = KL_train_distrib(
            config['n_eval_samples'], diff, stationary_distrib, ret_diff_sample=True)
        curr_kl_target, targ_X = KL_targ_distrib(
            config['n_eval_samples'], diff, stationary_distrib, ret_targ_sample=True)
        # KL wrt train
        exp_results['kl_train'] = curr_kl_train.item()
        # KL wrt target
        exp_results['kl_target'] = curr_kl_target.item()
        # Energy-based dist.
        exp_results['energy_based'] = energy_based_distance(diff_X, targ_X, dim=dim).item()
        t_est_elaps = time.perf_counter() - t_est_strt
        exp_results['time_train'] = t_tr_elaps
        exp_results['time_est'] = t_est_elaps
        file_manager.save(exp_results, exp_number)
    

def conv_comp_em_mix_gauss_targ_experiment(config):
    dim = config['dim']
    exp_numbers = config['exp_numbers']
    em_dt = config['dt']
    n_particles = config['n_particles']
    n_eval_samples = config['n_eval_samples']
    if n_eval_samples == -1:
        n_eval_samples = n_particles
    verbose = config['verbose']
    device = 'cpu'
    exp_name = config['experiment_name']
    file_manager = config['file_manager'] if 'file_manager' in config else ConvergenceComparison_EFM.fromconfig(config)
    
    for exp_number in exp_numbers:
        if verbose:
            print(f'[EM convergence № {exp_number}] starts: dim: {dim}')
        # seed all random sources
        r_m = get_random_manager(config['random_key'], dim, exp_number)
        r_m.seed()
        # creates stationary distribution
        stationary_distrib = create_stationary_distrib(config, device)
        
        # create init sampler
        init_distrib = TD.MultivariateNormal(
            torch.zeros(dim).to(device), 
            config['init_variance'] * torch.eye(dim).to(device))
        
        t_tr_strt = time.perf_counter()
        # sample particles from the initial distribution
        x0 = init_distrib.sample((n_particles,))
        
        final_x = torchBatchItoEulerDistrib(stationary_distrib, x0, em_dt, config['t_fin']).cpu().numpy()
        if verbose:
            print(f'[EM convergence № {exp_number}] particles have been simulated')
        final_distrib = gaussian_kde(final_x.transpose())
        if verbose:
            print(f'[EM convergence № {exp_number}] kde has been built')
        t_tr_elaps = time.perf_counter() - t_tr_strt
        
        exp_results = {}
        t_ev_strt = time.perf_counter()
        # KL wrt train est.
        em_sample = final_distrib.resample(n_eval_samples).T
        exp_results['kl_train'] = KL_train_distrib(
            em_sample, final_distrib, 
            stationary_distrib, device=device)
        # KL wrt target est.
        curr_kl_target, X_targ = KL_targ_distrib(
            n_eval_samples, final_distrib, 
            stationary_distrib, ret_targ_sample=True)
        exp_results['kl_target'] = curr_kl_target
        # Energy-based dist. est.
        exp_results['energy_based'] = energy_based_distance(
                final_x[:n_eval_samples, :], X_targ, dim=dim)
        t_ev_elaps = time.perf_counter() - t_ev_strt
        
        exp_results['time_train'] = t_tr_elaps
        exp_results['time_est'] = t_ev_elaps
        file_manager.save(exp_results, exp_number)

def conv_comp_em_proxrec_mix_gauss_targ_experiment(config):
    beta=1.
    dim = config['dim']
    exp_numbers = config['exp_numbers']
    em_dt = config['dt']
    n_particles = config['n_particles']
    n_eval_samples = config['n_eval_samples']
    if n_eval_samples == -1:
        n_eval_samples = n_particles
    verbose = config['verbose']
    exp_name = config['experiment_name']
    file_manager = config['file_manager'] if 'file_manager' in config else ConvergenceComparison_EFM.fromconfig(config)
    dtype = config['dtype']
    device = config['device']
    dtype = torch.float32 if dtype == 'float32' else torch.float64

    for exp_number in exp_numbers:
        if verbose:
            print(f'[EM PR convergence № {exp_number}] starts: dim: {dim}')
        # seed all random sources
        r_m = get_random_manager(config['random_key'], dim, exp_number)
        r_m.seed()
        # creates stationary distribution
        stationary_distrib = create_stationary_distrib(config, device, dtype=dtype)
        
        # create init sampler
        init_distrib = TD.MultivariateNormal(
            torch.zeros(dim, dtype=dtype).to(device), 
            config['init_variance'] * torch.eye(dim, dtype=dtype).to(device))
        
        t_tr_strt = time.perf_counter()

        # sample particles from the initial distribution
        prev_em_samples = init_distrib.sample((n_particles,))
        prev_pdf = torch.exp(init_distrib.log_prob(prev_em_samples))

        def stat_pot_func(x):
            return - stationary_distrib.log_prob(x)/beta
        
        fin_em_samples, fin_pdf = torchBatchItoEulerProxrec(
            stat_pot_func, prev_em_samples, prev_pdf, em_dt, config['t_fin'], 
            reg=config['proxrec_reg'], tol=config['proxrec_tol'], 
            maxiter=config['proxrec_maxiter'], beta=beta, verbose=verbose)

        if verbose:
            print(f'[EM PR convergence № {exp_number}] particles along with pdf have been simulated')
        
        # normalize distribution
        norm_constant = normalize_pdf_kde(fin_em_samples, fin_pdf)
        fin_pdf = fin_pdf / norm_constant

        if verbose:
            print(f'[EM PR convergence № {exp_number}] pdf has been normalized')
        t_tr_elaps = time.perf_counter() - t_tr_strt
        
        exp_results = {}
        t_ev_strt = time.perf_counter()
        # KL wrt train est.
        exp_results['kl_train'] = em_proxrec_KL_train(
            fin_em_samples[:n_eval_samples, :], fin_pdf[:n_eval_samples], stationary_distrib)
        # KL wrt target est.
        exp_results['kl_target'] = em_proxrec_KL_targ(
            fin_em_samples[:n_eval_samples, :], fin_pdf[:n_eval_samples], stationary_distrib)
        t_ev_elaps = time.perf_counter() - t_ev_strt
        
        exp_results['time_train'] = t_tr_elaps
        exp_results['time_est'] = t_ev_elaps
        file_manager.save(exp_results, exp_number)

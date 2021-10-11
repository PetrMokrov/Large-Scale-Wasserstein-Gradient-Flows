import torch
import numpy as np
import torch.distributions as TD
import time

from ..icnn import DenseICNN
from ..diffusion import Diffusion
from ..filtering import DiffusionFilteringMCMC, model_filtering_cc, model_observations
from ..filtering import KL_filtering_targ_distrib, KL_filtering_train_distrib
from ..filtering import model_filtering_frogner
from ..filtering import model_filtering_bbf
from ..utils import id_pretrain_model, train_diffusion_model
from ..changcooper import CCTarget
from .manual_random import get_random_manager
from .exp_file_manager import Filtering_EFM
from scipy.stats import gaussian_kde

class FWTarget(CCTarget):

    def __init__(self):
        super().__init__()
        pass
    
    def potential(self, x):
        return np.sin(2. * np.pi * x)/np.pi + (x**2) / 4.
    
    def grad_potential(self, x):
        return 2. * np.cos(2. * np.pi * x) + x / 2.
    
    def log_prob(self, x):
        return - (torch.sin(2. * np.pi * x)/np.pi + (x**2)/ 4.)

def filtering_preprocess(exp_number, config, target, device='cpu', squeezed=False):
    verbose = config['verbose']
    # observation timesteps creation
    t_observations = np.linspace(0., config['t_fin'], config['n_observations'] + 1, endpoint=False)[1:]
    # create initial process distribution
    if not squeezed:
        init_distrib = TD.Normal(
            torch.tensor([0.]).to(device), 
            torch.tensor([config['init_std']]).to(device))
        # create noise observation distribution
        noise_distrib = TD.Normal(
            torch.tensor([0.]).to(device), 
            torch.tensor([config['noise_std']]).to(device))
    else:
        init_distrib = TD.Normal(
            torch.tensor(0.).to(device), 
            torch.tensor(config['init_std']).to(device))
        # create noise observation distribution
        noise_distrib = TD.Normal(
            torch.tensor(0.).to(device), 
            torch.tensor(config['noise_std']).to(device))
    # seed all random sources
    r_m = get_random_manager(config['random_key'], exp_number)
    r_m.seed()
    init_distrib_cpu = TD.Normal(0., config['init_std'])
    noise_distrib_cpu = TD.Normal(0., config['noise_std'])
    
    # model observations 
    if verbose:
        print(f'[№ {exp_number}] Obtaining model observations')
    noise_sampled = model_observations(
        init_distrib_cpu, target, t_observations, config['em_observations_dt'], noise_distrib_cpu)
    
    # Chang&Cooper ground-truth modelling
    min_cc_grid = -5
    max_cc_grid = 5
    cc_n_grid = int((max_cc_grid - min_cc_grid)/config['cc_dx'])
    
    if verbose:
        print(
            f'[№ {exp_number}] Launching Chang-Cooper,', 
            f'on [{min_cc_grid}, {max_cc_grid}], n_grid={cc_n_grid}')
    
    xs = torch.tensor(
        np.linspace(min_cc_grid, max_cc_grid, cc_n_grid, endpoint=True), 
        dtype=torch.float32, device=device)

    cc_px = model_filtering_cc(
        init_distrib, target, xs, config['cc_dt'], 
        config['t_fin'], t_observations, noise_sampled, noise_distrib)
    return t_observations, init_distrib, noise_distrib, noise_sampled, xs, cc_px

def filtering_icnn_jko_experiment(config):
    device = config['device']
    verbose = config['verbose']
    jko_dt = config['dt']
    target = FWTarget()
    batch_size=config['batch_size']
    file_manager = config['file_manager'] if 'file_manager' in config else Filtering_EFM.fromconfig(config)
    
    for exp_number in config['exp_numbers']:
        if verbose:
            print(f'[№ {exp_number}] Start filtering ICNN jko experiment')
        
        t_observations, init_distrib, noise_distrib, noise_sampled, xs, cc_px = filtering_preprocess(
            exp_number, config, target, device=device)
        
        # nonlin.filtering via ICNN JKO
        
        layer_width = config['ICNN_width']
        model_args = [1, [layer_width,]*config['n_ICNN_layers']]
        model_kwargs = {
            'rank':5, 
            'activation':'softplus', 
            'batch_size':batch_size}
        df_mcmc = DiffusionFilteringMCMC(init_distrib, method=config['mcmc_method'])
        
        def train_diff_once(n_iterate):
            D0 = DenseICNN(*model_args, **model_kwargs).to(device)
            # initialize the model
            for p in D0.parameters():
                p.data = torch.randn(
                    p.shape, device=device, 
                    dtype=torch.float32) / np.sqrt(float(layer_width))
            
            # pretrain the model to be identity function
            D0 = id_pretrain_model(
                D0, df_mcmc, batch_size=batch_size, 
                lr=config['pretrain_learning_rate'], verbose=verbose)
            diff = Diffusion(init_distrib)
            X_test = df_mcmc.sample_n(batch_size)
            
            # train the current block of ICNNs
            diff, _ = train_diffusion_model(
                diff, D0, (model_args, model_kwargs), target, n_iterate,
                init_sampler=df_mcmc,
                batch_size=batch_size, 
                lr = config['learning_rate'],
                step_size=config['dt'], 
                step_iterations=config['n_step_iterations'], 
                n_max_prop=None, 
                X_test=X_test,
                device=device, plot_loss=False, verbose=verbose)
            return diff
        
        if verbose:
            print(f'[№ {exp_number}] ICNN JKO: Launching')
        
        t_exp_strt = time.perf_counter()
        t_start = 0.
        for i, t_curr in enumerate(t_observations):
            if verbose:
                print(f'[№ {exp_number}] ICNN JKO: block {i}, t_curr: {t_curr} starting')
            t_diff = t_curr - t_start
            y_impl = noise_sampled[i][-1]
            n_iterate = int(t_diff / config['dt'])
            assert n_iterate * config['dt'] == t_diff
            if verbose:
                print(
                    f'[№ {exp_number}] ICNN JKO: Step launch info:', 
                    f' t_diff={t_diff}, y_impl={y_impl}, n_iterate={n_iterate}')
            diff = train_diff_once(n_iterate)
            df_mcmc.add_terminated_diff(
                diff, y_impl, noise_distrib, config['n_warm_up'])
            df_mcmc.n_decorrelate = config['n_decorrelate']
            t_start = t_curr

        if verbose:
            print(f'[№ {exp_number}] ICNN JKO: final block')
        t_diff = config['t_fin'] - t_curr
        n_iterate = int(t_diff / config['dt'])
        assert n_iterate * config['dt'] == t_diff
        last_diff = train_diff_once(n_iterate)
        t_exp_dur = time.perf_counter() - t_exp_strt
        
        t_est_strt = time.perf_counter()
        kl_train, pred_px = KL_filtering_train_distrib(xs, cc_px, df_mcmc, last_diff, ret_pred_px=True)
        kl_targ = KL_filtering_targ_distrib(xs, cc_px, df_mcmc, last_diff)
        t_est_dur = time.perf_counter() - t_est_strt

        res_dict = {
            'kl_train': kl_train,
            'kl_targ' : kl_targ,
            'time_train': t_exp_dur,
            'time_est': t_est_dur
        }
        file_manager.save(res_dict, exp_number)
        # save final distributions image
        np_xs = xs.cpu().numpy()
        np_res_pred = np.stack([np_xs, pred_px])
        np_res_cc = np.stack([np_xs, cc_px])
        file_manager.save_np(np_res_pred, exp_number, 'ICNN_jko')
        file_manager.save_np(np_res_cc, exp_number, 'ChangCooper')
        
def filtering_dual_jko_experiment(config):
    verbose = config['verbose']
    jko_dt = config['dt']
    target = FWTarget()
    file_manager = config['file_manager'] if 'file_manager' in config else Filtering_EFM.fromconfig(config)

    for exp_number in config['exp_numbers']:
        if verbose:
            print(f'[№ {exp_number}] Start filtering dual JKO experiment')
        
        t_obs, init_distrib, noise_distrib, noise_sampled, xs, cc_px = filtering_preprocess(
            exp_number, config, target, device='cpu', squeezed=True)
        np_xs = xs.cpu().numpy()

        if verbose:
            print(f'[№ {exp_number}] dual JKO: Launching')
        
        t_exp_strt = time.perf_counter()
        res_frogner = model_filtering_frogner(
            init_distrib, target, jko_dt, config['t_fin'], t_obs, noise_sampled, noise_distrib, verbose=verbose)
        t_exp_dur = time.perf_counter() - t_exp_strt
        
        if verbose:
            print(f'[№ {exp_number}] dual JKO: estimation')
        t_est_strt = time.perf_counter()
        kl_train = KL_filtering_train_distrib(np_xs, cc_px, res_frogner)
        kl_target = KL_filtering_targ_distrib(np_xs, cc_px, res_frogner)
        t_est_dur = time.perf_counter() - t_est_strt
        # saving the resutls
        res_dict = {
            'kl_train': kl_train,
            'kl_targ' : kl_target,
            'time_train': t_exp_dur,
            'time_est': t_est_dur
        }
        file_manager.save(res_dict, exp_number)
        # saving final distribution image
        pred_px = res_frogner(np_xs)
        np_res_pred = np.stack([np_xs, pred_px])
        # np_res_cc = np.stack([np_xs, cc_px])

        file_manager.save_np(np_res_pred, exp_number, 'dual_jko')
        # file_manager.save_np(np_res_cc, exp_number, 'ChangCooper_dual') # just to test

def filtering_bbf_experiment(config):
    verbose = config['verbose']
    target = FWTarget()
    file_manager = config['file_manager'] if 'file_manager' in config else Filtering_EFM.fromconfig(config)

    for exp_number in config['exp_numbers']:
        if verbose:
            print(f'[№ {exp_number}] Start filtering Bayesiam Bootstrap filter experiment')
        
        t_obs, init_distrib, noise_distrib, noise_sampled, xs, cc_px = filtering_preprocess(
            exp_number, config, target, device='cpu', squeezed=True)
        np_xs = xs.cpu().numpy()
        t_obs = t_obs.tolist()

        if verbose:
            print(f'[№ {exp_number}] BBF: Launching')
        
        t_exp_strt = time.perf_counter()
        res_bbf = model_filtering_bbf(
            init_distrib, target, config['n_particles'], config['dt'], 
            config['t_fin'], t_obs, noise_sampled, noise_distrib)
        kde_distrib_bbf = gaussian_kde(res_bbf.transpose())
        t_exp_dur = time.perf_counter() - t_exp_strt
        
        if verbose:
            print(f'[№ {exp_number}] BBF: estimation')
        t_est_strt = time.perf_counter()
        kl_train = KL_filtering_train_distrib(np_xs, cc_px, kde_distrib_bbf)
        kl_target = KL_filtering_targ_distrib(np_xs, cc_px, kde_distrib_bbf)
        t_est_dur = time.perf_counter() - t_est_strt
        # saving the resutls
        res_dict = {
            'kl_train': kl_train,
            'kl_targ' : kl_target,
            'time_train': t_exp_dur,
            'time_est': t_est_dur
        }
        file_manager.save(res_dict, exp_number)
        # saving final distribution image
        pred_px = kde_distrib_bbf(xs)
        np_res_pred = np.stack([np_xs, pred_px])
        # np_res_cc = np.stack([np_xs, cc_px])

        file_manager.save_np(np_res_pred, exp_number, config['experiment_method'])
        # file_manager.save_np(np_res_cc, exp_number, f'ChangCooper_{config["experiment_method"]}') # just to test

        
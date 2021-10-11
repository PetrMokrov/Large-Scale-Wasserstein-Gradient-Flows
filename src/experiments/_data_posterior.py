import torch
import numpy as np
import torch.distributions as TD
from torch.utils.data import DataLoader
from ..utils import id_pretrain_model, train_diffusion_model
from .manual_random import get_random_manager
from ..icnn import DenseICNN
from ..diffusion import Diffusion
import time
from ..data_posterior import LogRegDPTarget, posterior_sample_evaluation
from ..datasets import get_train_test_datasets, dataset2numpy
from .exp_file_manager import DataPosterior_EFM

def data_posterior_experiment(config):
    device = config['device']
    verbose = config['verbose']
    file_manager = config['file_manager'] if 'file_manager' in config else DataPosterior_EFM.fromconfig(config)
    if verbose:
        print(f"[Data Posterior {config['ds_name']}]: starts")
    # split datasets
    dataset, train_ds, test_ds = get_train_test_datasets(config['ds_name'])
    train_dl = DataLoader(train_ds, batch_size=config['data_batch_size'], shuffle=True)
    target = LogRegDPTarget(
        train_dl, dataset.n_features, device=device, clip_alpha=config['clip_alpha'])
    if verbose:
        print(f"[Data Posterior {config['ds_name']}]: dataset loaded")
    
    # seed random sources
    r_m = get_random_manager(config['random_key'])
    r_m.seed()

    # create ICNN base model
    n_features = dataset.n_features
    dim = n_features + 1
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
    
    # preinitialize the model with standard multivariate normal
    #TODO: consider variance to be parameter
    pre_init_distrib = TD.MultivariateNormal(
        torch.zeros(dim).to(device),  100 * torch.eye(dim).to(device))
    
    D0 = id_pretrain_model(
        D0, pre_init_distrib, lr=config['pretrain_lr'], 
        n_max_iterations=4000, batch_size=batch_size, verbose=verbose)
    
    # pretrain the model to be identity on the 
    # distribution of interest
    init_sampler = target.create_init_sampler()
    D0 = id_pretrain_model(
        D0, init_sampler, lr=config['pretrain_lr'], 
        n_max_iterations=4000, batch_size=batch_size, verbose=verbose)

    if verbose:
        print(f"[Data Posterior {config['ds_name']}]: base ICNN pretrained")
    
    # launch the diffusion
    t_tr_strt = time.perf_counter()
    diff = Diffusion(init_sampler)
    for i_iter in range(config['n_jko_steps']):
        diff = train_diffusion_model(
            diff, D0, (model_args, model_kwargs), target, 1, 
            batch_size=batch_size, lr=config['lr'], step_size=config['dt'],
            step_iterations=config['n_step_iterations'], n_max_prop=config['n_max_prop'], 
            device=device, plot_loss=False, 
            ret_loss_history=False, verbose=verbose, target_type='data_posterior')
    t_tr_elaps = time.perf_counter() - t_tr_strt

    if verbose:
        print(f"[Data Posterior {config['ds_name']}]: ICNNs have trained")

    # evaluation
    X_test, y_test = dataset2numpy(test_ds)
    t_est_strt = time.perf_counter()
    theta = diff.sample(config['n_eval_samples']).cpu().numpy()
    acc, llh = posterior_sample_evaluation(theta, X_test, y_test)
    t_est_elaps = time.perf_counter() - t_est_strt

    exp_results = {}
    exp_results['accuracy'] = acc
    exp_results['log_lik'] = llh
    exp_results['time_train'] = t_tr_elaps
    exp_results['time_est'] = t_est_elaps
    file_manager.save(exp_results)

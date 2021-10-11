from .diffusion import Diffusion
from multipledispatch import dispatch

import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from IPython.display import clear_output
import torch.nn.functional as F
from scipy.stats import gaussian_kde
from .changcooper import get_dx
import torch.distributions as TD
import scipy.stats as sps
from scipy.spatial.distance import cdist as np_cdist

class DataLoaderWrapper:
    '''
    Helpful class for using the 
    DistributionSampler's in torch's 
    DataLoader manner
    '''

    class FiniteRepeatDSIterator:

        def __init__(self, sampler, batch_size, n_batches):
            dataset = sampler.sample(batch_size * n_batches)
            assert(len(dataset.shape) >= 2)
            new_size = (n_batches, batch_size) + dataset.shape[1:]
            self.dataset = dataset.view(new_size)
            self.batch_size = batch_size
            self.n_batches = n_batches
        
        def __iter__(self):
            for i in range(self.n_batches):
                yield self.dataset[i]
    
    class FiniteUpdDSIterator:

        def __init__(self, sampler, batch_size, n_batches):
            self.sampler = sampler
            self.batch_size = batch_size
            self.n_batches = n_batches
        
        def __iter__(self):
            for i in range(self.n_batches):
                yield self.sampler.sample(self.batch_size)
            
    class InfiniteDsIterator:

        def __init__(self, sampler, batch_size):
            self.sampler = sampler
            self.batch_size = batch_size
        
        def __iter__(self):
            return self
        
        def __next__(self):
            return self.sampler.sample(self.batch_size)


    def __init__(self, sampler, batch_size, n_batches=None, store_dataset=False):
        '''
        n_batches : count of batches before stop_iterations, if None, the dataset is infinite
        store_datset : if n_batches is not None and store_dataset is True, 
        during the first passage through the dataset the data will be stored,
        and all other epochs will use the same dataset, stored during the first pass
        '''
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.store_dataset = store_dataset
        self.sampler = sampler

        if self.n_batches is None:
            self.ds_iter = DataLoaderWrapper.InfiniteDsIterator(
                sampler, self.batch_size)
            return
        
        if not self.store_dataset:
            self.ds_iter = DataLoaderWrapper.FiniteUpdDSIterator(
                sampler, self.batch_size, self.n_batches)
            return
        
        self.ds_iter = DataLoaderWrapper.FiniteRepeatDSIterator(
            sampler, self.batch_size, self.n_batches)

    
    def __iter__(self):
        return iter(self.ds_iter)

class TransformableDataset:

    def __init__(self, sampler, n_sample):
        self.sampler = sampler
        self.n_sample = n_sample
        self.dataset = self.sampler.sample(n_sample)
        self.dataset.requires_grad_(True)
        assert(len(self.dataset.shape) >= 2)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i]
    
    def transform(self, *args, **kwargs):
        pass
    
class GFTransformableDataset(TransformableDataset):

    def __init__(self, sampler, n_sample):
        super().__init__(sampler, n_sample)
    
    def transform(self, D):
        batch_size = 1024
        self.dataset.requires_grad_(True)
        self.dataset = D.push_nograd(self.dataset)

class StatsManager:

    @staticmethod
    def traverse(o, tree_types=(list, tuple, np.ndarray)):
        if isinstance(o, tree_types):
            for value in o:
                for subvalue in StatsManager.traverse(value, tree_types):
                    yield subvalue
        else:
            yield o

    def __init__(self, *names):
        self.stats = {}
        self.names = list(names)
        for name in names:
            self.stats[name] = []
    
    def add_all(self, *vals):
        if len(vals) == 1:
            for name in self.names:
                self.add(name, vals[0])
            return
        if len(vals) == len(self.names):
            for name, val in zip(self.names, vals):
                self.add(name, val)
            return
        raise Exception('stats update is ambiguous')

    def add(self, name, val):
        self.stats[name][-1] += val
    
    def upd_all(self, *vals):
        if len(vals) == 1:
            for name in self.names:
                self.upd(name, vals[0])
            return
        if len(vals) == len(self.names):
            for name, val in zip(self.names, vals):
                self.upd(name, val)
            return 
        raise Exception('stats update is ambiguous')
    
    def upd(self, name, val):
        self.stats[name].append(val)
    
    def get(self, name):
        return self.stats[name]
    
    def draw(self, axs):
        axs_list = list(self.traverse(axs))
        for i, name in enumerate(self.names):
            axs_list[i].plot(self.get(name))
            axs_list[i].set_title(name)

@dispatch(torch.Tensor, Diffusion, object)
def KL_train_distrib(X, diff, targ, ret_diff_sample=False):
    entropy, X = diff.mc_entropy(X, return_X_transformed=True)
    kl = (entropy - targ.log_prob(X).mean()).detach()
    if ret_diff_sample:
        return kl, X
    return kl

@dispatch(int, Diffusion, object)
def KL_train_distrib(n, diff, targ, ret_diff_sample=False):
    X = diff.sample_init(n)
    return KL_train_distrib(X, diff, targ, ret_diff_sample=ret_diff_sample)

@dispatch(torch.Tensor, Diffusion, object)
def KL_targ_distrib(X, diff, targ, **kwargs):
    return (targ.log_prob(X).mean() - diff.log_prob(X, **kwargs).mean()).detach()

@dispatch(int, Diffusion, object)
def KL_targ_distrib(n, diff, targ, ret_targ_sample=False, **kwargs):
    X = targ.sample((n,))
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    if ret_targ_sample:
        return KL_targ_distrib(X, diff, targ, **kwargs), X
    return KL_targ_distrib(X, diff, targ, **kwargs)

@dispatch(int, gaussian_kde, object)
def KL_targ_distrib(n, em_kde_distrib, targ, try_true_entropy=False, ret_targ_sample=False):
    kl = 0.
    X = targ.sample((n,))
    if try_true_entropy:
        try:
            kl -= targ.entropy().item()
        except:
            kl += targ.log_prob(X).mean().item()
    else:
        kl += targ.log_prob(X).mean().item()
    np_X = X.cpu().numpy()
    kl -= np.log(em_kde_distrib(np_X.transpose())).mean()
    if ret_targ_sample:
        return kl, np_X
    return kl

@dispatch(np.ndarray, gaussian_kde, object)
def KL_train_distrib(em_sample, em_kde_distrib, targ, device='cpu', dtype=torch.float32):
    kl = np.log(em_kde_distrib(em_sample.transpose())).mean()
    trc_em_sample = torch.tensor(em_sample, device=device, dtype=dtype)
    kl -= targ.log_prob(trc_em_sample).mean().item()
    return kl


def id_pretrain_model(
    model, sampler, lr=1e-3, n_max_iterations=2000, batch_size=1024, loss_stop=1e-5, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    for it in tqdm(range(n_max_iterations), disable = not verbose):
        X = sampler.sample_n(batch_size)
        if len(X.shape) == 1:
            X = X.view(-1, 1)
        X.requires_grad_(True)
        loss = F.mse_loss(model.push(X), X)
        loss.backward()
        
        opt.step()
        opt.zero_grad() 
        model.convexify()
        
        if verbose:
            if it % 100 == 99:
                clear_output(wait=True)
                print('Loss:', loss.item())
            
            if loss.item() < loss_stop:
                clear_output(wait=True)
                print('Final loss:', loss.item())
                break
    return model

def train_diffusion_model(
    diffusion_model, init_model, model_params, target, n_steps,
    init_sampler=None,
    batch_size=1024, 
    lr = 5.e-3,  
    step_size=0.1, 
    step_iterations=300, 
    n_max_prop=None, 
    X_test=None,
    device='cpu', plot_loss=True, 
    ret_loss_history=False, verbose=True, target_type='distribution'):
    assert target_type in ['distribution', 'data_posterior'], f"target_type='{target_type}' is not known"
    kl_train = []
    model_args = model_params[0]
    model_kwargs = model_params[1]
    model_kwargs['batch_size'] = batch_size
    if ret_loss_history:
        cum_loss_history = []
    if init_model is not None:
        model_class = init_model.__class__
    else:
        model_class = diffusion_model.Ds[-1].__class__
    if n_max_prop is None:
        n_max_prop = batch_size

    for i in range(n_steps):
        D = model_class(*model_args, **model_kwargs).to(device)
        if init_model is None:
            D.load_state_dict(diffusion_model.Ds[-1].state_dict())
        else:
            D.load_state_dict(init_model.state_dict() if i == 0 else diffusion_model.Ds[-1].state_dict())
        loss_history = []
        opt = torch.optim.Adam(D.parameters(), lr=lr, weight_decay=1e-9)

        def compute_loss_distribution(X, factor):
            X.requires_grad_(True)
            push_X = D.push(X)
            neg_loss = factor*diffusion_model.negative_entropy_gain(push_X, X).sum()
            w_loss = factor * 0.5 *  ((push_X - X)**2).sum(dim=1).sum()
            targ_loss = factor * target.log_prob(push_X).sum()
            loss = w_loss + step_size * (neg_loss - targ_loss)
            return loss
        
        def compute_loss_data_posterior(X, factor):
            X.requires_grad_(True)
            push_X = D.push(X)
            neg_loss = factor*diffusion_model.negative_entropy_gain(push_X, X).sum()
            w_loss = factor * 0.5 *  ((push_X - X)**2).sum(dim=1).sum()
            S = target.sample_data()
            init_targ_loss = factor * target.est_log_init_prob(push_X, reduction='sum')
            data_targ_loss = target.len_dataset * factor * target.est_log_data_prob(push_X, S, reduction='sum')
            loss = w_loss + step_size * (neg_loss - init_targ_loss - data_targ_loss)
            return loss
        
        losses_map = {
            'distribution': compute_loss_distribution,
            'data_posterior': compute_loss_data_posterior
        }
        compute_loss = losses_map[target_type]
        
        for it in tqdm(range(step_iterations), disable = not verbose):
            opt.zero_grad() 
            if init_sampler is None:
                X = diffusion_model.sample(batch_size)
            else:
                X_0 = init_sampler.sample(batch_size)
                X = diffusion_model.propagate(X_0)

            loss_history.append(0.0)
            for i_prop in range(0, X.size(0), n_max_prop):
                X_prop = X[i_prop:i_prop + n_max_prop]
                c_loss = compute_loss(X_prop, 1./X.size(0))
                c_loss.backward()
                loss_history[-1] += c_loss.item()
            
            opt.step()
            D.convexify()
            opt.zero_grad() 
        diffusion_model.Ds.append(D)
        if X_test is not None:
            kl_train.append(KL_train_distrib(X_test, diffusion_model, target).item())
        if ret_loss_history:
            cum_loss_history.append(loss_history)
        
        if verbose:
            clear_output(wait=True)
            print('Step {} summary'.format(i))
            if X_test is not None:
                print('KL: ', kl_train[-1])
            if plot_loss:
                plt.plot(loss_history)
                plt.title('Main loss')
                plt.show()
    if X_test is not None:
        if ret_loss_history:
            return diffusion_model, kl_train, cum_loss_history
        return diffusion_model, kl_train
    if ret_loss_history:
        return diffusion_model, cum_loss_history
    return diffusion_model

@dispatch((list, np.ndarray), TD.Distribution)
def discretize_distrib(xs, distrib, dtype=torch.float32, np_dtype=np.float64, device='cpu'):
    trc_xs = torch.tensor(xs, device=device, dtype=dtype)
    return discretize_distrib(trc_xs, distrib, np_dtype=np_dtype)

@dispatch(torch.Tensor, TD.Distribution)
def discretize_distrib(xs, distrib, np_dtype=np.float64):
    xs_prob = np.exp(distrib.log_prob(xs).cpu().numpy()).astype(np_dtype)
    return xs_prob 

def normalize_grid(vals, xs, ret_int=False):
    '''
    Normalizes the function, defined on the grid
    :Parameters: 
    vals : iterable : function values on the grid
    xs : iterable : ascending equidistant grid
    '''
    dx = get_dx(xs)
    if isinstance(vals, torch.Tensor):
        _int = torch.sum(vals)*dx
    else:
        _int = np.sum(vals)*dx
    if ret_int:
        return _int
    return vals/_int

def energy_based_distance(X, Y, dim=None):
    assert isinstance(X, (torch.Tensor, np.ndarray))
    assert isinstance(Y, (torch.Tensor, np.ndarray))
    assert len(X.shape) == 2
    assert len(Y.shape) == 2
    assert X.shape[1] == Y.shape[1]
    if dim is not None:
        assert X.shape[1] == dim
    if isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):
        A = torch.mean(torch.cdist(X, Y, p = 2))
        B = torch.mean(torch.cdist(X, X, p = 2))
        C = torch.mean(torch.cdist(Y, Y, p = 2))
        return 2. * A - B - C
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    A = np.mean(np_cdist(X, Y, p=2))
    B = np.mean(np_cdist(X, X, p=2))
    C = np.mean(np_cdist(Y, Y, p=2))
    return 2. * A - B - C
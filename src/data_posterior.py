import torch
import torch.distributions as TD
import numpy as np
import numpy.matlib as nm

class DataPosteriorTarget:
    
    def __init__(self):
        pass
    
    def sample_init(self, n):
        '''Samples from the prior distribution
        '''
        raise NotImplementedError()
    
    def sample_data(self):
        '''Draws batch sample from the dataset
        '''
        raise NotImplementedError()
    
    @property
    def len_dataset(self):
        raise NotImplementedError()
    
    def est_log_init_prob(self, X):
        '''Estimates \int \log(p_{prior}(x)) p_{curr}(x) dx 
        based on batch sample from p_{curr}
        '''
        raise NotImplementedError()
    
    def est_log_data_prob(self, X, S):
        '''Estimates (1/|Data|)\sum_{s \in Data} \int \log(p(s | x)) p_{curr}(x) dx
        based on batch sample X from p_{curr}
        and data sample S from Data (drawn uniformly from the dataset)
        '''
        raise NotImplementedError()
    
class LogRegDPTarget(DataPosteriorTarget):

    class _InitSampler:
    
        def __init__(self, data_posterior_target):
            self.dpt = data_posterior_target
        
        def sample_n(self, n):
            return self.dpt.sample_init(n)
        
        def sample(self, n):
            return self.sample_n(n)
    
    def __init__(
        self, dataloader, n_features, 
        device='cpu', g_alpha=1., g_beta=100.0, clip_alpha=None):
        super().__init__()
        self.device = device
        self.a0, self.b0 = g_alpha, g_beta # alpha, beta (size, 1/scale)
        self.gamma0 = TD.Gamma(
            torch.tensor(self.a0, dtype=torch.float32, device=device), 
            torch.tensor(self.b0, dtype=torch.float32, device=device))
        self.normal0 = TD.Normal(
            torch.tensor(0., dtype=torch.float32, device=device), 
            torch.tensor(1., dtype=torch.float32, device=device))
        self.n_features = n_features # num features in the dataset
        self.dataloader = dataloader
        self._dataloader_iter = iter(self.dataloader)
        self.n_data_samples_drawn = 0
        self.n_data_epochs_drawn = 0
        self.clip_alpha = clip_alpha
    
    def reset(self):
        self._dataloader_iter = iter(self.dataloader)
        self.n_data_samples_drawn = 0
        self.n_data_epochs_drawn = 0
    
    def sample_init(self, n):
        alpha_sample = self.gamma0.sample((n,)).view(-1, 1)
        if self.clip_alpha is not None:
            alpha_sample = torch.clamp(alpha_sample, np.exp(-self.clip_alpha), np.exp(self.clip_alpha))
        theta_sample = self.normal0.sample((n, self.n_features)) / torch.sqrt(alpha_sample)
        return torch.cat([theta_sample, torch.log(alpha_sample)], dim=-1)
    
    def sample_data(self):
        try:
            data, classes = next(self._dataloader_iter)
            assert data.size(1) == self.n_features
            self.n_data_samples_drawn += 1
        except StopIteration:
            self._dataloader_iter = iter(self.dataloader)
            self.n_data_epochs_drawn += 1
            data, classes = next(self._dataloader_iter)
        batch = torch.cat([
            classes.view(-1, 1).type(torch.float32), 
            data.type(torch.float32)], dim=-1).to(self.device)
        return batch
    
    @property
    def len_dataset(self):
        return len(self.dataloader.dataset)
    
    def est_log_init_prob(self, X, reduction='mean'):
        assert len(X.shape) == 2
        assert X.size(1) == self.n_features + 1 # featrues + alpha
        log_alpha_sample = X[:, -1].view(-1, 1)
        if self.clip_alpha is not None:
            log_alpha_sample = torch.clamp(log_alpha_sample, -self.clip_alpha, self.clip_alpha)
        alpha_sample = torch.exp(log_alpha_sample)
        theta_sample = X[:, :-1]
        log_p_w_cond_alp = torch.sum(
            self.normal0.log_prob(theta_sample * torch.sqrt(alpha_sample)) + log_alpha_sample/2., dim=-1)
        log_p_alp = self.gamma0.log_prob(alpha_sample) + log_alpha_sample
        log_p = log_p_alp.view(-1) + log_p_w_cond_alp
        if reduction=='mean':
            return torch.mean(log_p)
        if reduction=='sum':
            return torch.sum(log_p)
        raise Exception(f"Reduction '{reduction}' not defined")
    
    def est_log_data_prob(self, X, S, reduction='mean'):
        # S[0] is class label -1 or 1
        assert X.size(1) == S.size(1)
        assert X.size(1) == self.n_features + 1
        probas = torch.sigmoid(torch.matmul(X[:, :-1], S[:, 1:].T)) # (x_bs, s_bs)
        classes = S[:, 0].view(1, -1)
        probas = (1. - classes)/2. + classes * probas
        probas = torch.clamp(probas, 1e-5)
        log_probas = torch.log(probas)
        mean_log_probas = torch.mean(log_probas, dim=-1)
        assert mean_log_probas.size(0) == X.size(0)
        if reduction == 'mean':
            return torch.mean(mean_log_probas)
        if reduction == 'sum':
            return torch.sum(mean_log_probas)
        raise Exception(f"Reduction '{reduction}' not defined")
    
    def create_init_sampler(self):
        return self._InitSampler(self)

def posterior_sample_evaluation(theta, X_test, y_test):
    theta = theta[:, :-1]
    M, n_test = theta.shape[0], len(y_test)

    prob = np.zeros([n_test, M])
    for t in range(M):
        coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
        prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))

    prob = np.mean(prob, axis=1)
    acc = np.mean(prob > 0.5)
    llh = np.mean(np.log(prob))
    return acc, llh

import torch
import numpy as np
from scipy.linalg import sqrtm
import sklearn.datasets
import random

def symmetrize(X):
    return np.real((X + X.T) / 2)

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
        self.mean, self.var, self.cov = None, None, None
    
    def sample(self, size=5):
        pass
    
    def _estimate_moments(self, size=2**15, mean=True, var=True, cov=True):
        if (not mean) and (not var) and (not cov):
            return
        
        sample = self.sample(size).cpu().detach().numpy().astype(np.float32)
        if mean:
            self.mean = sample.mean(axis=0)
        if var:
            self.var = sample.var(axis=0).sum()
        if cov:
            self.cov = np.cov(sample.T).astype(np.float32)
    
class StandardNormalSampler(Sampler):
    def __init__(
        self, dim=2, device='cuda'
    ):
        super(StandardNormalSampler, self).__init__(device)
        self.dim = dim
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.cov = np.eye(self.dim, dtype=np.float32)
        self.var = self.dim
        
    def sample(self, size=10):
        return torch.randn(
            size, self.dim,
            device=self.device
        )
    
class CubeUniformSampler(Sampler):
    def __init__(
        self, dim=1, centered=False, normalized=False, device='cuda'
    ):
        super(CubeUniformSampler, self).__init__(device=device)
        self.dim = dim
        self.centered = centered
        self.normalized = normalized
        self.var = self.dim if self.normalized else (self.dim / 12)
        self.cov = np.eye(self.dim, dtype=np.float32) if self.normalized else np.eye(self.dim, dtype=np.float32) / 12
        self.mean = np.zeros(self.dim, dtype=np.float32) if self.centered else .5 * np.ones(self.dim, dtype=np.float32)

        self.bias = torch.tensor(self.mean, device=self.device)

    def sample(self, size=10):
        with torch.no_grad():
            sample = np.sqrt(self.var) * (torch.rand(
                size, self.dim, device=self.device
            ) - .5) / np.sqrt(self.dim / 12)  + self.bias
        return sample
    
class NormalSampler(Sampler):
    def __init__(
        self, mean, cov=None, weight=None, device='cuda'
    ):
        super(NormalSampler, self).__init__(device=device)
        self.mean = np.array(mean, dtype=np.float32)
        self.dim = self.mean.shape[0]
        
        if weight is not None:
            weight = np.array(weight, dtype=np.float32)
        
        if cov is not None:
            self.cov = np.array(cov, dtype=np.float32)
        elif weight is not None:
            self.cov = weight @ weight.T
        else:
            self.cov = np.eye(self.dim, dtype=np.float32)
            
        if weight is None:
            weight = symmetrize(sqrtm(self.cov))
            
        self.var = np.trace(self.cov)
        
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        self.bias = torch.tensor(self.mean, device=self.device, dtype=torch.float32)

    def sample(self, size=4):
        sample = torch.randn(size, self.dim, device=self.device)
        with torch.no_grad():
            sample = sample @ self.weight.T
            if self.bias is not None:
                sample += self.bias
        return sample
    
class RandomGaussianMixSampler(Sampler):
    def __init__(
        self, dim=2, num=10, dist=1, std=0.4,
        standardized=True, estimate_size=2**15,
        batch_size=1024, device='cuda'
    ):
        super(RandomGaussianMixSampler, self).__init__(device=device)
        self.dim = dim
        self.num = num
        self.dist = dist
        self.std = std
        self.batch_size = batch_size
        
        centers = np.zeros((self.num, self.dim), dtype=np.float32)
        for d in range(self.dim):
            idx = np.random.choice(list(range(self.num)), self.num, replace=False)
            centers[:, d] += self.dist * idx
        centers -= self.dist * (self.num - 1) / 2
        
        maps = np.random.normal(size=(self.num, self.dim, self.dim)).astype(np.float32)
        maps /= np.sqrt((maps ** 2).sum(axis=2, keepdims=True))
        
        if standardized:
            mult = np.sqrt((centers ** 2).sum(axis=1).mean() + self.dim * self.std ** 2) / np.sqrt(self.dim)
            centers /= mult
            maps /= mult
        
        self.centers = torch.tensor(centers, device=self.device, dtype=torch.float32)  
        self.maps = torch.tensor(maps, device=self.device, dtype=torch.float32)
        
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self._estimate_moments(mean=False) # This can be also be done analytically
        
    def sample(self, size=10):          
        if size <= self.batch_size:
            idx = np.random.randint(0, self.num, size=size)
            sample = torch.randn(size, self.dim, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                sample = torch.matmul(self.maps[idx], sample[:, :, None])[:, :, 0] * self.std
                sample += self.centers[idx]
            return sample
        
        sample = torch.zeros(size, self.dim, dtype=torch.float32, device=self.device)
        for i in range(0, size, self.batch_size):
            batch = self.sample(min(i + self.batch_size, size) - i)
            with torch.no_grad():
                sample[i:i+self.batch_size] = batch
            torch.cuda.empty_cache()
        return sample

class Transformer(Sampler):
    def __init__(
        self, device='cuda'
    ):
        self.device = device
        
class LinearTransformer(Transformer):
    def __init__(
        self, weight, bias=None,
        device='cuda'
    ):
        super(LinearTransformer, self).__init__(
            device=device
        )
        
        self.fitted = False
        self.dim = weight.shape[0]
        self.weight = torch.tensor(weight, device=device, dtype=torch.float32, requires_grad=False)
        if bias is not None:
            self.bias = torch.tensor(bias, device=device, dtype=torch.float32, requires_grad=False)
        else:
            self.bias = torch.zeros(self.dim, device=device, dtype=torch.float32, requires_grad=False)
                
    def fit(self, base_sampler):
        assert base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        weight, bias = self.weight.cpu().numpy(), self.bias.cpu().numpy()
        
        self.mean = weight @ self.base_sampler.mean + bias
        self.cov = weight @ self.base_sampler.cov @ weight.T
        self.var = np.trace(self.cov)
        
        self.fitted = True
        return self
        
    def sample(self, size=4):
        assert self.fitted == True
        
        sample = torch.tensor(
            self.base_sampler.sample(size),
            device=self.device
        )
        with torch.no_grad():
            sample = sample @ self.weight.T
            if self.bias is not None:
                sample += self.bias
        return sample
    
class StandardNormalScaler(Transformer):
    def __init__(self, device='cuda'):
        super(StandardNormalScaler, self).__init__(device=device)
        
    def fit(self, base_sampler, size=1000):
        assert self.base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        self.dim = self.base_sampler.dim
        
        self.bias = torch.tensor(
            self.base_sampler.mean, device=self.device, dtype=torch.float32
        )
        
        weight = symmetrize(np.linalg.inv(sqrtm(self.base_sampler.cov)))
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        
        self.mean = np.zeros(self.dim, dtype=np.float32)
        self.cov = np.eye(self.dim, dtype=np.float32) # weight @ self.base_sampler.cov @ weight.T
        self.var = float(self.dim) #np.trace(self.cov)
        
        return self
        
    def sample(self, size=10):
        sample = torch.tensor(
            self.base_sampler.sample(size),
            device=self.device
        )
        with torch.no_grad():
            sample -= self.bias
            sample @= self.weight
        return sample
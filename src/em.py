import numpy as np
import torch
from torch.autograd.functional import jacobian

def batchItoEuler(f, y0, dt, t_fin, t_strt=0., beta=1.):
    '''
    :Parameters:
    f: callable(y), return (n, d) array
    y0: array of shape (n, d), n is count of points to propagate
    dt: float : iteration segment
    t_fin: float : fin time
    '''
    def deltaW(n, m, h):
        return np.random.normal(0.0, np.sqrt(h), (n, m))
    
    N = int((t_fin - t_strt)/dt)
    assert N * dt == (t_fin - t_strt)
    n = y0.shape[0]
    d = y0.shape[1]
    y_curr = y0.copy() # (n, d)
    for i in range(0, N):
        dWn = deltaW(n, d, dt) # (n, d)
        y_curr = y_curr + f(y_curr)*dt + np.sqrt(2./beta) * dWn
    return y_curr

def torchBatchItoEulerDistrib(distrib, y0, dt, t_fin, t_strt=0., beta=1.):
    '''
    :Parameters:
    distrib: torch.Distribution : stationary distribution
    y0: torch.tensor of shape (n, d), n is count of points to propagate
    dt: float: iteration segment
    t_fin: float : fin time
    '''
    def potential(x):
        return torch.sum(distrib.log_prob(x)/beta)
    
    def f(x):
        return jacobian(potential, x)
    
    _mean = torch.tensor(0.0, dtype=torch.float32).to(y0.device)
    _std = torch.tensor(np.sqrt(dt), dtype=torch.float32).to(y0.device)
    
    def deltaW(n, m):
        return torch.normal(_mean, _std, (n, m)).to(y0.device)
    
    N = int((t_fin - t_strt)/dt)
    assert N * dt == (t_fin - t_strt)
    n = y0.shape[0]
    d = y0.shape[1]
    y_curr = y0.clone() # (n, d)
    for i in range(0, N):
        dWn = deltaW(n, d) # (n, d)
        y_curr = y_curr + f(y_curr)*dt + np.sqrt(2./beta) * dWn
    return y_curr

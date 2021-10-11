from .icnn import DenseICNN
from .diffusion import Diffusion
from .utils import id_pretrain_model, train_diffusion_model

from .libopttorch import unconstr_solvers as solvers
from .libopttorch import step_size as ss
from .libopttorch import restarts

import torch
import numpy as np

class GenericBackwardPassSolver:

    def func(self, D, X_opt, X_ref):
        return D(X_opt) - (X_opt * X_ref).sum(-1).view(-1, 1)

    def func_grad(self, D, X_opt, X_ref):
        X_opt.requires_grad_(True)
        return D.push_nograd(X_opt) - X_ref
    
    def func_grad_norm(self, D, X_opt, X_ref):
        return torch.norm(self.func_grad(D, X_opt, X_ref).detach(), dim=-1)

    def __init__(self, verbose=False):
        self.verbose = verbose

    def solve(self, X, D):
        raise NotImplementedError()

class Baseline_BPS(GenericBackwardPassSolver):

    def __init__(
        self, max_iter=1000, grad_tol=1e-6, 
        no_progress_stop=5, min_lambda=1e-8, verbose=False):
        super().__init__(verbose=verbose)
        self.max_iter = max_iter
        self.grad_tol = grad_tol
        self.no_progress_stop = no_progress_stop
        self.min_lambda = min_lambda
    
    def solve(self, X, D):
        Xs_curr = X
        Xs_prev = Xs_curr.clone()
        Xs_prev.detach_()
        lr_base = 1.0
        j = 0
        max_grad_norm_history = []
        mask = torch.range(0, X.size(0) - 1, dtype=int)
        while True:
            _Xs_prev = Xs_prev[mask] # we optimize
            _Xs_prev.requires_grad_(True) 
            _Xs_curr = Xs_curr[mask] # reference
            _Xs_prev.requires_grad_(True)
            _grad = self.func_grad(D, _Xs_prev, _Xs_curr).detach()
            prev_func_vals = self.func(D, _Xs_prev, _Xs_curr).detach()
            prev_grad_norms = torch.norm(_grad, dim=-1)
            _lambdas = (lr_base / torch.sqrt(prev_grad_norms + 1e-6)).view(-1, 1)
            while True:
                _new_Xs = _Xs_prev - _lambdas * _grad
                curr_grad_norms = self.func_grad_norm(D, _new_Xs, _Xs_curr)
                diff = prev_grad_norms - curr_grad_norms
                if torch.sum(diff <= 0.) == 0:
                    break
                if torch.min(_lambdas) < self.min_lambda:
                    break
                _lambdas[diff <= 0.] *= 0.5
            
            _Xs_prev = _Xs_prev - _lambdas * _grad
            final_grad_norms = self.func_grad_norm(D, _Xs_prev, _Xs_curr)
            max_grad_norm_history.append(final_grad_norms.max())
            acheve_mask = final_grad_norms < self.grad_tol
            Xs_prev[mask] = _Xs_prev.detach()
            mask = mask[~acheve_mask]

            if len(mask) == 0:
                if self.verbose:
                    print('pushback has taken {} iters'.format(j))
                    print('max grad diff: ', self.func_grad_norm(D, Xs_prev, Xs_curr).max())
                break
            if j > self.max_iter:
                if self.verbose:
                    print('stopped since max_iter acheved')
                    print('N not converged: ', len(mask))
                    print('max grad diff: ', self.func_grad_norm(D, Xs_prev, Xs_curr).max())
                break
            if j > self.no_progress_stop:
                if np.max(max_grad_norm_history[-self.no_progress_stop:]) - np.min(max_grad_norm_history[-self.no_progress_stop:]) < 1e-16:

                    if self.verbose:
                        print('stopped since no progress acheved')
                        print('N not acheved: ', len(mask))
                        print('pushback has taken {} iters'.format(j))
                        print('max grad diff: ', self.func_grad_norm(D, Xs_prev, Xs_curr).max())
                    break
            j += 1
        return Xs_prev.detach()

class LibopttorchFoBackwardPassSolver(GenericBackwardPassSolver):

    def __init__(
        self, solver_cls, *solver_args, max_iter=100, 
        tol=1e-6, verbose=0, **solver_kwargs):

        super().__init__(verbose=verbose)
        self.lop_solver_cls = solver_cls
        self.lop_solver_args = solver_args
        self.lop_solver_kwargs = solver_kwargs
        self.max_iter = max_iter
        self.tol= tol
    
    def solve(self, X, D):
        Xs_curr = X
        Xs_prev = Xs_curr.clone()
        Xs_prev.detach_()

        def targ_func(X, mask, bool_mask=None):
            if bool_mask is None:
                return self.func(D, X, Xs_prev[mask])
            return self.func(D, X[bool_mask], Xs_prev[mask][bool_mask])
        
        def targ_func_grad(X, mask, bool_mask=None):
            if bool_mask is None:
                return self.func_grad(D, X, Xs_prev[mask]).detach()
            return self.func_grad(D, X[bool_mask], Xs_prev[mask][bool_mask]).detach()
        
        lop_solver = self.lop_solver_cls(
            targ_func, targ_func_grad, *self.lop_solver_args, **self.lop_solver_kwargs)
        Xs_prev = lop_solver.solve(
            Xs_prev, max_iter = self.max_iter, tol=self.tol, disp=self.verbose)

        return Xs_prev.detach()

class RegBB_BPS(LibopttorchFoBackwardPassSolver):
    
    def __init__(
        self, verbose=0, tol=1e-6, max_iter=100, init_alpha=1e-4, 
        _type=1, delta=20.):
        
        super().__init__(
            solvers.fo.BarzilaiBorweinMethod, 
            verbose=verbose, 
            tol=tol, max_iter=max_iter, init_alpha=init_alpha, _type=_type, delta=delta)

class GD_GB_BPS(LibopttorchFoBackwardPassSolver):
    
    def __init__(
        self, verbose=0, tol=1e-6, max_iter=100, init_alpha=0.5, 
        device='cuda:0', raise_small_alp=False, small_alp=1e-3):
        
        super().__init__(
            solvers.fo.GradientDescent, ss.GradBacktracking(
                init_alpha, est_grad=True, device=device, 
                raise_small_alp=raise_small_alp, small_alp=small_alp), 
            verbose=verbose, tol=tol, max_iter=max_iter)
    
class GD_CSS_BPS(LibopttorchFoBackwardPassSolver):
    
    def __init__(self, verbose=0, tol=1e-6, max_iter=100, alpha=2e-1, device='cuda:0'):
        
        super().__init__(
            solvers.fo.GradientDescent, 
            ss.ConstantStepSize(alpha, device=device), verbose=verbose, tol=tol, max_iter=max_iter)

class AccGD_BPS(LibopttorchFoBackwardPassSolver):
    
    def __init__(self, verbose=0, tol=1e-6, max_iter=100, alpha=4e-1, device='cuda:0'):
        
        super().__init__(
            solvers.fo.AcceleratedGD, 
            ss.ConstantStepSize(
                alpha, device=device), verbose=verbose, tol=tol, max_iter=max_iter)

class CG_GB_BPS(LibopttorchFoBackwardPassSolver):
    
    def __init__(self, verbose=0, tol=1e-6, max_iter=100, init_alpha=0.5, 
        device='cuda:0', raise_small_alp=False, small_alp=1e-3, restart_lim_dim=None):
        
        super().__init__(
            solvers.fo.ConjugateGradientFR, ss.GradBacktracking(
                init_alpha, est_grad=True, raise_small_alp=raise_small_alp, 
                small_alp=small_alp, device=device), restart=restarts.Restart(restart_lim_dim),
            verbose=verbose, tol=tol, max_iter=max_iter)
    
    
class CG_CSS_BPS(LibopttorchFoBackwardPassSolver):
    
    def __init__(self, verbose=0, tol=1e-6, max_iter=100, alpha=2e-1, device='cuda:0', restart_lim_dim=None):
        
        super().__init__(
            solvers.fo.ConjugateGradientFR, ss.ConstantStepSize(
                alpha, device=device), restart=restarts.Restart(restart_lim_dim), 
            verbose=verbose, tol=tol, max_iter=max_iter)

def grad_backward_path(diff, X, solver):
    Xs_backward = []
    Xs_backward.append(X.clone())
    Ds = diff.Ds
    for i in range(len(Ds)):
        if solver.verbose:
            print('Ds[{}] pushback starts'.format(len(Ds) - i - 1))
        ds_num = -i-1
        D = Ds[ds_num]
        X = Xs_backward[-1].clone()
        X_prev = solver.solve(X, D)
        Xs_backward.append(X_prev.detach())
    return Xs_backward

def grad_backward_path_norms_estimator(Xs_forward, Xs_backward, reduction='max', which='all'):
    with torch.no_grad():
        assert reduction in ['max', 'mean']
        assert which in ['first', 'last', 'all']
        assert len(Xs_forward) == len(Xs_backward)
        reduction_func = torch.max if reduction=='max' else torch.mean
        if which == 'all':
            norms = []
            for i in range(len(Xs_forward)):
                batch_norms = torch.norm(Xs_forward[len(Xs_forward) - i - 1] - Xs_backward[i], dim=-1)
                norms.append(reduction_func(batch_norms).item())
            return norms
        if which == 'first':
            batch_norms = torch.norm(Xs_forward[0] - Xs_backward[-1], dim=-1)
            return reduction_func(batch_norms).item()
        if which == 'last':
            batch_norms = torch.norm(Xs_forward[-1] - Xs_backward[0], dim=-1)
            return reduction_func(batch_norms).item()

def grad_backward_path_pdfs_estimator(diff, Xs_forward, Xs_backward, reduction='max'):
    assert reduction in ['max', 'mean', 'hist', 'none']
    assert len(Xs_forward) == len(Xs_backward)
    ref_log_prob = diff.log_prob_trace(Xs_forward, backward_order=False).view(-1)
    obt_log_prob = diff.log_prob_trace(Xs_backward).view(-1)
    pdf_log_diffs = torch.abs(ref_log_prob - obt_log_prob).detach().cpu().numpy()
    if reduction == 'none':
        return pdf_log_diffs
    if reduction == 'max':
        return np.max(pdf_log_diffs)
    if reduction == 'mean':
        return np.mean(pdf_log_diffs)
    if reduction == 'hist':
        plt.hist(pdf_log_diffs, bins=100)
        plt.show()


from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize, check_grad
from scipy.spatial.distance import cdist
from scipy.special import factorial
from scipy.stats import multivariate_normal, uniform
import sys
from multipledispatch import dispatch
from collections.abc import Callable
import torch

class multivariate_uniform:

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.dim = self.loc.shape[0]
        self._u_distrib = uniform(self.loc, self.scale)
    
    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        if isinstance(size, tuple):
            size = size + (self.dim,)
        else:
            raise Exception('"size" argument not understood')
        return self._u_distrib.rvs(size=size)

    def logpdf(self, x):
        x_logpdf = np.sum(self._u_distrib.logpdf(x), axis=-1)
        return x_logpdf
    
    def pdf(self, x):
        x_pdf = np.prod(self._u_distrib.pdf(x), axis=-1)
        return x_pdf

def get_mv_uniform_loc_scale_params(mean, var):
    mean = np.array(mean)
    var = np.array(var)
    if len(var.shape) == 2:
        assert (np.diag(np.diag(var)) == var).all()
        var = np.diag(var)
    loc = mean - np.sqrt(3. * var)
    scale = np.sqrt(12. * var)
    return loc, scale

def make_mv_uniform(mean, var):
    loc, scale = get_mv_uniform_loc_scale_params(mean, var)
    return multivariate_uniform(loc, scale)

def compute_gaussian_kernel(x, y, sigmasq=1e-1, scale=1e-4):
    return (scale / np.sqrt(
        2 * np.pi * sigmasq) ** x.shape[1]) * np.exp(
            -(1 / (2 * sigmasq)) * np.power(
                cdist(x, y, metric='euclidean'), 2))

def compute_poly_n_basis(Nbasis_basic, dim, degree=3):
    nb = 1
    for dd in range(degree):
        combos = combinations_with_replacement(range(dim), dd + 1)
        for c in combos:
            nb += 1
    return int(min(Nbasis_basic, nb))

def compute_poly_kernel(x, y, degree=3, scale=1e-4):
    n = x.shape[0]
    d = x.shape[1]
    Q = [np.ones((n, 1))]
    for dd in range(degree):
        combos = combinations_with_replacement(range(d), dd + 1)
        for c in combos:
            Q.append(np.prod(x[:, c], axis=1).reshape((-1, 1)))
    Q = scale * np.concatenate(Q, axis=1)
    n_basis = min(y.shape[0], Q.shape[1])
    return Q[:, :n_basis]

def run_diffusion(
    init_pdf, targ_pot, t_fin, n_steps, dim, n_basis, 
    meanx=None, covx=None, meany=None, covy=None, beta=1.,
    gamma=1e-6, kernel_type='gaussian', kernel_params={}, 
    n_zero_spls=20000, verbose=False, do_uniform=True, do_hess=False,
    nufunc_norm_constant=None, opt_method='l-bfgs-b', options={'gtol':1e-5, 'maxiter':500}):
    '''
    runs diffusion using frogner's method
    :Parameters:
    init_pdf : callable : pdf of initial distribution (dim first format)
    targ_pdf : callable : target potential
    t_fin : float : final diffusion timestep
    n_steps : int : number of JKO steps
    dim : int : space dimensionality
    n_basis : int : count of basis functions
    Regarding the other parameteres see `compute_flow` function
    :Result:
    qfunc: callable : unnormalized resulting pdf
    '''

    if meanx is None:
        meanx = np.zeros(dim) 
    if covx is None:
        covx = np.eye(dim)
    if meany is None:
        meany = np.zeros(dim)
    if covy is None:
        covy = np.eye(dim)
    
    tau = 2 * t_fin / float(n_steps)
    
    qfunc = init_pdf
    for ii in range(n_steps):
        if verbose:
            print('->->->->-> start step {}'.format(float(ii+1)))
        qfunc0, alpha, ng, xsamp = compute_flow(
            qfunc, n_zero_spls, n_basis, meanx, covx, meany, covy, tau, 
            gamma, targ_pot,beta=beta, kernel_type=kernel_type, kernel_params=kernel_params,
            do_uniform=do_uniform, do_hess=do_hess, verbose=verbose, 
            nufunc_norm_constant=nufunc_norm_constant,
            opt_method=opt_method, options=options)
        qfunc = lambda x : qfunc0(x.T)
        if verbose:
            print('t %g, final normg %g' % (float(ii + 1) * tau / 2, ng))
    return qfunc

def compute_flow(
    nufunc, Nsamp, Nbasis, meanx, covx, meany, covy, 
    tau, gamma, wfunc, beta=1.0, kernel_type='gaussian', kernel_params={}, 
    lamb=0.0, nufunc_norm_constant=None,
    do_uniform=True, do_hess=False, verbose=False,
    opt_method='l-bfgs-b', tol_main=None, callback=None, options=None):
    '''
    Compute Frogner's wFlow
    :Parameters:
    nufunc : callable : \nu (or \rho_t) - current flow distribution
    Nsamp : int : sample size from \nu_0 and \mu_0
    Nbasis : int : basis dimensionality
    sigmaQ : float : gaussian basis kernels variance parameter
    meanx, covx : list/np.ndarray : parameters of \mu_0 distribution
    meany, covy : list/np.ndarray : parameters of \nu_0 distribution
    tau : float : JKO timestep
    gamma : float : float : W-regularization coefficient
    wfunc : callable : target potential 
    beta : float : process inverse temperature
    '''
    assert kernel_type in ['gaussian', 'poly']

    # xsamp and ysamp are samples from \mu_0 and \nu_0 (see original article)
    if do_uniform:
        x_distrib = make_mv_uniform(meanx, covx)
        y_distrib = make_mv_uniform(meany, covy)
    else:
        x_distrib = multivariate_normal(meanx, covx)
        y_distrib = multivariate_normal(meany, covy)
    xsamp = x_distrib.rvs((Nsamp,))
    ysamp = y_distrib.rvs((Nsamp,))
    mu0 = x_distrib.pdf(xsamp)
    nu0 = y_distrib.pdf(ysamp)

    # l2 metric squared
    c = np.sum((xsamp - ysamp)**2, axis=1)

    ###################
    ### functions
    if kernel_type == 'gaussian':
        compute_kernel = compute_gaussian_kernel
    else:
        compute_kernel = compute_poly_kernel
        kernel_degree = 3
        if 'degree' in kernel_params:
            kernel_degree = kernel_params['degree']
        Nbasis = compute_poly_n_basis(Nbasis, len(meanx), degree=kernel_degree)
        if verbose:
            print('Nbasis:', Nbasis)
        
    Qx = compute_kernel(xsamp, xsamp[:Nbasis], **kernel_params)
    Qy = compute_kernel(ysamp, ysamp[:Nbasis], **kernel_params)

    Qx_eval = lambda alpha : np.matmul(Qx, alpha)
    Qxt_eval = lambda alpha : np.matmul(Qx.T, alpha)
    Qy_eval = lambda alpha : np.matmul(Qy, alpha)
    Qyt_eval = lambda alpha : np.matmul(Qy.T, alpha)

    if do_hess:
        QxtQxt = Qx.T.reshape((Nbasis, 1, Nsamp)) * Qx.T.reshape((1, Nbasis, Nsamp))
        QxtQyt = Qx.T.reshape((Nbasis, 1, Nsamp)) * Qy.T.reshape((1, Nbasis, Nsamp))
        QytQyt = Qy.T.reshape((Nbasis, 1, Nsamp)) * Qy.T.reshape((1, Nbasis, Nsamp))
        QxtQxt_eval = lambda alpha : np.sum(QxtQxt * alpha.reshape((1, 1, -1)), axis=2)
        QxtQyt_eval = lambda alpha : np.sum(QxtQyt * alpha.reshape((1, 1, -1)), axis=2)
        QytQyt_eval = lambda alpha : np.sum(QytQyt * alpha.reshape((1, 1, -1)), axis=2)
    
    # here we estimate target potential in xsamp points
    w = wfunc(xsamp.T).reshape((-1,))

    # estimate \rho_t in sampled points
    nu = nufunc(ysamp.T).reshape((-1,))
    if nufunc_norm_constant is not None:
        nu = nu / nufunc_norm_constant
    else:
        # nufunc_norm_constant = normalize_unnorm_pdf_np(ysamp, nufunc, y_distrib, method='importance')
        # print('nufunc_norm_constant:', nufunc_norm_constant)
        nufunc_norm_constant = np.max(nu)
        nu = nu / nufunc_norm_constant
    if verbose:
        print('nu: [' + str(np.min(nu)) + ', ' + str(np.max(nu)) + ']')


    #########################
    ### L2 OBJECTIVE

    obj = lambda alpha : flow_obj_l2(alpha, tau, gamma, beta, w, c, Qx_eval, Qy_eval, nu, mu0, nu0, lamb)
    jac = lambda alpha : flow_jac_l2(alpha, tau, gamma, beta, w, c, Qx_eval, Qxt_eval, Qy_eval, Qyt_eval, nu, mu0, nu0, lamb)
    if do_hess:
        hess = lambda alpha : flow_hess_l2(
            alpha, tau, gamma, beta, w, c, Qx_eval, Qy_eval, QxtQxt_eval, QxtQyt_eval, QytQyt_eval, nu, mu0, nu0, lamb)
    else:
        hess = None

    alpha0 = np.zeros((2 * Nbasis,), order='C')
    
    
    ####################
    ### OPTIMIZATION

    # custom second-order optimization
    if opt_method == 'so-custom':
        alpha = alpha0
        ftol = 1e-24
        gtol = 1e-10 #KERNEL_SCALE * 1e-4
        sz0 = 1e0
        c1 = 1e-4
        c2 = 0.9
        obj0 = obj(alpha)
        jalpha = jac(alpha)
        for ii in range(1000):
            sz = sz0
            v = np.dot(np.linalg.pinv(hess(alpha)), jalpha)
            while sz > 1e-32:
                alphatest = alpha - sz * v
                objtest = obj(alphatest)
                jactest = jac(alphatest)
                fdiff = np.abs(objtest - obj0)
                if ((objtest <= (obj0 + c1 * sz * np.dot(jalpha, v)))): # and 
                    alpha = alphatest
                    obj0 = objtest
                    jalpha = jactest
                    break
                else:
                    sz = float(3 / 4) * sz
            ng = np.max(np.abs(jac(alpha)))
            if verbose:
                print('obj, ng(-), sz, fdiff: ' + str(obj(alpha)) + ', ' + str(ng) + ', ' + str(sz) + ', ' + str(fdiff))
            if sz <= 1e-32 or ng < gtol or fdiff < ftol:
                break
        class res:
            x = alpha
    
    else:
        res = minimize(obj, alpha0, jac=jac, hess=hess, method=opt_method, tol=tol_main, callback=callback, options=options)

        obj0 = obj(res.x) # resulting objective: D(g^*, h^*)
        if verbose:
            ng = np.max(np.abs(jac(res.x))) 
            print('obj, ng(-): ' + str(obj0) + ', ' + str(ng))
            sys.stdout.flush()

    alpha = res.x
    if verbose:
        print('alpha: [' + str(np.amin(alpha[:Nbasis])) + ',' + str(np.amax(alpha[:Nbasis])) + ']')
    gfunc = lambda x : np.dot(compute_kernel(x, xsamp[:Nbasis], **kernel_params), alpha[:Nbasis]) # g^*
    mufunc = lambda x : np.exp(-(beta / tau) * gfunc(x) - beta * wfunc(x.T).reshape((-1,))) # \rho_{t + \tau} - resulting unnorm pdf 

    ng = np.linalg.norm(jac(alpha))

    return mufunc, alpha, ng, xsamp[:Nbasis]

def flow_obj_l2(alpha, tau, gamma, beta, w, c, Qx_eval, Qy_eval, nu, mu0, nu0, lamb):
    N = mu0.shape[0]
    Nbasis = round(alpha.size / 2)

    alphag = alpha[:Nbasis]
    alphah = alpha[Nbasis:]

    g = Qx_eval(alphag)
    h = Qy_eval(alphah)

    fstar = (1 / beta) * np.exp(-(beta / tau) * g - beta * w)

    ell2sq = (1 / 2) * (1 / gamma) * np.power(np.maximum(g + h - c, 0), 2)

    obj = -np.mean(-tau * fstar / mu0 + h * nu / nu0 - ell2sq / (mu0 * nu0))
    obj += (lamb / (2 * N)) * (np.sum(np.power(alphag, 2)) + np.sum(np.power(alphah, 2)))


    return obj


def flow_jac_l2(alpha, tau, gamma, beta, w, c, Qx_eval, Qxt_eval, Qy_eval, Qyt_eval, nu, mu0, nu0, lamb):
    N = mu0.shape[0]
    Nbasis = round(alpha.size / 2)

    alphag = alpha[:Nbasis]
    alphah = alpha[Nbasis:]

    g = Qx_eval(alphag)
    h = Qy_eval(alphah)

    fstar = (1 / beta) * np.exp(-(beta / tau) * g - beta * w)

    dfstar = beta * fstar

    dell2sq = (1 / gamma) * np.maximum(g + h - c, 0)

    Qxetc = Qxt_eval(dfstar / mu0 - dell2sq / (mu0 * nu0))
    Qyetc = Qyt_eval(nu / nu0 - dell2sq / (mu0 * nu0))

    grad = np.concatenate([-(1 / N) * Qxetc + lamb / N * alphag,
                           -(1 / N) * Qyetc + lamb / N * alphah], axis=0)

    return grad

def flow_hess_l2(alpha, tau, gamma, beta, w, c, Qx_eval, Qy_eval, QxtQxt_eval, QxtQyt_eval, QytQyt_eval, nu, mu0, nu0, lamb):
    N = mu0.shape[0]
    Nbasis = round(alpha.size / 2)

    alphag = alpha[:Nbasis]
    alphah = alpha[Nbasis:]

    g = Qx_eval(alphag)
    h = Qy_eval(alphah)

    fstar = (1 / beta) * np.exp(-(beta / tau) * g - beta * w)

    d2fstar = beta * beta * fstar

    d2ell2sq = (1 / gamma) * ((g + h - c) >= 0)

    QxQxetc = QxtQxt_eval((-1 / tau) * d2fstar / mu0 - d2ell2sq / (mu0 * nu0))
    QxQyetc = QxtQyt_eval(-d2ell2sq / (mu0 * nu0))
    QyQyetc = QytQyt_eval(-d2ell2sq / (mu0 * nu0))

    hess = -(1 / N) * np.concatenate([np.concatenate([QxQxetc, QxQyetc], axis=1),
                                      np.concatenate([QxQyetc.T, QyQyetc], axis=1)],
                                     axis=0)
    hess += (lamb / N) * np.eye(2 * Nbasis)

    return hess


@dispatch(torch.Tensor, Callable, object)
def KL_targ_distrib(X, train_pdf, targ, norm_constant=1.):
    q_ = train_pdf(X.cpu().numpy().T)/norm_constant
    if q_.min() < -1e-16:
        raise Exceptin('wrong train_pdf')
    q_ = np.maximum(q_, 1e-16)
    log_q_ = np.log(q_)
    log_q_exact = targ.log_prob(X).cpu().numpy()
    res = np.mean(log_q_exact - log_q_)
    return res

@dispatch(torch.Tensor, Callable, object)
def KL_train_distrib_importance(X, train_pdf, targ, norm_constant=1.):
    qexact_ = np.exp(targ.log_prob(X).cpu().numpy())
    q_ = train_pdf(X.cpu().numpy().T)/norm_constant
    res = np.nanmean((q_ / np.maximum(qexact_, 1e-16)) * np.log(q_ / np.maximum(qexact_, 1e-16)))
    return res

@dispatch(int, Callable, object)
def normalize_unnorm_pdf(n, train_pdf, true_distrib, method='importance'):
    X = true_distrib.sample((n,))
    return normalize_unnorm_pdf(X, train_pdf, true_distrib, method=method)

def accept_reject_sample(n, train_pdf, prop_distrib, norm_constant=1., C=10.0):
    curr_n = n
    curr_sample = None
    while True:
        Y = prop_distrib.sample((curr_n,))
        np_Y = Y.cpu().numpy()
        q_Y = torch.exp(prop_distrib.log_prob(Y))
        u_unsc = np.random.uniform(low=0., high=1., size=curr_n)
        u = q_Y.cpu().numpy() * u_unsc * C
        gamma = train_pdf(np_Y.T).reshape((-1,))/norm_constant
        acc_Y = np_Y[u < gamma]
        if curr_sample is None:
            curr_sample = acc_Y
        else:
            curr_sample = np.concatenate([
                curr_sample, acc_Y])
        curr_n = n - curr_sample.shape[0]
        if curr_n == 0:
            return curr_sample
    

@dispatch(torch.Tensor, Callable, object)
def normalize_unnorm_pdf(X, train_pdf, true_distrib, method='importance'):
    assert method in ['leastsq', 'importance']
    if method == 'leastsq':
        q_exact_ = np.exp(true_distrib.log_prob(X).cpu().numpy())
        q_ = train_pdf(X.cpu().numpy().T)
        qc = np.sum(q_ * q_exact_) / np.sum(q_ * q_)
        return 1./qc
    elif method == 'importance':
        pdf_true_spl = np.exp(true_distrib.log_prob(X).cpu().numpy())
        pdf_train_spl = train_pdf(X.cpu().numpy().T)
        return np.mean(pdf_train_spl/pdf_true_spl)

@dispatch(int, Callable, object)
def normalize_unnorm_pdf_np(n, train_pdf, true_distrib, method='importance'):
    X = true_distrib.rvs((n,))
    return normalize_unnorm_pdf_np(X, train_pdf, true_distrib, method=method)

@dispatch(np.ndarray, Callable, object)
def normalize_unnorm_pdf_np(X, train_pdf, true_distrib, method='importance'):
    assert method in ['leastsq', 'importance']
    s_true = true_distrib.pdf(X)
    s_train = train_pdf(X.T)
    if s_train.min() < -1e-16:
        raise Exceptin('wrong train_pdf')
    s_train = np.maximum(s_train, 1e-16)
    if method == 'leastsq':
        return 1./np.sum(s_true * s_train) / np.sum(s_train * s_train)
    elif method == 'importance':
        return np.mean(s_train/s_true)

import numpy as np
import torch

from botorch.acquisition import UpperConfidenceBound, qKnowledgeGradient
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
from scipy.stats._qmc import LatinHypercube


def init_data(n_point, dimension, bounds, function, support=None, seed=None, **kwargs):
    """

    :param n_point: int
        Number of initial point.
    :param dimension: int
        Dimension of the input space.
    :param bounds: ndarray of shape (2, )
        Lower and Upper bounds of the input space.
    :param function: callable
        Function to evaluate.
    :param support: ndarray
        Number of bins of the histogram
    :param seed:
    :return:
    """
    # train_x = torch.Tensor(np.array([
    #     (bounds[0] - bounds[1]) * np.random.rand(dimension) + bounds[1] for _ in range(n_point)
    # ]))
    # train_x = train_x.double()

    sample = LatinHypercube(d=dimension, seed=seed).random(n=n_point)
    train_x = torch.Tensor((bounds[0] - bounds[1]) * sample + bounds[1]).double()

    train_y, train_s = function(train_x, **kwargs)
    if type(train_y) == np.ndarray:
        train_y = train_y.astype(float)
        train_y = torch.Tensor(train_y)
    train_y = train_y.reshape(-1, 1).double()

    train_h = None
    if support is not None:
        train_h = np.array([(np.histogram(s, bins=support)[0]) for s in train_s])
        train_h = train_h / train_h.sum(axis=1)[:, None]
        train_h = torch.Tensor(train_h).double()
    return train_x, train_y, train_h


def init_model(train_h, train_y, kernel, state_dict=None):
    gp = SingleTaskGP(train_h, train_y, covar_module=kernel)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if state_dict is not None:
        gp.load_state_dict(state_dict)
    return gp, mll


def histo_constraint(eta):
    indeces = torch.arange(0, eta)
    coefficients = torch.ones(eta).double()
    rhs = 1.0
    return indeces, coefficients, rhs


def optimize_acquisition(gp, bounds, constraint=None):
    ucb = UpperConfidenceBound(gp, beta=4, maximize=False)
    # ucb = qKnowledgeGradient(gp)
    candidate, _ = optimize_acqf(ucb, bounds=bounds, q=1, num_restarts=20,
                                 raw_samples=50, equality_constraints=constraint)
    # BOWS -> num_restarts = 5, raw_samples = 10
    # BO -> num_restarts = 20, raw_samples = 50
    new_point = candidate.detach()
    return new_point


def new_observation(x, function, eta, **kwargs):
    y, sample = function(torch.Tensor(x), **kwargs)
    if type(y) == np.ndarray:
        y = y.astype(float)
        y = torch.Tensor(y)
    y = y.reshape(-1, 1).double()
    h = np.array([np.histogram(s, bins=eta)[0] for s in sample])
    h = h / h.sum(axis=1)[:, None]
    h = torch.Tensor(h).double()
    return y, h

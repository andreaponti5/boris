import torch


def alpine01(x, **kwargs):
    """
    https://al-roomi.org/benchmarks/unconstrained/n-dimensions/162-alpine-function-no-1
    sum_i abs(x_i * sin(x_i) + 0.1 * x_i)
    Each component assumes value in [0, 10]

    :param x: Tensor n*d dimension
    :return: Tensor[float]
    """
    assert (x >= -10).all() and (x <= 10).all()
    components = torch.abs(x * x.sin() + 0.1 * x)
    return torch.sum(components, dim=1), components


def deb01(x, **kwargs):
    """
    http://infinity77.net/global_optimization/test_functions_nd_D.html#n-d-test-functions-d
    Each component assumes value in [0, 1]
    :param x:
    :param kwargs:
    :return:
    """
    assert (x >= -1).all() and (x <= 1).all()
    components = torch.pow(torch.sin(5 * torch.pi * x), 6)
    return - torch.sum(components, dim=1) / x.shape[1], components


def michalewicz(x, **kwargs):
    """
    http://www.sfu.ca/~ssurjano/michal.html
    Each component assumes value in [0, 1]

    :param x: Tensor n*d dimension
    :return: Tensor[float]
    """
    assert (x >= 0).all() and (x <= torch.pi).all()
    m = 10 if kwargs.get("m") is None else kwargs.get("m")
    i = torch.arange(1, x.shape[1] + 1)
    components = x.sin() * torch.pow(((i * torch.pow(x, 2)) / torch.pi).sin(), 2 * m)
    return -torch.sum(components, dim=1), components


def rastrigin(x, **kwargs):
    """
    https://www.sfu.ca/~ssurjano/rastr.html
    Each component assumes value in [-10, 31]

    :param x:
    :param kwargs:
    :return:
    """
    assert (x >= -5.12).all() and (x <= 5.12).all()
    components = torch.pow(x, 2) - 10 * torch.cos(2 * torch.pi * x)
    return 10 * x.shape[1] + torch.sum(components, dim=1), components


def rosenbrock(x, **kwargs):
    """
    http://www.sfu.ca/~ssurjano/rosen.html
    Each component assumes value in [0, 1102581] [0, 3900]

    :param x: Tensor n*d dimension
    :return: Tensor[float]
    """
    assert (x >= -2.048).all() and (x <= 2.048).all()
    x_i = x[:, :-1]
    x_shift = x[:, 1:]
    components = 100 * torch.pow(x_shift - torch.pow(x_i, 2), 2) + torch.pow(x_i - 1, 2)
    return torch.sum(components, dim=1), components


def schwefel(x, **kwargs):
    """
    http://www.sfu.ca/~ssurjano/schwef.html
    Each component assumes value in [-11180.3398, 11180.3398].

    :param x:
    :return:
    """
    assert (x >= -500).all() and (x <= 500).all()
    components = x * torch.sqrt(torch.abs(x))
    return 418.9829 * x.shape[1] - torch.sum(components, dim=1), components


def styb_tang(x, **kwargs):
    """
    http://www.sfu.ca/~ssurjano/stybtang.html
    Each component assumes value in [-80, 250].

    :param x:
    :param kwargs:
    :return:
    """
    assert (x >= -5).all() and (x <= 5).all()
    components = torch.pow(x, 4) - 16 * torch.pow(x, 2) + 5 * x
    return torch.sum(components, dim=1) / 2, components


def vincent(x):
    """
    Each component assumes value in [-1, 1].

    :param x:
    :return:
    """
    assert (x >= 0.25).all() and (x <= 10).all()
    components = torch.sin(10 * torch.log(x))
    return torch.sum(components, dim=1), components


def plateau(x):
    """
    Each component assumes value in [-6, 5].
    :param x:
    :return:
    """
    assert (x >= -5.12).all() and (x <= 5.12).all()
    components = torch.floor(x)
    return torch.sum(components, dim=1) + 30, components

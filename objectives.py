import numpy as np
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


def sensor_placement(x, **kwargs):
    x = torch.round(x, decimals=0).type(torch.long)
    res = []
    for sp in x:
        x_binary = torch.zeros(kwargs["impact_matrix"].shape[1])
        x_binary[sp] = 1
        x_matrix = _extract_x_matrix_from_solution(x_binary, kwargs["impact_matrix"])
        res.append(_mean_impact(x_matrix, kwargs["impact_matrix"]))
    res_y = np.array([element[0] for element in res])
    res_s = [element[1] for element in res]
    return res_y, res_s


def _extract_x_matrix_from_solution(solution, impact_matrix):
    """
    Extract which sensor should be active based on the solution parameter.
    For each scenario, the active sensor should be the one active in the solution with the minimum detection time.

    :solution: the sensor placement to test (list of bool).
    :det_times: matrix of detection times (numpy 2D-array).
    :scenarios: list of nodes which can be scenarios.
    :sensors: list of nodes in which can be placed a sensor.
    :return: matrix of the best sensor for each scenario (numpy 2D-array).
    """
    # Get active sensors from solution
    active_sensors = [index for index, sensor in enumerate(solution) if sensor == 1]
    x = []
    for scenarios_index in range(impact_matrix.shape[0]):
        scenario_row = list(np.zeros(impact_matrix.shape[1], dtype=np.int))
        min_det_tmp = 100000
        min_sensor_index = 0
        # Check which sensor out of the active sensor of the solution should be active in this scenario
        for sensor_index in active_sensors:
            det_time_tmp = impact_matrix[scenarios_index][sensor_index]
            if (det_time_tmp >= 0) & (det_time_tmp <= min_det_tmp):
                min_det_tmp = det_time_tmp
                min_sensor_index = sensor_index
        scenario_row[min_sensor_index] = 1
        x.append(scenario_row)
    return x


def _mean_impact(x, impact_matrix):
    """
    Compute the objective function (p-median) described in (pag.9)
    [Berry, J., Hart, W. E., Phillips, C. A., Uber, J. G., & Watson, J. P. (2006).
    Sensor placement in municipal water networks with temporal integer programming models.
    Journal of water resources planning and management, 132(4), 218-224.]

    :solution: the sensor placement to test (list of bool).
    :det_times: matrix of detection times (numpy 2D-array).
    :scenarios: list of nodes which can be scenarios.
    :sensors: list of nodes in which can be placed a sensor.
    :return: sum_(a in A) [ alpha_a * sum_(i in L_a) ( d_(ai) * x_(ai) ) ]
    """
    # if all(location == 0 for location in solution):
    #     return 86400
    result = 0
    scenario_times = []
    for scenario_index in range(impact_matrix.shape[0]):
        tmp_sum = 0
        for sensor_index in range(impact_matrix.shape[1]):
            # Take the impact for the couple scenario-sensor
            d_scenario_sensor = impact_matrix[scenario_index][sensor_index]
            x_scenario_sensor = x[scenario_index][sensor_index]
            # Sum of impact of the active sensors
            tmp_sum += d_scenario_sensor * x_scenario_sensor
        # Lets consider that all the scenarios have the same probability
        # TODO Probability of each scenario can be parametrized
        scenario_times.append(tmp_sum)
        result += tmp_sum / impact_matrix.shape[0]
    return result, scenario_times

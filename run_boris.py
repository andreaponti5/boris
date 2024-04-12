import time

import gpytorch
import numpy as np
import torch
from botorch import fit_gpytorch_mll

import objectives
import utils
from mlp import regressor, predict
from bo import init_data, init_model, optimize_acquisition, histo_constraint, new_observation

# Experiment config
exp_config = {
    "TRIAL": 10,
    "dimension": 100,
    "function": "rosenbrock",
    "bounds": [-2.048, 2.048],
    "component_bounds": [0, 3900]
}
exp_config["NAME"] = f"bows_{exp_config['function']}_{exp_config['dimension']}"
exp_config["eta"] = int(np.sqrt(exp_config["dimension"]))
# exp_config["eta"] = 15
exp_config["ITER"] = exp_config["eta"] * 20
exp_config["n_point"] = exp_config["eta"] + 1
function = getattr(objectives, exp_config["function"])
print(exp_config)

support = np.arange(exp_config["component_bounds"][0], exp_config["component_bounds"][1] +
                    ((exp_config["component_bounds"][1] - exp_config["component_bounds"][0]) / exp_config["eta"]),
                    (exp_config["component_bounds"][1] - exp_config["component_bounds"][0]) / exp_config["eta"])

keys = ["X", "Y", "H", "H_hat", "times"]
res = {"config": exp_config}
utils.save_res(f"results/{exp_config['NAME']}.json", res)

# BO config
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
acquisition_constraint = histo_constraint(exp_config["eta"])
acquisition_bounds = torch.stack([torch.zeros(exp_config["eta"]), torch.ones(exp_config["eta"])]).double()

# Perform multiple trial
for trial in range(exp_config["TRIAL"]):
    print(f"\nTRIAL {trial}...")
    torch.manual_seed(trial)
    np.random.seed(trial)

    start, iter_times = time.time(), []
    # Initial training data
    train_X, train_Y, train_H = init_data(exp_config["n_point"], exp_config["dimension"], exp_config["bounds"],
                                          function, support, trial)
    H_hat = torch.Tensor()
    # Initialize the surrogate model
    gp, mll = init_model(train_H, train_Y, kernel)
    # Initialize MLP to map data back in the input space
    regr = regressor(train_H, train_X, max(exp_config["dimension"], train_X.shape[0]))
    iter_times.append(time.time() - start)
    # BO loop
    for it in range(exp_config["ITER"]):
        # Fit the surrogate model
        fit_gpytorch_mll(mll)  # , method='L-BFGS-B', options={'maxfun': 20000, 'maxiter': 20000})
        # Optimize the acquisition function and get the candidate
        new_H_hat = optimize_acquisition(gp, acquisition_bounds, [acquisition_constraint])
        # Map the histogram back in the input space and evaluate the new point
        new_X = predict(regr, new_H_hat, exp_config["bounds"])
        new_Y, new_H = new_observation(new_X, function, exp_config["eta"])
        # Update the dataset
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        train_H = torch.cat([train_H, new_H])
        H_hat = torch.cat([H_hat, new_H_hat])
        # Update the MLP regression model with the new observation
        # if it < exp_config["n_point"] * 3 or it % exp_config["n_point"] == 0:
        regr = regressor(train_H, train_X, max(exp_config["dimension"], train_X.shape[0]))
        # Update the surrogate model
        gp, mll = init_model(train_H, train_Y, kernel, gp.state_dict())

        iter_times.append(time.time() - start)

        # VERBOSE
        print(f"Iteration {it};\t Time = {iter_times[-1]: .2f};\t Best seen = {train_Y.min(): .2f};"
              f"\t Current value = {new_Y.numpy()[0, 0]:.2f};")
    res_trial = utils.res_dict(keys,
                               train_X.tolist(),
                               train_Y.flatten().tolist(),
                               train_H.tolist(), H_hat.tolist(),
                               iter_times)
    res["trial_" + str(trial)] = res_trial
    utils.save_res(f"results/{exp_config['NAME']}.json", res)

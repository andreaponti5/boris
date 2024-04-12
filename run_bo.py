import gpytorch
import numpy as np
import time
import torch

import objectives
import utils

from botorch import fit_gpytorch_mll

from bo import init_data, init_model, optimize_acquisition


# Experiment config
exp_config = {"TRIAL": 10,
              "dimension": 10,
              "function": "michalewicz",
              "bounds": [0, torch.pi],
              "ITER": 500}
exp_config["NAME"] = f"bo_{exp_config['function']}_{exp_config['dimension']}"
exp_config["n_point"] = int(np.sqrt(exp_config["dimension"])) + 1
# exp_config["n_point"] = 16
function = getattr(objectives, exp_config["function"])
print(exp_config)

keys = ["X", "Y", "times"]
res = {"config": exp_config}
utils.save_res(f"results/{exp_config['NAME']}.json", res)

kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
bounds = torch.stack([torch.full((exp_config["dimension"],), exp_config["bounds"][0]),
                      torch.full((exp_config["dimension"],), exp_config["bounds"][1])])
bounds = bounds.double()

# Loop over multiple runs
for trial in range(exp_config["TRIAL"]):
    print(f"\nTRIAL {trial}...")
    torch.manual_seed(trial)
    np.random.seed(trial)

    start, iter_times = time.time(), []
    # Initialize the design set
    train_X, train_Y, _ = init_data(exp_config["n_point"], exp_config["dimension"], exp_config["bounds"],
                                    function, seed=trial)
    # Initialize the surrogate model
    gp, mll = init_model(train_X, train_Y, kernel)
    iter_times.append(time.time() - start)
    # Loop over BO iterations
    for it in range(exp_config["ITER"]):
        # Fit the surrogate model
        fit_gpytorch_mll(mll, method='L-BFGS-B', options={'maxfun': 10000, 'maxiter': 10000})
        # Optimize the acquisition function and get the candidate
        new_X = optimize_acquisition(gp, bounds)
        new_Y, _ = function(new_X)
        new_Y = new_Y.reshape(-1, 1)
        # Add the new observation to the train set
        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        # Update the surrogate model
        gp, mll = init_model(train_X, train_Y, kernel, gp.state_dict())
        print(f"Iteration {it};\t Time = {iter_times[-1]: .2f};\t Best seen = {train_Y.min(): .8f};"
              f"\t Current value = {new_Y.numpy()[0, 0]:.2f};")
        iter_times.append(time.time() - start)
    # Save the results
    res_trial = utils.res_dict(keys,
                               train_X.tolist(),
                               train_Y.flatten().tolist(),
                               iter_times)
    res["trial_" + str(trial)] = res_trial
    # utils.save_res(f"results/{exp_config['NAME']}.json", res)

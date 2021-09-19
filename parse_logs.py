import numpy as np
import pandas as pd
import os
from datetime import datetime
import json


def get_test_name(model, name_params):
    """
    Create a name for this test
    :param model: dict: the test cfg
    :param name_params: list: the parameters we changed for this test
    :return: string: the name
    """
    return model["model"] + "_" + model["data"] + "_".join(name_params)

def get_run_name(cfg, name_params):
    """
    Create a name for this run
    :param model: dict: the run cfg
    :param name_params: list: the parameters we changed for this test
    :return: string: the name
    """
    optimizer = cfg["optimizer"]
    if optimizer == "newton":
        optimizer += "_" + cfg["globalization"]
        if optimizer == "newton_tr":
            optimizer += "_" + cfg["tr_type"]

    params = ""
    for p in name_params:
        if str(cfg.get(p, "")) != "":
            params += "_" + p + "_" + str(cfg.get(p, ""))

    return optimizer + params


def parse_runs(run_cfgs, freq, name_params = [], root_log_dir="logs/active"):
    """
    Parse the log files for a test and write them to a JSON file
    :param run_cfgs:
    :param freq:
    :param name_params:
    :return: string: filepath of the JSON output
    """
    dirs = sorted(os.listdir(root_log_dir))
    assert(len(run_cfgs) == len(dirs))

    runs_metric = [pd.read_csv(f"{root_log_dir}/{dir}/{freq}.out") for dir in dirs]

    results = {"name": get_test_name(run_cfgs[0], name_params),
               "vars": "TODO!",
               "runs": []}

    for cfg, metric in zip(run_cfgs, runs_metric):
        r = {"cfg": cfg,
             "name": get_run_name(cfg, name_params)}
        for header, col in metric.iteritems():
            r[header] = list(col)

        results["runs"].append(r)

    file_name = "logs/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + results["name"] + ".json"
    with open(file_name, 'w') as outfile:
        json.dump(results, outfile)

    return file_name

def interpolate(values, intervals, timestep=1):
    """
    Interpolate values to get a point at regular timesteps
    :param values: iterable: the input values
    :param intervals: iterable: the relative time for each value
    :param timestep: int: the regular interval to interpolate at
    :return: ndarray: values interpolated at regular timestep
    """
    count = 0
    interp_values = []
    for index, x in enumerate(intervals):
        while count < x:
            alpha = (count - intervals[index - 1]) / (x - intervals[index - 1])
            interp_values.append(alpha * values[index] + (1 - alpha) * values[index - 1])
            count += timestep
    return np.array(interp_values)


def average_seeds(seed_files):
    """
    Average the values across JSON outputs from different seeded test runs and write them to a JSON file
    :param seed_files: list: paths of JSON files from each test
    :return: string: filepath of the JSON output of averages
    """
    seed_results = []
    for file_name in seed_files:
        with open(file_name, 'r') as f:
            seed_data = json.load(f)
            seed_results.append(seed_data)

    # loss
    averages = {"name": seed_results[0]["name"],
               "runs": []}
    for run_example in seed_results[0]["runs"]:
        run_cfg = run_example["cfg"]
        runs = [run for seed in seed_results for run in seed["runs"] if run["cfg"] == run_cfg]

        loss = [run["test_loss"] for run in runs]
        sweeps = [interpolate(run["test_loss"], run["sweeps"]) for run in runs]
        min_sweeps = min(len(s) for s in sweeps)
        sweeps = [s[:min_sweeps] for s in sweeps]
        time = [interpolate(run["test_loss"], run["time"]) for run in runs]
        min_time = min(len(t) for t in time)
        time = [t[:min_time] for t in time]

        averages["runs"].append({"cfg": run_cfg,
                                 "name": run_example["name"],
                                 "iter": np.mean(loss, axis=0).tolist(),
                                 "sweeps": np.mean(sweeps, axis=0).tolist(),
                                 "time": np.mean(time, axis=0).tolist()})

    file_name = "logs/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + averages["name"] + "_average.json"
    with open(file_name, 'w') as outfile:
        json.dump(averages, outfile)

    return file_name

if __name__ == "__main__":
    parse_runs([{'model': 'dnn', 'data': 'mnist', 'loss': 'cross_entropy', 'hidden_layers': [100, 10], 'epochs': 1, 'optimizer': 'newton', 'lr': 0.1, 'globalization': 'tr', 'tr_type': 'subspace_2d', 'regularization': 0.1, 'lanczos': False, 'cg_tol': 'ew', 'subsamples': None, 'reduce_tr_reg': 0.9, 'precondition': False, 'hv_reuse': True, 'momentum': 0, 'momentum_type': None, 'cg_iters': 20}],
               "batch",
               root_log_dir="logs/active2/")

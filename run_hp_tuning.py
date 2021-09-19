import run
import random
import traceback
import tensorflow as tf
import plot_runs
import parse_logs
import os
import shutil

root_log_dir = "logs/active/"

model_data_loss = [{"model":"dnn", "data": "mnist", "loss": "cross_entropy", "hidden_layers": [100,10], "epochs": 10},
                   {"model":"autoencoder", "data": "mnist", "loss": "least_squares", "epochs": 10},
                   {"model": "autoencoder", "data": "cifar10", "loss": "least_squares", "epochs": 1},
                   {"model":"lenet5", "data": "mnist", "loss": "cross_entropy"}]
model = model_data_loss[2]

runs = 5
seeds = random.sample(range(1000), runs)

optimizer = [{"optimizer": "adam", "lr": lr, "batch_size": 1000} for lr in [1e-1, 1e-2, 1e-3, 1e-4]]# for bs in [32, 64, 128, 256]]#\
            #+ [{"optimizer": "sgd", "lr": lr, "batch_size": bs} for lr in [1e-1, 1e-2, 1e-3, 1e-4] for bs in [32, 64, 128, 256]]

test_params = ["lr", "batch_size"]

def safe_run(params, in_params=None, seed=None):
    """
    Run test and handle all exceptions
    :param params: dict: test parameters
    :param in_params: dict: inexact newton parameters, optional
    :param seed: int: random seed to allow reproducibility for runs, optional, None means random initialization
    """
    try:
        run.run_config(params, inexact_params=in_params, seed=seed, root_log_dir=root_log_dir)
    except Exception as e:
        print(type(e))
        print(e)
        print(traceback.format_exc())


def clear_dir(folder):
    """
    Remove all files and directories in a given directory
    :param folder: string: file path to the directory to empty
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == '__main__':
    print(tf.__version__)
    # Enable eager execution
    #tf.compat.v1.enable_eager_execution()
    try:
        os.mkdir(root_log_dir)
    except FileExistsError:
        pass

    seed_tests = []
    for seed in seeds:
        print("Rando Seed", seed)
        clear_dir(root_log_dir)
        run_cfgs = []
        for opt in optimizer:
            params = {}
            params.update(model)
            params.update(opt)

            safe_run(params, seed=seed)
            run_cfgs.append(params)

        seed_tests.append(parse_logs.parse_runs(run_cfgs, "batch", test_params, root_log_dir))

    plot_runs.plot_comparison(parse_logs.average_seeds(seed_tests), average=True)

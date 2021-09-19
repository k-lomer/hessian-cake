
import tensorflow as tf
from datasets.mnist import processed_mnist
from datasets.cifar10 import processed_cifar10
from datasets.imdb import processed_imdb
from datasets.linear import generate_linear_data
from models.keras_lenet import lenet5
from models.keras_dnn import dnn
from models.keras_lstm import lstm
from models.linear_regression import LinearRegression
from models.autoencoder import autoencoder
from train import train
from optimizers.sgd import StochasticGradientDescent
from optimizers.adam import Adam
from optimizers.inexact_newton import InexactNewton
from optimizers.inexact_newton_tr import InexactNewtonTR
from loss import NN_Loss
from logger import Logger


default_params = {"model": "dnn", "data": "mnist", "optimizer": "newton", "loss": "cross_entropy", "lr": 0.01,
                  "epochs": 1, "batch_size": 1000}
default_inexact_params = {"globalization": "tr", "tr_type": "subspace_2d", "regularization": 0.1, "update_reg": False,
                          "cg_tol": "ew", "subsamples": 100, "momentum": True, "lanczos": True}
default_inexact_verbosity = {"globalization": True, "cg": True, "tr_update": True, "grad_size": True,
                             "loss_breakdown": True, "forcing_term": True, "lanczos": True}


def get_param(params, key):
    return params.get(key, default_params[key])

def run_config(params, inexact_params=default_inexact_params, inexact_verbose=default_inexact_verbosity, seed=None, root_log_dir=None):
    if seed is not None:
        tf.random.set_seed(seed)

    if params["data"] == "mnist":
        if params["model"] == "lenet5":
            data = processed_mnist(image_dim=32)
        else:
            data = processed_mnist(image_dim=28)
        input_size = data[0][0][0].shape
    elif params["data"] == "cifar10":
        data = processed_cifar10()
        input_size = data[0][0][0].shape
    elif params["data"] == "imdb":
        data = processed_imdb()
        input_size = data[0][0][0].shape
    elif params["data"] == "linear":
        data = generate_linear_data(w=5, b=3)
    else:
        raise ValueError(f"unknown value {params['data']} for parameter data")
    # For autoencoders the output is the same as the input
    if params["model"] == "autoencoder":
        data = ((data[0][0], data[0][0]), (data[1][0], data[1][0]))
    print(f"\ndataset {params['data']},  train size {data[0][0].shape}, test size {data[1][0].shape}")

    if params["model"] == "lenet5":
        model = lenet5()
    elif params["model"] == "dnn":
        hidden_layers = params.get("hidden_layers", [32])
        model = dnn(input_shape=input_size, hidden_layer_dims=hidden_layers)
    elif params["model"] == "linear":
        model = LinearRegression()
    elif params["model"] == "autoencoder":
        if params["data"] == "mnist":
            model = autoencoder(input_size, n_filters=[4, 4], filter_sizes=[8, 4])
        elif params["data"] == "cifar10":
            model = autoencoder(input_size, n_filters=[4, 4, 4, 8], filter_sizes=[16, 8, 8, 4])
    elif params["model"] == "lstm":
        model = lstm(input_size)

    else:
        raise ValueError(f"unknown value {params['model']} for parameter model")

    if params["loss"] == "cross_entropy":
        loss = NN_Loss(model, lambda x, y: tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y))
    elif params["loss"] == "least_squares":
        loss = NN_Loss(model, tf.compat.v1.losses.mean_squared_error)
    elif params["loss"] == "binary_loss":
        loss = NN_Loss(model, tf.keras.losses.BinaryCrossentropy())
    else:
        raise ValueError("unknown value {params['loss']} for parameter loss")

    if params["optimizer"] == "sgd":
        optimizer = StochasticGradientDescent(params["lr"], model.trainable_variables, loss)
    elif params["optimizer"] == "adam":
        optimizer = Adam(params["lr"], model.trainable_variables, loss)
    elif params["optimizer"] == "newton":
        print(inexact_params)
        if inexact_params["globalization"] == "tr":
            optimizer = InexactNewtonTR(params["lr"], model.trainable_variables, loss, inexact_params["tr_type"],
                                      inexact_params,
                                      verbose=inexact_verbose)
        else:
            optimizer = InexactNewton(params["lr"], model.trainable_variables, loss,
                                      inexact_params,
                                      verbose=inexact_verbose)
    else:
        raise ValueError(f"unknown value {params['optimizer']} for parameter optimizer")

    log = Logger(root_logdir=root_log_dir, metrics=["CG Iter"])

    print(params)
    train(model, data, optimizer, loss, log, get_param(params, "epochs"), get_param(params, "batch_size"), accuracy=(params['loss']=="cross_entropy"))

if __name__ == '__main__':
    run_config(default_params, default_inexact_params, default_inexact_verbosity)





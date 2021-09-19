from optimizers.inexact_newton_tr import InexactNewtonTR
from datasets.mnist import processed_mnist
from models.autoencoder import autoencoder
from loss import NN_Loss
from operations import flatten,multi_dim_dot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def write_hessian(filepath, optimizer):
    with open(filepath, "w") as f:
        # create zero with correct dims

        for i in range(len(optimizer.variables)):
            v = [tf.zeros_like(v) for v in optimizer.variables]
            var_shape = v[i].shape
            vi_flat = tf.reshape(v[i], shape=[-1]).numpy()
            v[i] = tf.constant(vi_flat, shape=var_shape)
            for j in range(len(vi_flat)):
                vi_flat[j] = 1.
                v[i] = tf.constant(vi_flat, shape=var_shape)
                assert multi_dim_dot(v, v) == 1
                hessian_row = optimizer.hessian_v_prod(v)
                #hessian_row = optimizer.finite_diff_hessian_v_prod(v, 1e-3)
                f.write(",".join(map(str, flatten(hessian_row))))
                f.write("\n")
                vi_flat[j] = 0.

def plot_hessian(filename):
    h = np.genfromtxt(filename, delimiter=',')
    #h[-1][-1]=0
    # test symmetry
    sym_diff = abs(h - h.T)
    print("symmetry max diff", np.max(sym_diff))
    max_diff_index = np.unravel_index(np.argmax(sym_diff), h.shape)
    print("symmetry rel diff", sym_diff[max_diff_index] / abs(h[max_diff_index]))
    lim = np.max(abs(h))
    plt.imshow(h, vmin=-lim, vmax=lim, cmap='seismic', interpolation='nearest')
    plt.colorbar()
    plt.savefig(filename + ".png")
    #plt.show()
    plt.clf()




if __name__ == "__main__":
    filename = "hessian_reg_0.1.out"
    if not os.path.isfile(filename):
        data = processed_mnist(image_dim=28)
        data = ((data[0][0], data[0][0]), (data[1][0], data[1][0]))
        model = autoencoder(data[0][0][0].shape, n_filters=[4, 4], filter_sizes=[8, 4])
        loss = NN_Loss(model, tf.compat.v1.losses.mean_squared_error)
        optimizer = InexactNewtonTR(0.1, model.trainable_variables, loss, "subspace_2d", {"hv_reuse": False, "regularization": 0.1})

        (x_train, y_train), (x_test, y_test) = data
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        batched_data = train_data.shuffle(buffer_size=1000 * 4).batch(1000)
        i = 0
        for epoch in range(5):
            for batch_x, batch_y in batched_data:
                loss.update_data(batch_x, batch_y)
                if i % 100 == 0:
                    write_hessian(str(i)+"_"+filename, optimizer)
                    plot_hessian(str(i)+"_"+filename)
                    print("bias_value", optimizer.variables[-1])

                optimizer.minimize()
                i += 1


    plot_hessian(filename)


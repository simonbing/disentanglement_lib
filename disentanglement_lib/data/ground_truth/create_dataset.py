"""
Script to create custom datasets from disentanglement_lib.

Simon Bing
ETHZ 2020
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import dsprites
import norb
import cars3d
import shapes3d

from absl import app
from absl import flags
import time

FLAGS = flags.FLAGS

flags.DEFINE_enum('data_type', 'dsprites', ['dsprites','smallnorb','cars3d','shapes3d'], 'Type of data set to create.')
flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_bool('save_data', False, 'Save data set and ground truth factors')
flags.DEFINE_bool('debug', False, 'Debugging plots')
flags.DEFINE_bool('univ_rescaling', False, 'Rescale all GP sampled factors with same factor, or individually')
flags.DEFINE_bool('resample_period', False, 'Randomly resample frequencies across dimensions')
flags.DEFINE_integer('rand_static_factors', 0, 'Number of ground truth factors to hold constant per timeseries')
flags.DEFINE_integer('num_timeseries', 100, 'Total number of time series to generate')
flags.DEFINE_list('periods', [0,0,0,5,10,20], 'Periods for latent dimension time series')
flags.DEFINE_integer('length', 100, 'Time steps per time series')
flags.DEFINE_enum('kernel', 'sinusoid', ['sinusoid', 'rbf'], 'Underlying dynamics of factors')
flags.DEFINE_float('gp_weight', 1.0, 'Weight for dynanic part of GP kernel')
flags.DEFINE_string('file_name', 'dsprites', 'Name for features')
flags.DEFINE_string('factors_name', 'factors_dsprites', 'Name for factors')

def sample_inits(N): # TODO: this is where static factor matching happens
    """
    Samples initial values of timeseries from state space of latent factors.

    Args:
        N:  Number of samples to take. (N_max = 737280)
    """
    np.random.seed(FLAGS.seed)

    inits = np.zeros([N, 6], dtype=int)
    # Sample from entire latent space
    N_max = 737280
    rand_idxs = np.random.choice(N_max, N, replace=False)
    inits_idxs = np.unravel_index(rand_idxs, (1,3,6,40,32,32))

    # Choose inits from first N in latent space. Uncomment to apply.
    # inits_idxs = np.unravel_index(range(N), (1,3,6,40,32,32))

    for i in range(6):
        inits[:,i] = inits_idxs[i]

    # Additionally sample scale and shape. Uncomment to apply.
    # shapes = np.random.choice((0,1,2), N)
    # scales = np.random.choice((0,1,2,3,4,5), N)
    # inits[:,1] = shapes
    # inits[:,2] = scales

    # Hardcoded: remove shape, scale variation. Uncomment to apply.
    # inits[:,0:3] = 0

    return inits


def sample_factors(inits, periods, length):
    """
    Creates latent factor tensor to sample from.

    Args:
        inits:    [M] initial values of latent factors
        periods:  [M] periods of latent factors
        length: [N] length of resulting tensor
    Output:
        factors:[NxM] tensor of latent factors
    """
    N = inits.shape[0]
    factors = np.zeros([inits.shape[0], length, periods.size], dtype=int)
    if FLAGS.data_type == "dsprites":
        amplitudes = [0, 2, 5, 39, 31, 31] # Hardcoded for DSprites for now
    elif FLAGS.data_type == "smallnorb":
        amplitudes = [4, 8, 17, 5]
    elif FLAGS.data_type == "cars3d":
        amplitudes = [3, 23, 182]
    elif FLAGS.data_type == "shapes3d":
        amplitudes = [9, 9, 9, 7, 3, 14]
    xaxis = np.arange(0,length,1)


    if FLAGS.kernel == 'sinusoid':
        for i in range(N):
            for j in range(0,periods.size):
                if amplitudes[j]:
                    c = np.arccos(1 - 2*inits[i,j]/amplitudes[j])
                else:
                    c = 0

                if FLAGS.resample_period:
                    if periods[j]:
                        # Randomly sample period from list
                        period = np.random.choice(periods[np.nonzero(periods)])
                    else:
                        period = 0
                else:
                    period = periods[j]

                factors[i,:,j] = np.rint(-0.5*amplitudes[j] *
                                         np.cos(period * xaxis * 2*np.pi/length + c)
                                         + 0.5*amplitudes[j])

    elif FLAGS.kernel == 'rbf':
        for j, period in enumerate(periods): # Using period as proxy for length scale
            if period: # If we actually have dynamics
                kernel = FLAGS.gp_weight * RBF(length_scale=period) + ConstantKernel(5.0)
                gp = GaussianProcessRegressor(kernel=kernel)
                y = gp.sample_y(xaxis[:, np.newaxis], N,
                                random_state=np.random.randint(1e6))
                _, y_std = gp.predict([[1]], return_std=True)
                #######
                max_vals = np.amax(y, axis=0)
                min_vals = np.amin(y, axis=0)
                diffs = max_vals - min_vals

                # plt.plot(y[:,:100])
                # plt.show()

                # Discard variables outside of std_dev range
                # y_bound = y[:, np.all(abs(y) <= 1.8 * y_std, axis = 0)]

                if FLAGS.univ_rescaling:  # Rescale all samples the same
                    max_val = np.amax(y)
                    min_val = np.amin(y)
                    diff = max_val - min_val
                    if amplitudes[j]:
                        c = amplitudes[j] / diff
                    else:
                        c = 0.0
                    y_rescale = np.rint(c * (y - min_val))
                else:  # Rescale each sample itself
                    max_vals = np.amax(y, axis=0)
                    min_vals = np.amin(y, axis=0)
                    diffs = max_vals - min_vals
                    # if amplitudes[j]:
                    c = amplitudes[j] / diffs
                    # else:
                    # c = np.zeros_like(diffs)
                    c = np.tile(c, (length, 1))
                    min_vals = np.tile(min_vals, (length, 1))
                    y_rescale = np.rint(c * (y - min_vals))
            else: # Static factors
                if amplitudes[j] == 0:
                    y_init = np.zeros([N], dtype=int)
                else:
                    y_init = np.random.choice(amplitudes[j]+1, N)
                y_rescale = np.tile(y_init, (length,1))

            factors[:, :, j] = np.transpose(y_rescale)

            # y_rescale = y_rescale[:, np.all(abs(y) <= 1.5 * y_std, axis = 0)]

            # plt.plot(y_rescale[:,:], "black", alpha=0.01)
            # plt.plot(y_bound[:,:], "black", alpha=0.01)
            # plt.show()
    else:
        raise ValueError("Kernel must be one of ['sinusoid', 'rbf']")

    if FLAGS.rand_static_factors:
        assert FLAGS.rand_static_factors <= periods.size, 'rand_static_factors must be <= num ground truth factors'

        const_idxs = np.asarray([np.random.choice(np.nonzero(periods)[0],
                                       size=FLAGS.rand_static_factors,
                                       replace=False) for _ in np.arange(N)])
        for i in np.arange(const_idxs.shape[0]):
            for idx in const_idxs[i,:]:
                const_val = np.random.choice(amplitudes[idx]+1)
                factors[i,:,idx] = const_val

    return factors


def create_data(N, periods, length):
    """
    Samples underlying factors and generates data for GP-VAE model.

    Args:
        N:  number of samples
        periods:  [M] periods of latent factors
        length: [N] length of resulting tensor
    """
    if FLAGS.data_type == "dsprites":
        dsp = dsprites.DSprites()
    elif FLAGS.data_type == "smallnorb":
        snb = norb.SmallNORB()
    elif FLAGS.data_type == "cars3d":
        cars = cars3d.Cars3D()
    elif FLAGS.data_type == "shapes3d":
        shp = shapes3d.Shapes3D()

    random_state = np.random.RandomState(FLAGS.seed)

    inits = sample_inits(N)
    factors = sample_factors(inits, periods, length)
    if FLAGS.save_data: # Only allocate storage for data if we actually want to save it
        if FLAGS.data_type in ["dsprites", "smallnorb"]:
            dataset = np.zeros([N, length, 64*64])
        elif FLAGS.data_type in ["cars3d", "shapes3d"]:
            dataset = np.zeros([N, length, 64*64*3])

        start_time = time.time()
        for i in range(N):
            if FLAGS.data_type == "dsprites":
                data = np.squeeze(dsp.sample_observations_from_factors_no_color(
                                  factors=factors[i,:,:], random_state=random_state))
                data_reshape = data.reshape(data.shape[0], 64 * 64)
            elif FLAGS.data_type == "smallnorb":
                data = np.squeeze(snb.sample_observations_from_factors(
                                  factors=factors[i,:,:], random_state=random_state))
                data_reshape = data.reshape(data.shape[0], 64 * 64)
            elif FLAGS.data_type == "cars3d":
                data = cars.sample_observations_from_factors(factors=factors[i,:,:],
                                                             random_state=random_state)
                data_reshape = data.reshape(data.shape[0], 64 * 64 * 3)
            elif FLAGS.data_type == "shapes3d":
                if not(i % 1000):
                    print(F'At step {N}, after {(time.time() - start_time)//60} minutes.')
                data = shp.sample_observations_from_factors(factors=factors[i,:,:],
                                                            random_state=random_state)
                data_reshape = data.reshape(data.shape[0], 64 * 64 * 3)
            dataset[i,:,:] = data_reshape

        return dataset.astype('float32'), factors

    else:
        return None, factors


def split_train_test(data, factors, ratio):
    """
    Splits dataset and factors into explicit train and test sets.
    """
    assert len(data) == len(factors)

    N = len(data)
    data_train = data[:int(ratio*N),:,:]
    data_test = data[int(ratio*N):,:,:]

    factors_train = factors[:int(ratio*N),:,:]
    factors_test = factors[int(ratio*N):,:,:]

    return data_train, data_test, factors_train, factors_test

def count_unique_factors(factors):
    """
    Counts number of unique factors of variation.
    """
    factors_shape = factors.shape
    f_all = np.reshape(factors,
                       (factors_shape[0] * factors_shape[1], factors_shape[2]))
    if FLAGS.data_type == "dsprites":
        amplitudes = (1, 3, 6, 40, 32, 32)
    elif FLAGS.data_type == "smallnorb":
        amplitudes = (5, 9, 18, 6)
    elif FLAGS.data_type == "cars3d":
        amplitudes = (4, 24, 183)
    elif FLAGS.data_type == "shapes3d":
        amplitudes = (10, 10, 10, 8, 4, 15)
    f_all_flat = np.ravel_multi_index(np.transpose(f_all.astype(int)),
                                      amplitudes, order='F')

    return np.shape(np.unique(f_all_flat))

def plot_factors_series(factors, num_samples, show_factors=[3,4,5]):
    """
    Plots time series of sampled factors.

    Args:
        factors: Input factors array.
        num_samples: Number of timeseries to plot.
        show_factors: Which factors to plot per timeseries.
    """
    if FLAGS.data_type == "dsprites":
        names = [('color', 'black'), ('shape', 'pink'), ('scale', 'orange'),
                 ('orientation', 'green'), ('x position', 'blue'), ('y position', 'red')]
    elif FLAGS.data_type == "smallnorb":
        names = [('category', 'black'), ('elevation', 'orange'),
                 ('azimuth', 'green'), ('lighting', 'blue')]
    elif FLAGS.data_type == "cars3d":
        names = [('elevation', 'orange'), ('azimuth', 'green'), ('object type', 'blue')]
    elif FLAGS.data_type == "shapes3d":
        names = [('floor color', 'black'), ('wall color', 'pink'), ('object color', 'orange'),
                 ('object size', 'green'), ('object type', 'blue'), ('azimuth', 'red')]

    N = factors.shape[0]

    np.random.seed(FLAGS.seed)
    idxs = np.random.choice(N, size=num_samples, replace=False)

    factors_plot = factors[idxs, ...]

    for i in range(num_samples):
        for j,factor in enumerate(show_factors):
            plt.subplot(len(show_factors), 1, j+1)
            plt.plot(factors_plot[i,:,factor], marker='x', color=names[factor][1])
            plt.xlabel(names[factor][0])
        plt.show()

def count_avg_step_size(factors):
    """
    Counts average step size per dimension.
    """
    factors_re = np.reshape(factors, (factors.shape[0]*factors.shape[1], factors.shape[2]))
    diff = abs(np.diff(factors_re, axis=0))
    mean = np.mean(diff, axis=0)
    median = np.median(diff, axis=0)

    print(F"Mean difference: {mean}")
    print(F"Median difference: {median}")

def main(argv):
    del argv  # Unused

    # periods = np.array([0, 0, 0, 5, 10, 20]) # Should be integer multiples of 0.5
    # periods = np.array([0, 0, 0, 1.0, 0.1, 0.1])
    periods = np.asarray(FLAGS.periods).astype(float)
    length = FLAGS.length

    data, all_factors = create_data(FLAGS.num_timeseries, periods, length)
    unique_factors = count_unique_factors(all_factors)
    print(F"Number of unique underlying factors: {unique_factors}")

    if FLAGS.debug:
        count_avg_step_size(all_factors)
        plot_factors_series(all_factors, num_samples=3, show_factors=[0,1,2,3,5])

    if FLAGS.save_data:
        data_train, data_test, factors_train, factors_test = split_train_test(
            data, np.transpose(all_factors, (0,2,1)), 100 / 105)
        filename_data = FLAGS.file_name
        np.savez(filename_data, x_train_full=data_train, x_train_miss=data_train,
                 m_train_miss=np.zeros_like(data_train), x_test_full=data_test,
                 x_test_miss=data_test, m_test_miss=np.zeros_like(data_test))
        filename_factors = FLAGS.factors_name
        np.savez(filename_factors, factors_train=factors_train,
                 factors_test=factors_test)


if __name__ == '__main__':
    app.run(main)

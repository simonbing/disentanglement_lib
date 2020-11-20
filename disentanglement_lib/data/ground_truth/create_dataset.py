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

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_bool('save_data', False, 'Save data set and ground truth factors')
flags.DEFINE_bool('debug', False, 'Debugging plots')
flags.DEFINE_bool('univ_rescaling', False, 'Rescale all GP sampled factors with same factor, or individually')
flags.DEFINE_bool('resample_period', False, 'Randomly resample frequencies across dimensions')
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
    amplitudes = [0, 2, 5, 39, 31, 31] # Hardcoded for DSprites for now
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
            if not amplitudes[j]:
                kernel = ConstantKernel(0.0)
            else:
                if not period:
                    kernel = ConstantKernel(1.0)
                else:
                    kernel = FLAGS.gp_weight * RBF(length_scale=period) + ConstantKernel(5.0)
            gp = GaussianProcessRegressor(kernel=kernel)
            y = gp.sample_y(xaxis[:, np.newaxis], N, random_state=np.random.randint(1e6))

            if not period or FLAGS.univ_rescaling: # Rescale all samples the same
                max_val = np.amax(y)
                min_val = np.amin(y)
                diff = max_val - min_val
                if amplitudes[j]:
                    c = amplitudes[j] / diff
                else:
                    c = 0.0
                y_rescale = np.rint(c * (y - min_val))
            else: # Rescale each sample itself
                max_vals = np.amax(y, axis=0)
                min_vals = np.amin(y, axis=0)
                diffs = max_vals - min_vals
                # if amplitudes[j]:
                c = amplitudes[j] / diffs
                # else:
                # c = np.zeros_like(diffs)
                c = np.tile(c, (length,1))
                min_vals = np.tile(min_vals, (length,1))
                y_rescale = np.rint(c * (y - min_vals))

            factors[:,:,j] = np.transpose(y_rescale)
            # plt.plot(y_rescale[:,:10])
            # plt.show()
    else:
        raise ValueError("Kernel must be one of ['sinusoid', 'rbf']")

    return factors


def create_data(N, periods, length):
    """
    Samples underlying factors and generates data for GP-VAE model.

    Args:
        N:  number of samples
        periods:  [M] periods of latent factors
        length: [N] length of resulting tensor
    """
    dsp = dsprites.DSprites()

    random_state = np.random.RandomState(FLAGS.seed)

    inits = sample_inits(N)
    factors = sample_factors(inits, periods, length)

    if FLAGS.save_data: # Only allocate storage for data if we actually want to save it
        dataset = np.zeros([N, length, 64*64])

        for i in range(N):
            data = np.squeeze(dsp.sample_observations_from_factors_no_color(
                              factors=factors[i,:,:], random_state=random_state))
            dataset[i,:,:] = data.reshape(data.shape[0], 64*64)

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
    f_all_flat = np.ravel_multi_index(np.transpose(f_all.astype(int)),
                                      (1, 3, 6, 40, 32, 32), order='F')

    return np.shape(np.unique(f_all_flat))

def plot_factors_series(factors, num_samples, show_factors=[3,4,5]):
    """
    Plots time series of sampled factors.

    Args:
        factors: Input factors array.
        num_samples: Number of timeseries to plot.
        show_factors: Which factors to plot per timeseries.
    """
    names = [('color', 'black'), ('shape', 'pink'), ('scale', 'yellow'),
             ('orientation', 'green'), ('x position', 'blue'), ('y position', 'red')]

    N = factors.shape[0]

    np.random.seed(FLAGS.seed)
    idxs = np.random.choice(N, size=num_samples, replace=False)

    factors_plot = factors[idxs, ...]

    for i in range(num_samples):
        for j,factor in enumerate(show_factors):
            plt.subplot(len(show_factors), 1, j+1)
            plt.plot(factors_plot[i,:10,factor], marker='x', color=names[factor][1])
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
        # plot_factors_series(all_factors, num_samples=3)

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

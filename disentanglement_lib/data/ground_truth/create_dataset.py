"""
Script to create custom datasets from disentanglement_lib.

Simon Bing
ETHZ 2020
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import dsprites

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 42, 'Random seed')
flags.DEFINE_bool('save_data', False, 'Save data set and ground truth factors')
flags.DEFINE_integer('num_samples', 5000, 'Total number of samples to generate')

def sample_inits(N):
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

    factors = np.zeros([inits.shape[0], length, periods.size], dtype=int)
    amplitudes = [0, 2, 5, 39, 31, 31] # Hardcoded for DSprites for now
    xaxis = np.arange(0,length,1)

    for j in range(0,periods.size):
        if amplitudes[j]:
            c = np.arccos(1 - 2*inits[:,j]/amplitudes[j])
        else:
            c = np.zeros(inits.shape[0])

        factors[:,:,j] = np.rint(-0.5*amplitudes[j] * np.cos(
                       np.tile(periods[j] * xaxis * 2*np.pi/length, (inits.shape[0],1))
                       + np.transpose(np.tile(c, (length,1)))) + 0.5*amplitudes[j])

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


def data_from_factors(factors): # TODO: UNUSED?
    """
    Creates data dataset from array of latent features.

    Args:
        factors:    [N, length, n_factors] nparray

    Returns:
        data:      [N, length, 64*64] nparray
    """
    dsp = dsprites.DSprites()
    random_seed = 42
    random_state = np.random.RandomState(random_seed)

    N = factors.shape[0]
    length = factors.shape[2]
    print(length)

    data = np.zeros([N, length, 64*64])

    for i in range(N):
        factors_single = factors[i,:,:].transpose()
        sample_single = np.squeeze(dsp.sample_observations_from_factors_no_color(factors=factors_single, random_state=random_state))
        sample_single = sample_single.reshape(sample_single.shape[0], 64*64)
        data[i,:,:] = sample_single

    return data.astype('float32')

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

def main(argv):
    del argv  # Unused

    periods = np.array([0, 0, 0, 0.5, 1, 2]) # Should be integer multiples of 0.5
    length = 10 # Hardcoded for now

    data, all_factors = create_data(FLAGS.num_samples, periods, length) # TODO: split this so we dont always create the data
    unique_factors = count_unique_factors(all_factors)
    print(F"Number of unique underlying factors: {unique_factors}")

    if FLAGS.save_data:
        data_train, data_test, factors_train, factors_test = split_train_test(
            data, np.transpose(all_factors, (0,2,1)), 100 / 105)
        filename_data = 'dsprites_100k_5k'
        np.savez('dsprites_100k_5k', x_train_full=data_train, x_train_miss=data_train,
                 m_train_miss=np.zeros_like(data_train), x_test_full=data_test,
                 x_test_miss=data_test, m_test_miss=np.zeros_like(data_test))
        filename_factors = 'factors_dsprites_100k_5k'
        np.savez('factors_100k_5k', factors_train=factors_train,
                 factors_test=factors_test)


if __name__ == '__main__':
    app.run(main)

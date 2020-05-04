"""
Script to create custom datasets from disentanglement_lib.

Simon Bing
ETHZ 2020
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import dsprites

# Global random seed
np.random.seed(42)

def create_factors(inits, periods, length):
    """
    Creates latent factor tensor to sample from.

    Args:
        inits:    [M] initial values of latent factors
        periods:  [M] periods of latent factors
        length: [N] length of resulting tensor
    Output:
        factors:[NxM] tensor of latent factors
    """

    factors = np.zeros([length, periods.size], dtype=int)
    amplitudes = [0, 2, 5, 39, 31, 31] # Hardcoded for DSprites for now
    xaxis = np.arange(0,length,1)

    for i in range(0,periods.size):
        if amplitudes[i]:
            c = np.arccos(1 - 2*inits[i]/amplitudes[i])
        else:
            c = 0
        factors[:,i] = np.rint(-0.5*amplitudes[i] * np.cos(periods[i] * xaxis * 2*np.pi/length + c) + 0.5*amplitudes[i])

    return factors

def sample_inits(N):
    """
    Samples initial values of timeseries from state space of latent factors.

    Args:
        N:  Number of samples to take. (N_max = 737280)
    """
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


def create_input(N, periods, length):
    """
    Generates N dimensional input for GP-VAE model.

    Args:
        N:  Dimensionality of input
        periods:  [M] periods of latent factors
        length: [N] length of resulting tensor
    """
    dsp = dsprites.DSprites()
    # Hardcoded for now
    random_seed = 42
    random_state = np.random.RandomState(random_seed)

    inits = sample_inits(N)

    input = np.zeros([N, length, 64*64])

    all_factors = np.empty([N,6,length])

    for i in range(N):
        factors = create_factors(inits[i,:], periods, length)
        # print('FACTORS SHAPE {}'.format(factors.shape))
        all_factors[i,:,:] = factors.transpose()
        dataset = np.squeeze(dsp.sample_observations_from_factors_no_color(factors=factors, random_state=random_state))
        # print('DATASET SHAPE {}'.format(dataset.shape))
        dataset = dataset.reshape(dataset.shape[0], 64*64)
        # print('DATASET RESHAPE SHAPE {}'.format(dataset.shape))
        input[i,:,:] = dataset

    print(all_factors.shape)

    return input.astype('float32'), all_factors

def input_from_factors(factors):
    """
    Creates input dataset from array of latent features.

    Args:
        factors:    [N, length, n_factors] nparray

    Returns:
        input:      [N, length, 64*64] nparray
    """
    dsp = dsprites.DSprites()
    random_seed = 42
    random_state = np.random.RandomState(random_seed)

    N = factors.shape[0]
    length = factors.shape[2]
    print(length)

    input = np.zeros([N, length, 64*64])

    for i in range(N):
        factors_single = factors[i,:,:].transpose()
        # print('FACTORS SHAPE {}'.format(factors_single.shape))
        sample_single = np.squeeze(dsp.sample_observations_from_factors_no_color(factors=factors_single, random_state=random_state))
        sample_single = sample_single.reshape(sample_single.shape[0], 64*64)
        input[i,:,:] = sample_single

    return input.astype('float32')

def split_train_test(input, factors, ratio):
    """
    Splits fataset and factors into explicit train and test sets.
    """
    assert len(input) == len(factors)

    N = len(input)
    input_train = input[:int(ratio*N),:,:]
    input_test = input[int(ratio*N):,:,:]

    factors_train = factors[:int(ratio*N),:,:]
    factors_test = factors[int(ratio*N):,:,:]

    return input_train, input_test, factors_train, factors_test

def main():

    periods = np.array([0, 0, 0, 0.5, 1, 2]) # Should be integer multiples of 0.5
    length = 10

    # factors_gp_path = 'factors_gp_5000.npy'
    # factors_gp = np.load(factors_gp_path)
    # factors_gp_full_init_path = 'factors_gp_full_init_5000.npy'
    # factors_gp_full_init = np.load(factors_gp_full_init_path)

    input, all_factors = create_input(105000, periods, length)

    input_train, input_test, factors_train, factors_test = split_train_test(input, all_factors, 100/105)

    # input = input_from_factors(factors_gp_full_init)
    print("Train set shape: ", input_train.shape)
    print("Test set shape: ", input_test.shape)

    # input = input.astype('float32')

    save_input = False
    save_factors = True

    if save_input:
        filename_input = 'dsprites_100k_5k'
        np.savez('dsprites_100k_5k', x_train_full=input_train, x_train_miss=input_train,
                 m_train_miss=np.zeros_like(input_train), x_test_full=input_test,
                 x_test_miss=input_test, m_test_miss=np.zeros_like(input_test))
    if save_factors:
        filename_factors = 'factors_dsprites_100k_5k'
        np.savez('factors_100k_5k', factors_train=factors_train, factors_test=factors_test)
        # np.save(filename_factors, all_factors)


if __name__ == '__main__':
    main()

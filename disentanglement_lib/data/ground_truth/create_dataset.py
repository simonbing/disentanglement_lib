"""
Script to create custom datasets from disentanglement_lib.

Simon Bing
ETHZ 2020
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import dsprites

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

def sample_inits(N, flags):
    """
    Samples initial values of timeseries from state space of latent factors.

    Args:
        N:  Number of samples to take. (N_max = 737280)
        flags: List of strings of latent factors to sample
    """
    inits = np.zeros([N, 6], dtype=int)

    inits_idxs = np.unravel_index(range(N), (1,3,6,40,32,32))

    for i in range(6):
        inits[:,i] = inits_idxs[i]

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

    inits = sample_inits(N, flags=None)

    input = np.zeros([N, length, 64*64])

    all_factors = np.empty([N,6,length])

    for i in range(N):
        factors = create_factors(inits[i,:], periods, length)
        all_factors[i,:,:] = factors.transpose()
        dataset = np.squeeze(dsp.sample_observations_from_factors_no_color(factors=factors, random_state=random_state))
        dataset = dataset.reshape(dataset.shape[0], 64*64)
        input[i,:,:] = dataset

    print(all_factors.shape)

    return input, all_factors

def main():

    periods = np.array([0, 0, 0, 0.5, 1, 2]) # Should be integer multiples of 0.5
    inits = np.array([0, 0, 0, 0, 0, 0]) # Make sure these are in range amplitudes
    length = 10

    input, all_factors = create_input(5000, periods, length)
    print("Dataset shape: ", input.shape)

    save_input = False
    save_factors = True

    if save_input:
        filename_input = 'dsprites_5000'
        np.savez(filename_input, x_train_full=input, x_train_miss=input, m_train_miss=np.zeros_like(input), x_test_full=[], x_test_miss=[], m_test_miss=[])
        # np.save(filename, dataset)
    if save_factors:
        filename_factors = 'factors_5000'
        np.save(filename_factors, all_factors)


if __name__ == '__main__':
    main()

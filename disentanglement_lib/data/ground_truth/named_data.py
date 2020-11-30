# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides named, gin configurable ground truth data sets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import cars3d
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.data.ground_truth import norb
from disentanglement_lib.data.ground_truth import shapes3d
import gin.tf
import os
import numpy as np


@gin.configurable("dataset")
def get_named_ground_truth_data(name):
  """Returns ground truth data set based on name.

  Args:
    name: String with the name of the dataset.

  Raises:
    ValueError: if an invalid data set name is provided.
  """

  if name == "dsprites_full":
    return dsprites.DSprites([1, 2, 3, 4, 5])
  elif name == "dsprites_custom":
    return dsprites.DSprites([3, 4, 5])
  elif name == "dsprites_noshape":
    return dsprites.DSprites([2, 3, 4, 5])
  elif name == "color_dsprites":
    return dsprites.ColorDSprites([1, 2, 3, 4, 5])
  elif name == "noisy_dsprites":
    return dsprites.NoisyDSprites([1, 2, 3, 4, 5])
  elif name == "scream_dsprites":
    return dsprites.ScreamDSprites([1, 2, 3, 4, 5])
  elif name == "smallnorb":
    return norb.SmallNORB()
  elif name == "cars3d":
    return cars3d.Cars3D()
  elif name == "mpi3d_toy":
    return mpi3d.MPI3D(mode="mpi3d_toy")
  elif name == "mpi3d_realistic":
    return mpi3d.MPI3D(mode="mpi3d_realistic")
  elif name == "mpi3d_real":
    return mpi3d.MPI3D(mode="mpi3d_real")
  elif name == "shapes3d":
    return shapes3d.Shapes3D()
  elif name == "dummy_data":
    return dummy_data.DummyData()
  else:
    raise ValueError("Invalid data set name.")

@gin.configurable("subset")
def get_named_subset(name):
    """
    Returns subset of latent factors to explicitly be considered.
    """
    if name == "":
        return None
    elif name == "sin":
        subset_path = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."),
                                   "dsprites","factors","factors_5000.npy")
        subset = np.load(subset_path)
        subset_shape = subset.shape
        subset = np.reshape(np.transpose(subset, (0,2,1)), (subset_shape[0]*subset_shape[2],subset_shape[1]))
        return subset
    elif name == "sin_order_ss":
        subset_path = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."),
                                   "dsprites","factors","factors_sin_order_ss_5000.npy")
        subset = np.load(subset_path)
        subset_shape = subset.shape
        subset = np.reshape(np.transpose(subset, (0,2,1)), (subset_shape[0]*subset_shape[2],subset_shape[1]))
        return subset
    elif name == "sin_rand":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_100k_5k.npz'
        # subset_path = '/Users/Simon/git/disentanglement_lib/disentanglement_lib/data/ground_truth/factors_100k_5k.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate((subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0,2,1)), (subset_shape[0]*subset_shape[2],subset_shape[1]))
        return subset_full
    elif name == "sin_rand_100":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_100k_5k_100.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate((subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0,2,1)), (subset_shape[0]*subset_shape[2],subset_shape[1]))
        return subset_full
    elif name == "gp_full_1":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range1.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_part_1":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range1.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_full_2":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range2.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_part_2":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range2.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_full_3":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range3.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_part_3":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range3.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_full_4":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range4.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_part_4":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range4.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_full_const_1":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range_const_1.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    elif name == "gp_part_const_1":
        subset_path = '/cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_part_range_const_1.npz'
        subset = np.load(subset_path)
        subset_full = np.concatenate(
            (subset['factors_train'], subset['factors_test']))
        subset_shape = subset_full.shape
        subset_full = np.reshape(np.transpose(subset_full, (0, 2, 1)), (
        subset_shape[0] * subset_shape[2], subset_shape[1]))
        return subset_full
    else:
        raise ValueError("Invalid subset name.")

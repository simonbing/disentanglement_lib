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

"""Shapes3D data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
import h5py


# SHAPES3D_PATH = os.path.join(
#     os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes",
#     "look-at-object-room_floor-hueXwall-hueXobj-"
#     "hueXobj-sizeXobj-shapeXview-azi.npz"
# )
SHAPES3D_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes",
    "3dshapes.h5"
)



class Shapes3D(ground_truth_data.GroundTruthData):
  """Shapes3D dataset.

  The data set was originally introduced in "Disentangling by Factorising".

  The ground-truth factors of variation are:
  0 - floor color (10 different values)
  1 - wall color (10 different values)
  2 - object color (10 different values)
  3 - object size (8 different values)
  4 - object type (4 different values)
  5 - azimuth (15 different values)
  """

  def __init__(self):
    with tf.gfile.GFile(SHAPES3D_PATH, "rb") as f:
      # Data was saved originally using python2, so we need to set the encoding.
      # data = np.load(f, encoding="latin1")
      data = h5py.File(f, 'r')

    images = data["images"]
    labels = data["labels"]
    n_samples = images.shape[0]
    # n_samples = np.prod(images.shape[0:6])
    # self.images = (
    #     images.reshape([n_samples, 64, 64, 3]).astype(np.float32) / 255.)
    # features = labels.reshape([n_samples, 6])
    self.images = images
    features = labels
    self.factor_sizes = [10, 10, 10, 8, 4, 15]
    self.latent_factor_indices = list(range(6))
    self.num_total_factors = features.shape[1]
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes

  @property
  def observation_shape(self):
    return [64, 64, 3]


  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_factor_pairs(self, num, random_state):
    """Sample paired observations for weakly supervised models"""
    if self.subset is None:
        k = 2 # Hardcoded TODO change this
        # Sample first factors and from those, first sample
        factors1, sample1 = self.sample(num, random_state)
        # Sample k, if k is random
        if k is None:
          k = np.random.choice(np.arange(1, len(self.factors_num_values)+1))
        # Sample k indices
        s = np.random.choice(np.arange(len(self.factors_num_values)), k, replace=False)
        # Resample factors at indices s
        factors2 = np.copy(factors1)
        for idx in s:
          new_factor = np.random.choice(np.arange(self.factors_num_values[idx]))
          factors2[0, idx] = new_factor
        # Resample with new factors
        sample2 = self.sample_observations_from_factors(factors2, random_state)
    else:
        init_idx = np.random.randint(0, self.subset.shape[0], 1)
        # Make sure that first index is even. Needed to ensure correct pairing.
        if init_idx % 2:
            init_idx = init_idx-1
        idxs = np.arange(init_idx, init_idx + num * 2)
        idxs_wrap = [np.arange(0, self.subset.shape[0])[idx % self.subset.shape[0]] for idx in idxs]
        factors = self.subset[idxs_wrap]
        factors1 = factors[0::2, :]
        factors2 = factors[1::2, :]
    return factors1, factors2

  def sample_observations_from_factors(self, factors, random_state):
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    ims = []
    for idx in indices:
      im = self.images[idx]
      im = np.asarray(im)
      ims.append(im)
    ims = np.stack(ims, axis=0)
    ims = ims / 255.  # normalise values to range [0,1]
    ims = ims.astype(np.float32)
    return ims.reshape([factors.shape[0], 64, 64, 3])

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

"""Abstract class for data sets that are two-step generative models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


class GroundTruthData(object):
  """Abstract class for data sets that are two-step generative models."""

  @property
  def num_factors(self):
    raise NotImplementedError()

  @property
  def factors_num_values(self):
    raise NotImplementedError()

  @property
  def observation_shape(self):
    raise NotImplementedError()

  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    raise NotImplementedError()

  def sample_observations_from_factors(self, factors, random_state):
    """Sample a batch of observations X given a batch of factors Y."""
    raise NotImplementedError()

  def sample(self, num, random_state):
    """Sample a batch of factors Y and observations X."""
    factors = self.sample_factors(num, random_state)
    return factors, self.sample_observations_from_factors(factors, random_state)

  def sample_observations(self, num, random_state):
    """Sample a batch of observations X."""
    return self.sample(num, random_state)[1]

  def sample_pair_observations(self, num, random_state, k):
    """Sample a batch of paired observations X with k shared factors"""
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
    return np.vstack((sample1, sample2))

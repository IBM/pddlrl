# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import chex

from rlax._src import base
from rlax._src import distributions

import jax.numpy as jnp

Array = chex.Array
Numeric = chex.Numeric


def _masked_mix_with_uniform(probs, mask, epsilon):
  """Mix an arbitrary categorical distribution with a uniform distribution."""
  num_actions = jnp.sum(mask, axis=-1)
  uniform_probs = mask / num_actions
  return (1 - epsilon) * probs + epsilon * uniform_probs


def masked_epsilon_greedy(epsilon):

    def sample_fn(key: Array, preferences: Array, mask: Array, epsilon=epsilon):
        probs = distributions._argmax_with_random_tie_breaking(preferences)
        probs = _masked_mix_with_uniform(probs, mask, epsilon)
        return distributions.categorical_sample(key, probs)

    def probs_fn(preferences: Array, mask: Array, epsilon=epsilon):
        probs = distributions._argmax_with_random_tie_breaking(preferences)
        return _masked_mix_with_uniform(probs, mask, epsilon)

    def logprob_fn(sample: Array, preferences: Array, mask: Array, epsilon=epsilon):
        probs = distributions._argmax_with_random_tie_breaking(preferences)
        probs = _masked_mix_with_uniform(probs, mask, epsilon)
        return base.batched_index(jnp.log(probs), sample)

    def entropy_fn(preferences: Array, mask: Array, epsilon=epsilon):
        probs = distributions._argmax_with_random_tie_breaking(preferences)
        probs = _masked_mix_with_uniform(probs, mask, epsilon)
        return -jnp.nansum(probs * jnp.log(probs), axis=-1)

    return distributions.DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn, None)

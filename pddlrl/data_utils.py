# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import collections
import os
import pickle

import numpy as np

import jax
import gin

from pddlrl import env_utils
from pddlrl.benchmarks import pddl_vi

raise NotImplementedError("This hasn't been ported to the new env api yet")



class Data(collections.namedtuple("Data", "states values")):
    __slots__ = ()

    @property
    def num_samples(self):
        return len(self.values)


class PolicyData(collections.namedtuple("PolicyData", "states actions")):
    __slots__ = ()

    @property
    def num_samples(self):
        return len(self.actions)


def generate_data(env, discount):
    states, values = generate_raw_data(env, discount)

    env.reset()
    all_obs_literals = env.observation_space.all_ground_literals(
        env.get_state(), valid_only=False)
    state_encoder = env_utils.StateEncoder(all_obs_literals)
    states = state_encoder(states)

    return Data(np.array(states), np.array(values))


def generate_raw_data(env, discount):
    env.reset()

    mdp = pddl_vi.PDDLMDP(env)
    values, _, _ = pddl_vi.deterministic_value_iteration(mdp, discount)

    return Data(*zip(*values.items()))


def generate_policy_data(env, discount):
    env.reset()

    mdp = pddl_vi.PDDLMDP(env)
    _, policy, _ = pddl_vi.deterministic_value_iteration(mdp, discount)

    return PolicyData(*zip(*policy.items()))


def export_data(filepath, data):
    filepath = os.path.expanduser(filepath)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


@gin.configurable
def load_data(filepath):
    filepath = os.path.expanduser(filepath)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def split_data(rng, data, ratio):
    num_samples = data.num_samples
    indices = rng.permutation(num_samples)

    split_idx = int(ratio*num_samples)
    train_idx, test_idx = np.split(indices, [split_idx])

    test_data = jax.tree_map(lambda x: x[train_idx], data)
    train_data = jax.tree_map(lambda x: x[test_idx], data)

    return train_data, test_data


@gin.configurable
def batched_epoch(rng, data, batch_size, drop_remainder=True):
    rng = np.random.default_rng(rng)
    num_samples = data.num_samples
    indices = rng.permutation(num_samples)

    for i in range(0, num_samples, batch_size):
        last_idx = i + batch_size
        if drop_remainder and last_idx > num_samples:
            continue

        batch_idx = indices[i:last_idx]
        yield jax.tree_map(lambda x: x[batch_idx], data)


@gin.configurable
def generate_batches(rng, data, batch_size, drop_remainder=True):
    rng = np.random.default_rng(rng)
    while True:
        yield from batched_epoch(rng, data, batch_size, drop_remainder)

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import functools
from typing import Callable, Optional

import acme
from acme import adders
from acme import types
from acme.jax import networks
from acme.jax import variable_utils

import dm_env

import haiku as hk

import jax
import jax.numpy as jnp

import numpy as np

from rlax._src import distributions

import pddlenv

from pddlrl import encoding

ValueNetwork = Callable[[types.NestedArray], networks.Value]


def _pad_to_nearest_power(array, power, axis):
    n = array.shape[axis]
    ndim = array.ndim
    exponent = np.ceil(np.log(n)/np.log(power))
    batch_size = max(n, int(power**exponent))

    pad_size = batch_size - n
    pad_width = ((0, 0),) * axis + ((0, pad_size),) + ((0, 0),) * (ndim - axis - 1)
    return jnp.pad(array, pad_width), pad_size


def with_padding(func, argnums=None, power=1.3):
    if isinstance(argnums, int):
        argnums = (argnums,)

    @functools.wraps(func)
    def _padded_function(*args):
        args = list(args)
        pad_size = None
        _argnums = argnums
        if _argnums is None:
            _argnums = range(len(args))

        for i in _argnums:
            leaves, treedef = jax.tree_flatten(args[i])
            padded_leaves, arg_pad_sizes = zip(*(
                _pad_to_nearest_power(x, power, axis=0) for x in leaves))
            args[i] = jax.tree_unflatten(treedef, padded_leaves)

            arg_pad_size = arg_pad_sizes[0]
            assert all(size == arg_pad_size for size in arg_pad_sizes)
            if pad_size is None:
                pad_size = arg_pad_size
            else:
                assert pad_size == arg_pad_size

        outputs = func(*args)
        if pad_size:
            outputs = outputs[:-pad_size]

        return outputs

    return _padded_function


class LookaheadValueFunction:
    #
    # Note: reward shaping is performed inside PDDLDynamics.
    # There is no explicit subtraction of hvalues from V or Q functions.
    #
    def __init__(self,
                 network: ValueNetwork,
                 dynamics: pddlenv.PDDLDynamics,
                 encoder: encoding.StateEncoder):
        self.network = hk.without_apply_rng(hk.transform(network, apply_rng=True))
        self._apply = jax.jit(self.network.apply)

        def qvalues_from_transitions(params, rewards, discounts, next_states):
            # evaluate value at next state then discount and add reward
            next_vals = self.network.apply(params, next_states)
            return rewards + discounts * next_vals

        self._qvalues_from_transitions = with_padding(
            jax.jit(qvalues_from_transitions), argnums=(1, 2, 3))

        self.dynamics = dynamics
        self.state_encoder = encoder

    def init(self, key, states):
        return self.network.init(key, states)

    def apply(self, params, states):
        return self.network.apply(params, states)

    def preprocess(self, states):
        return self.state_encoder.encode_states(states)

    def value(self, network_params, state):
        return self._apply(network_params, self.preprocess([state]))

    def qvalues_from_lookahead(self, network_params, state):
        # lookahead for next states for all valid actions
        actions, timesteps = self.dynamics.sample_transitions(state)
        _, rewards, discounts, next_states = zip(*timesteps)
        rewards = jnp.array(rewards)
        discounts = jnp.array(discounts)

        # encode state into boolean vector
        next_features = self.state_encoder.encode_states(next_states)
        next_features = jax.tree_map(jnp.array, next_features)

        qvals = self._qvalues_from_transitions(
            network_params,
            rewards,
            discounts,
            next_features,
        )

        return actions, qvals


class LookaheadActor(acme.Actor):

    def __init__(self,
                 rng: hk.PRNGSequence,
                 lookahead_network: LookaheadValueFunction,
                 action_distribution: distributions.DiscreteDistribution,
                 variable_client: variable_utils.VariableClient,
                 adder: Optional[adders.Adder] = None):

        self._lookahead = lookahead_network
        self._action_distribution = action_distribution
        self._rng = rng
        self._adder = adder
        self._client = variable_client

    def value(self, observation: types.NestedArray) -> types.NestedArray:
        return self._lookahead.value(self._client.params, observation)

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        key = next(self._rng)
        actions, qvals = self._lookahead.qvalues_from_lookahead(self._client.params, observation)
        a_idx = self._action_distribution.sample(key, qvals)
        return actions[a_idx]

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self):
        self._client.update()

    def update_and_wait(self):
        self._client.update_and_wait()

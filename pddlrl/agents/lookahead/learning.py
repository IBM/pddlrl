# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

from typing import Callable, Iterator, Iterable, List, NamedTuple
from pddlenv import EnvState, Heuristic

import acme
from acme import types
from acme.jax import networks
from acme.jax import utils
from dm_env import specs
import haiku as hk

import jax
import jax.numpy as jnp
import numpy as np
import optax

import rlax
from rlax._src import distributions

import pddlenv

from pddlrl.agents.lookahead import actors

ValueNetwork = Callable[[types.NestedArray], networks.Value]


class TrainingState(NamedTuple):
    """Holds the agent's training state."""
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    steps: int


class ApproximateValueIteration(acme.Learner, acme.Saveable):
    action_distribution: distributions.DiscreteDistribution
    target_update_period: int
    _state: TrainingState

    def __init__(self,
                 rng: hk.PRNGSequence,
                 obs_spec: specs.Array,
                 lookahead_network: actors.LookaheadValueFunction,
                 optimizer: optax.GradientTransformation,
                 iterator: Iterator,
                 action_distribution: distributions.DiscreteDistribution,
                 target_update_period: int,
                 huber_loss_parameter: float = 1.):
        self.action_distribution = action_distribution
        self.lookahead_network = lookahead_network
        self._iterator = iterator

        def loss(params: hk.Params, states, E_a_Q_s_a):
            td_error = E_a_Q_s_a - lookahead_network.apply(params, states)
            # i.e., E_a Q(a,s) - V(s)
            # or its reward-shaped variants, E_a Q'(a,s) - V'(s)

            # return jnp.mean(rlax.huber_loss(td_error, huber_loss_parameter))
            return jnp.mean(rlax.l2_loss(td_error))

        def sgd_step(state: TrainingState, samples, E_a_Q_s_a):
            # Compute loss gradients. Since loss takes the targets as inputs, the loss
            # won't backprop through it which results in standard TD learning loss.
            dloss = jax.grad(loss)(state.params, samples, E_a_Q_s_a)

            # compute updates from gradient and update the optimizer's state
            updates, new_opt_state = optimizer.update(dloss, state.opt_state)
            # apply updates to parameters
            new_params = optax.apply_updates(state.params, updates)

            # Periodically update target networks.
            steps = state.steps + 1
            target_params = rlax.periodic_update(
                new_params, state.target_params,
                steps, target_update_period, )

            # purely for logging purpose
            _values = lookahead_network.apply(state.params, samples)
            _loss = loss(state.params, samples, E_a_Q_s_a)

            return TrainingState(
                params=new_params,
                target_params=target_params,
                opt_state=new_opt_state,
                steps=steps,
            ), _loss, _values

        self._sgd_step = jax.jit(sgd_step)

        # Initialise parameters and optimiser state.
        initial_params = lookahead_network.init(
            next(rng), utils.add_batch_dim(utils.zeros_like(obs_spec)))
        initial_target_params = initial_params
        initial_opt_state = optimizer.init(initial_params)

        self._state = TrainingState(
            params=initial_params,
            target_params=initial_target_params,
            opt_state=initial_opt_state,
            steps=0)

        # HACK: for logging the loss during the training. See also: experiment_loop
        # Why every RL framework is so stateful??? Looping over steps, not returning values?
        self.batch_losses = []
        self.batch_samples = []
        self.batch_targets = []
        self.batch_values = []
        self.batch_heuristics = []

    def E_a_Q_s_a(self, network_params, states: Iterable[pddlenv.EnvState]):
        results = []
        for state in states:
            if state.goal_state():
                E_a_Q_s_a = 0.
            else:
                # compute qvalues using a lookahead for every action
                actions, Q_a_s = self.lookahead_network.qvalues_from_lookahead(network_params, state)
                # mask zero prob to filter -inf qvalues when given zero prob
                Pr_a = self.action_distribution.probs(Q_a_s)
                mask = Pr_a > 0.
                E_a_Q_s_a = jnp.dot(Q_a_s[mask], Pr_a[mask])
            results.append(E_a_Q_s_a)
        return results

    # note: is called by acme.Agent.agent.update()
    def step(self):
        samples = next(self._iterator)
        if len(samples) == 0:
            return
        # compute the targets using lookaheads
        E_a_Q_s_a = jnp.array(self.E_a_Q_s_a(self._state.target_params, samples))
        # preprocess : convert PDDLEnv states into tensors compatible with NLM
        NLM_input_tensor = self.lookahead_network.preprocess(samples)
        self._state, mean_loss, values = self._sgd_step(self._state, NLM_input_tensor, E_a_Q_s_a)

        heuristics = np.array([self.lookahead_network.dynamics.heuristic(state.literals, state.problem) for state in samples])

        # for logging the loss during the training. See also: experiment_loop
        self.batch_losses.append(np.array(mean_loss * len(samples)))
        self.batch_samples.append(len(samples))
        self.batch_targets.append(np.sum(E_a_Q_s_a))
        self.batch_values.append(np.sum(values))
        self.batch_heuristics.append(np.sum(heuristics))
        return

    def get_variables(self, names: List[str]) -> List[hk.Params]:
        return [self._state.params]

    def save(self) -> TrainingState:
        return self._state

    def restore(self, state: TrainingState):
        self._state = state

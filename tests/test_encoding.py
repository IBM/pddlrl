import collections
import os
import pytest

import numpy as np

import jax

import pddlenv
from pddlenv import generators
from pddlenv.generators import literals
from pddlenv.generators import problems

from pddlrl import encoding


@pytest.fixture
def pddlenv_states(pytestconfig, monkeypatch):
    pddl_dir = os.path.join(pytestconfig.rootdir, "tests", "pddl")
    monkeypatch.setenv("PDDL_ROOT_DIR", pddl_dir)

    literals_sampler = literals.blocks.ClearSampler()
    problem_sampler = problems.blocks.SingleTowerSampler(2, 5, False)
    state_initializer = generators.as_state_initializer(3, problem_sampler, literals_sampler)
    environment = pddlenv.PDDLEnv(state_initializer, pddlenv.PDDLDynamics())

    states = collections.defaultdict(list)
    for _ in range(1000):
        state = environment.reset().observation
        states[state.problem.objects].append(state)

    return states


def nlm_state_encoder(states):
    literals = [state.literals for state in states]
    goals = [state.problem.goal for state in states]

    # this assumes all states are from the same problem
    problem = states[0].problem
    features = pddlenv.array.to_dense_binary(literals, problem)

    # return the features concatenated with the goal predicates
    goal_features = pddlenv.array.to_dense_binary(goals, problem)
    return tuple(
        np.concatenate(xs, axis=-1)
        for xs in zip(features.values(), goal_features.values())
    )


def test_nlm_state_encoder(pddlenv_states):
    encoder = encoding.NLMStateEncoder()

    for objects, states in pddlenv_states.items():
        encoded_states = encoder.encode_states(states)
        old_encoded_states = nlm_state_encoder(states)

        assert jax.tree_structure(encoded_states) == jax.tree_structure(old_encoded_states)

        for x, y in zip(encoded_states, old_encoded_states):
            np.testing.assert_array_equal(x, y)

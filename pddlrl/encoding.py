# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import dataclasses
import itertools
from typing import Tuple, Dict, Sequence, Type, Protocol

import numpy as np

import jax.numpy as jnp

from dm_env import specs
from acme.types import NestedSpec

import pddlenv

LiteralIndices = Dict[int, np.ndarray]
LiteralShapes = Dict[int, Tuple[int, ...]]


class StateEncoder(Protocol):

    def encode_states(self, states: Sequence[pddlenv.EnvState]) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    def encode_state(self, state: pddlenv.EnvState) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError

    def spec(self, problem: pddlenv.Problem) -> NestedSpec:
        raise NotImplementedError


def state_and_goal_indices(states):
    literals = [state.literals for state in states]
    goals = [state.problem.goal for state in states]

    # this assumes all states are from problems with the same objects and predicates
    problem = states[0].problem

    n = len(literals)
    indices, shapes = pddlenv.array.compute_indices(
        literals, problem.objectmap.objects, problem.predicates)
    goal_indices, goal_shapes = pddlenv.array.compute_indices(
        goals, problem.objectmap.objects, problem.predicates)
    assert goal_shapes == shapes

    def _combine_indices(idx, goal_idx, shape):
        if goal_idx is not None:
            goal_idx = goal_idx[:-1] + (np.add(goal_idx[-1], shape[-1]),)
        else:
            return idx
        if idx is None:
            return goal_idx
        merged_idx = [np.concatenate((i, j)) for i, j in zip(idx, goal_idx)]
        return tuple(merged_idx)

    indices = {
        k: _combine_indices(indices.get(k), goal_indices.get(k), shapes[k])
        for k in (indices.keys() | goal_indices.keys())
    }
    shapes = {
        k: (n,) + shape[:-1] + (shape[-1] * 2,)
        for k, shape in shapes.items()
    }

    return indices, shapes


@dataclasses.dataclass
class NLMStateEncoder:
    dtype: Type = np.float32

    def shapes(self, problem: pddlenv.Problem) -> LiteralShapes:
        num_objects = len(problem.objects)
        predicates = problem.predicates
        shapes = pddlenv.array.compute_shapes(num_objects, predicates)

        # Double number of predicates to account for the goal predicates.
        return {
            arity: shape[:-1] + (shape[-1] * 2,)
            for arity, shape in shapes.items()
        }

    def indices(self, states: Sequence[pddlenv.EnvState]
                ) -> Tuple[LiteralIndices, LiteralShapes]:
        return state_and_goal_indices(states)

    def encode_states(self, states: Sequence[pddlenv.EnvState]) -> Tuple[np.ndarray, ...]:
        indices, shapes = self.indices(states)

        def _create_array(idx, shape):
            x = np.zeros(shape, self.dtype)
            if idx is not None:
                x[idx] = 1
            return x

        return tuple(
            _create_array(indices.get(arity), shape)
            for arity, shape in shapes.items()
        )

    def encode_state(self, state: pddlenv.EnvState) -> Tuple[np.ndarray, ...]:
        return tuple(x[0] for x in self.encode_states([state]))

    def spec(self, problem: pddlenv.Problem):
        return tuple(
            specs.BoundedArray(shape, self.dtype, 0, 1, f"{arity}-arity-predicates")
            for arity, shape in self.shapes(problem).items()
        )


def _flat_valid_action_indices(state):
    problem = state.problem
    actions = problem.grounded_actions
    valid_actions = [[a for a in actions if a.applicable(state.literals)]]
    return pddlenv.array.ravel_literal_indices(
        *pddlenv.array.compute_indices(valid_actions, problem.objects, problem.actions))[1]


@dataclasses.dataclass
class ValidActionEncoder:
    dtype: Type = np.float32

    def shape(self, problem: pddlenv.Problem) -> Tuple[int]:
        action_shapes = pddlenv.array.compute_shapes(
            len(problem.objects),
            problem.actions,
        )
        num_actions = sum(np.prod(shape) for shape in action_shapes.values())
        return num_actions,

    def indices(self, states: Sequence[pddlenv.EnvState]) -> Tuple[np.ndarray, ...]:
        indices = (_flat_valid_action_indices(s) for s in states)
        indices = zip(*(
            (np.full_like(idx, i), idx)
            for i, idx in enumerate(indices)
        ))
        return tuple(np.concatenate(idx) for idx in indices)

    def action_mask(self, states: Sequence[pddlenv.EnvState]) -> np.ndarray:
        indices = self.indices(states)
        # we assume every state shares the same problem instance
        shape = (indices[0].shape[0],) + self.shape(states[0].problem)
        action_mask = np.zeros(shape, dtype=self.dtype)
        action_mask[indices] = 1
        return action_mask

    def spec(self, problem: pddlenv.Problem):
        return specs.BoundedArray(
            shape=self.shape(problem),
            minimum=0,
            maximum=1,
            dtype=self.dtype,
            name="valid_action_indices",
        )


@dataclasses.dataclass
class StateAndValidActionEncoder:
    state_encoder: NLMStateEncoder
    action_encoder: ValidActionEncoder

    def shapes(self, problem: pddlenv.Problem) -> Tuple[LiteralShapes, Tuple]:
        state_shapes = self.state_encoder.shapes(problem)
        action_shape = self.action_encoder.shape(problem)
        return state_shapes, action_shape

    def encode_states(self, states: Sequence[pddlenv.EnvState]) -> Tuple[np.ndarray, ...]:
        state_features = self.state_encoder.encode_states(states)
        action_masks = self.action_encoder.action_mask(states)
        return state_features + (action_masks,)

    def encode_state(self, state: pddlenv.EnvState) -> Tuple[np.ndarray, ...]:
        state_features = self.state_encoder.encode_state(state)
        action_masks = self.action_encoder.action_mask([state])[0]
        return state_features + (action_masks,)

    def spec(self, problem: pddlenv.Problem):
        return self.state_encoder.spec(problem) + (self.action_encoder.spec(problem),)

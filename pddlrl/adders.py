# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

from typing import Any, Mapping, Tuple, Optional, Callable

import dm_env

from acme import types
from acme.adders import base

import pddlenv

import numpy as np


class StateAdder(base.Adder):

    def __init__(self, buffer):
        self.buffer = buffer

    def add_first(self, timestep: dm_env.TimeStep):
        self.buffer.put(timestep.observation)

    def add(self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
            extras: types.NestedArray = ()):
        self.buffer.put(next_timestep.observation)

    def reset(self):
        pass


class MultiObjectSetStateAdder(base.Adder):
    adders: Mapping[Tuple[pddlenv.PDDLObject, ...], base.Adder]

    def __init__(self, adders: Mapping[Tuple[pddlenv.PDDLObject, ...], base.Adder]):
        self.adders = adders

    def _preprocess(self, timestep: dm_env.TimeStep):
        return timestep

    def add_first(self, timestep: dm_env.TimeStep):
        self.adders[timestep.observation.problem.objects].add_first(
            self._preprocess(timestep))

    def add(self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
            extras: types.NestedArray = ()):
        self.adders[next_timestep.observation.problem.objects].add(
            action, self._preprocess(next_timestep), extras)

        if next_timestep.last():
            self.reset()

    def reset(self):
        for adder in self.adders.values():
            adder.reset()


class PDDLIndexAdder(MultiObjectSetStateAdder):

    def _preprocess(self, timestep):
        state = timestep.observation
        problem = state.problem
        state_idx = pddlenv.array.compute_indices(
            state.literals, problem.objects, problem.predicates)[0]
        goal_idx = pddlenv.array.compute_indices(
            problem.goal, problem.objects, problem.predicates)[0]
        return timestep._replace(
            observation=(state_idx, goal_idx),
            reward=np.array(timestep.reward, np.float32),
            discount=np.array(timestep.discount, np.float32),
        )


class PDDLAdder(MultiObjectSetStateAdder):
    adders: Mapping[Tuple[pddlenv.PDDLObject, ...], base.Adder]
    state_encoder: Optional[Callable[[pddlenv.EnvState], Any]]

    def __init__(self,
                 adders: Mapping[Tuple[pddlenv.PDDLObject, ...], base.Adder],
                 state_encoder: Optional[Callable[[pddlenv.EnvState], Any]] = None):
        super().__init__(adders)
        self.state_encoder = state_encoder

    def _preprocess(self, timestep):
        if self.state_encoder is not None:
            timestep = timestep._replace(
                observation=self.state_encoder(timestep.observation))
        return timestep._replace(
            reward=np.array(timestep.reward, np.float32),
            discount=np.array(timestep.discount, np.float32),
        )

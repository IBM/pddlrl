# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import itertools
import operator

import numpy as np

from acme.wrappers import base

import pddlenv


class IntegerAction(base.EnvironmentWrapper):

    def _action_from_index(self, state: pddlenv.EnvState, action_idx: int):
        indices_dict = pddlenv.array.unravel_literal_indices(
            (np.zeros((1,), dtype=np.int), np.array((action_idx,), dtype=np.int)),
            self._shapes,
        )
        # since we're only dealing with 1 action, we should only have one arity in indices
        action_arity, indices = tuple(indices_dict.items())[0]
        action_cls = self._sorted_actions[action_arity][indices[-1][0]]

        objects = state.problem.objects
        return action_cls(*(objects[i[0]] for i in indices[1:-1]))

    def reset(self):
        timestep = self._environment.reset()

        problem = timestep.observation.problem
        grouped_actions = itertools.groupby(
            sorted(problem.actions, key=operator.attrgetter("arity")),
            key=operator.attrgetter("arity"),
        )
        self._sorted_actions = {
            arity: dict(enumerate(sorted(acts, key=operator.attrgetter("__name__"))))
            for arity, acts in grouped_actions
        }
        self._shapes = pddlenv.array.compute_shapes(len(problem.objects), problem.actions)
        return timestep

    def step(self, action):
        pddl_action = self._action_from_index(self._environment.state, action)
        return self._environment.step(pddl_action)

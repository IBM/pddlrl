# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

from typing import Callable

from acme.wrappers import base

import dm_env


class PreprocessWrapper(base.EnvironmentWrapper):
    preprocess_fn: Callable[[dm_env.TimeStep], dm_env.TimeStep]

    def __init__(self, environment, preprocess_fn):
        self.preprocess_fn = preprocess_fn
        super().__init__(environment)

    def reset(self):
        timestep = self._environment.reset()
        return self.preprocess_fn(timestep)

    def step(self, action):
        timestep = self._environment.step(action)
        return self.preprocess_fn(timestep)

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

from acme.wrappers import base

import dm_env


class BoundedEpisodes(base.EnvironmentWrapper):

    def __init__(self, environment, max_epsiode_length):
        self.max_epsiode_length = max_epsiode_length
        self._count = 0
        super().__init__(environment)

    def reset(self):
        self._count = 0
        return self._environment.reset()

    def step(self, action):
        self._count += 1
        timestep = self._environment.step(action)
        if self._count >= self.max_epsiode_length:
            timestep = timestep._replace(step_type=dm_env.StepType.LAST)

        return timestep

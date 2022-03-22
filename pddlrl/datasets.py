# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import numpy as np

from acme import specs

import pddlenv

from pddlrl import encoding
from pddlrl import env_utils


def create_environment_spec(problem: pddlenv.Problem, encoder: encoding.StateEncoder):
    action_shapes = pddlenv.array.compute_shapes(
        len(problem.objects),
        problem.actions,
    )
    num_actions = sum(np.prod(shape) for shape in action_shapes.values())
    return specs.EnvironmentSpec(
        encoder.spec(problem),
        specs.DiscreteArray(num_values=num_actions, dtype=np.int32, name="action_index"),
        specs.Array(shape=(), dtype=np.float32, name='reward'),
        specs.BoundedArray(shape=(), dtype=np.float32, minimum=0., maximum=1., name='discount'),
    )


class MultiObjectSetDataset:

    def __init__(self, rng, buffers, batch_size):
        self.rng = np.random.default_rng(rng)
        self.buffers = buffers
        self.batch_size = batch_size

    def sample(self, rng, batch_size):
        buffers = tuple(b for b in self.buffers if len(b) >= batch_size)
        if buffers:
            batch_buffer = rng.choice(buffers)
            return batch_buffer.sample(rng, batch_size)
        else:
            return ()

    def __iter__(self):
        while True:
            yield self.sample(self.rng, self.batch_size)


class ReachableStatesDataset:

    def __init__(self,
                 rng: np.random.Generator,
                 pddl_domain_path: str,
                 pddl_problem_dir: str,
                 batch_size: int):
        self.rng = rng
        self.batch_size = batch_size
        domain_path, problem_paths = env_utils.expand_pddl_paths(pddl_domain_path, pddl_problem_dir)

        self._reachable_states = {}
        for problem_path in problem_paths:
            literals, problem = pddlenv.parse_pddl_problem(domain_path, problem_path)
            self._reachable_states[problem] = sorted(
                pddlenv.reachable_states([pddlenv.EnvState(literals, problem)]),
                key=lambda x: (x.literals, x.problem),
            )
        self._problems = list(self._reachable_states.keys())

    def sample(self, rng, batch_size):
        problem = rng.choice(self._problems)
        return rng.choice(self._reachable_states[problem], batch_size)

    def __iter__(self):
        while True:
            yield self.sample(self.rng, self.batch_size)

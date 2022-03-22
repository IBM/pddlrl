# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import glob
import os
import string
from typing import Dict, Optional, Sequence, Set, Tuple, Type

import gin

import numpy as np

import pddlenv
from pddlenv import PDDLObject, Predicate

def pddl_root():
    return os.environ.get("PDDL_ROOT_DIR", ".")

def expand_pddl_paths(domain_path, problem_dir):
    domain_path = os.path.expanduser(os.path.join(pddl_root(), domain_path))

    problem_dir = os.path.expanduser(os.path.join(pddl_root(), problem_dir))
    problem_paths = sorted(glob.glob(os.path.join(problem_dir, "*.pddl")))
    return domain_path, problem_paths


@gin.configurable
def pddlgym_env(rng, domain_path, problem_dir, problem_index=None, random_reachable_init=False):
    domain_path = os.path.expanduser(os.path.join(pddl_root(), domain_path))
    problem_dir = os.path.expanduser(os.path.join(pddl_root(), problem_dir))

    if random_reachable_init:
        problem_path = sorted(glob.iglob(os.path.join(problem_dir, "*.pddl")))[problem_index]
        state_initializer = pddlenv.initializers.reachable_states_initializer(
            np.random.default_rng(rng), domain_path, problem_path)
    else:
        state_initializer = pddlenv.initializers.pddlgym_initializer(
            np.random.default_rng(rng), domain_path, problem_dir, problem_index)

    return pddlenv.PDDLEnv(state_initializer, pddlenv.PDDLDynamics())

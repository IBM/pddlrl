# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import itertools
import multiprocessing
import os
import secrets
import time
import signal
from typing import Any, Mapping, Optional, Protocol, Sequence
import json
import gin

import jax
import jax.numpy as jnp

import pandas as pd

from acme.jax import savers as acme_savers
from acme.jax import variable_utils

import haiku as hk

import rlax
from rlax._src import distributions

import pddlenv

from pddlrl import env_utils
from pddlrl.agents import lookahead
from pddlrl.exceptions import SignalInterrupt


# avoid CPU contention: each process (spawned by multiprocessing) will try to use all cores
def subprocess_initializer():
    xla_eigen_flag = f"--xla_cpu_multi_thread_eigen=false"
    intra_op_flag = f"intra_op_parallelism_threads=1"
    os.environ["XLA_FLAGS"] = f"{xla_eigen_flag} {intra_op_flag}"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] =""



class ValueBasedActor(Protocol):

    def select_action(self, observation):
        raise NotImplementedError

    def value(self, state: pddlenv.EnvState):
        raise NotImplementedError

    def update_and_wait(self):
        raise NotImplementedError


class SimpleLogger:

    def __init__(self, logs):
        self._logs = logs

    def write(self, data):
        self._logs.update(data)



# putting class names here for easier navigation with anaconda-mode in emacs
# lookahead.LookaheadActor
# lookahead.LookaheadValueFunction
# pddlenv.PDDLDynamics
# pddlenv.heuristic.Heuristic

def value_heuristic(actor: ValueBasedActor):
    def _value_heuristic(literals, problem):
        return (-jnp.squeeze(actor.value(pddlenv.EnvState(literals, problem)))
                # note: discounting is included in heuristic. See pddlenv.heuristic.Heuristic
                + actor._lookahead.dynamics.heuristic(literals, problem))
    return _value_heuristic


def output_path(problem_path, eval_dir, extension=".plan"):
    basename = os.path.basename(problem_path)
    name, _ = os.path.splitext(basename)
    return os.path.join(eval_dir, name+extension)


def write_plan(problem_path, plan, eval_dir):
    plan_path = output_path(problem_path, eval_dir, ".plan")
    with open(plan_path, "w") as f:
        for action in plan:
            print(action, file=f)
    return plan_path


def validate_plan(domain_path, problem_path, plan_path):
    from pddlrl.validate import validate, arrival
    return arrival(domain_path, problem_path, plan_path).returncode == 0


def write_log(problem_path, log, eval_dir):
    log_path = output_path(problem_path, eval_dir, ".json")
    print(json.dumps(log, indent=2, sort_keys=True))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, sort_keys=True)
    return log_path


def evaluate(heuristic_fn,
             domain_path: str,
             eval_dir: str,
             problem_path: str,
             evaluator_name: str,
             time_limit: int,
             expansion_limit: int,
             evaluation_limit: int):

    evaluator = {
        "gbfs" : pddlenv.search.GreedyBestFirst,
        "hc"   : pddlenv.search.HillClimbing,
    }[evaluator_name]

    logs = {
        "problem": os.path.basename(problem_path),
        "plan_length" : -1,
        "expanded_states" : -1,
        "search_instantiation": -1,
        "search_instantiation_elapsed": -1,
        "problem_instantiation": -1,
        "problem_instantiation_elapsed": -1,
        "init_instantiation": -1,
        "init_instantiation_elapsed": -1,
        "search_time" : -1,
        "search_time_elapsed" : -1,
        "solved"      : -1,
        "valid"       : -1,
        "exp/sec"     : -1,
    }

    time_start = time.perf_counter()
    def msg(message):
        now = time.perf_counter() # in seconds
        elapsed = now - time_start
        print(f"[msg] {elapsed} (sec): {message}")
        return now, elapsed

    time_prev  = time_start
    def record(step):
        nonlocal time_prev  # terrible
        now, elapsed = msg(step)
        logs[step] = now - time_prev
        logs[step+"_elapsed"] = elapsed
        time_prev = now
        return

    try:
        msg(f"{os.path.basename(problem_path)}: starting evaluation")

        msg(f"{os.path.basename(problem_path)}: setting signal")
        signal.signal(signal.SIGALRM,SignalInterrupt)
        signal.alarm(min(2147483647,int(time_limit)))
        msg(f"{os.path.basename(problem_path)}: signal set")
        try:
            msg(f"{os.path.basename(problem_path)}: starting search")
            bfs = evaluator(heuristic_fn, SimpleLogger(logs))
        finally:
            record("search_instantiation")
        try:
            parsed = pddlenv.parse_pddl_problem(domain_path, problem_path)
        finally:
            record("problem_instantiation")
        try:
            init_state = pddlenv.EnvState(*parsed)
        finally:
            record("init_instantiation")
        try:
            plan = bfs.search(init_state,
                              evaluation_limit=evaluation_limit,
                              expansion_limit=expansion_limit,)
        finally:
            record("search_time")
        # cancelling alarm
        signal.alarm(0)

        if plan is not None:
            logs["solved"] = True
            msg(f"{os.path.basename(problem_path)}: writing plan")
            plan_path      = write_plan(problem_path, plan, eval_dir)
            msg(f"{os.path.basename(problem_path)}: validating")
            logs["valid"]  = validate_plan(domain_path, problem_path, plan_path)
        logs["eval/sec"] = logs["evaluated_states"]/logs["search_time"]
        logs["exp/sec"] = logs["expanded_states"]/logs["search_time"]
    except SignalInterrupt as e:
        if e.signal == signal.SIGALRM:
            msg(f"{os.path.basename(problem_path)}: SIGALRM -- user set timed out")
            # when timed out, proceed to write logs
            pass
        elif e.signal == signal.SIGUSR2:
            msg(f"{os.path.basename(problem_path)}: SIGUSR2 -- it would be a job scheduler that killed me")
            # when killed by the job scheduler, we should not record this result
            import sys
            sys.exit(2)
        else:
            msg(f"{os.path.basename(problem_path)}: signal {e.signal} --- other than SIGALRM or SIGUSR2!")
            # when killed by the job scheduler, we should not record this result
            raise e
    except Exception as e:
        msg(f"{os.path.basename(problem_path)}: {type(e)}: {e}")
        # when errored, we should not record this result
        msg(f"{os.path.basename(problem_path)}: aborted, not logging the results")
        raise e
    # cancelling alarm
    signal.alarm(0)
    msg(f"{os.path.basename(problem_path)}: writing the log file")
    return write_log(problem_path, logs, eval_dir)


# not used
def discounted_return(rewards, discounts):
    returns = 0.
    for rew, discount in zip(rewards[::-1], discounts[::-1]):
        returns = rew + discount * returns
    return returns



################################################################
# note: code below is not checked

class VariableClientWrapper:

    def __init__(self, params, device=None):
        self.params = jax.tree_map(lambda x: jax.device_put(x, device=device), params)

    def update(self):
        pass

    def update_and_wait(self):
        pass


@gin.configurable
def make_lookahead_actor(seed: int,
                         lookahead_network: lookahead.LookaheadValueFunction,
                         action_distribution: distributions.DiscreteDistribution,
                         variable_client: variable_utils.VariableClient):
    return lookahead.LookaheadActor(
        hk.PRNGSequence(seed),
        lookahead_network,
        action_distribution,
        variable_client,
    )

def load_lookahead_actor(results_dir: str,
                         iteration: Optional[int] = None,
                         action_distribution: Optional[distributions.DiscreteDistribution] = None,
                         seed: Optional[int] = None):

    if iteration is None:
        import glob
        params_dir = os.path.dirname(sorted(glob.glob(os.path.join(results_dir, "*/nest_exemplar")))[-1])
    else:
        params_dir = os.path.join(results_dir, str(iteration))

    print(f"loading weights from {params_dir}")
    params = acme_savers.restore_from_path(params_dir)
    params = jax.tree_map(jnp.array, params)
    print(f"loaded")

    import pddlrl.main
    train_dir, hyper = pddlrl.main.load_hyper()
    print("train_dir:",train_dir)
    print("hyper:",json.dumps(hyper,indent=2))
    pddlrl.main.setup_gin(train_dir, hyper)
    gin.parse_config("""
    agent/LookaheadValueFunction.network = @make_network()
    agent/LookaheadValueFunction.encoder = @make_encoder()
    agent/LookaheadValueFunction.dynamics = @agent/PDDLDynamics()
    make_lookahead_actor.lookahead_network = @agent/LookaheadValueFunction()
    """)

    if action_distribution is None:
        action_distribution = rlax.greedy()
    if seed is None:
        seed = gin.query_parameter("%SEED")

    return make_lookahead_actor(
        seed=seed,
        action_distribution=action_distribution,
        variable_client=VariableClientWrapper(params),
    )

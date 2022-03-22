# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os
from typing import Any, Optional, Protocol, Mapping

import gin

import numpy as np

import dm_env

import acme
from acme.agents import agent
from acme.jax import savers as acme_savers
from dm_env import TimeStep
from pddlenv import PDDLEnv, EnvState, Predicate, Heuristic
from collections import deque
import math
import os
import subprocess

class Logger(Protocol):

    def write(self, data: Mapping[str, Any]):
        raise NotImplementedError


class Saver(Protocol):

    def save(self, step_count: int, var_source: acme.VariableSource):
        raise NotImplementedError


@gin.register
class SimpleSaver(Saver):

    def __init__(self, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = dir_path

    def save(self, step_count: int, var_source: acme.VariableSource):
        save_dir_path = os.path.join(self.dir_path, str(step_count))
        acme_savers.save_to_path(save_dir_path, var_source.get_variables([])[0])

        config_path = os.path.join(save_dir_path, "config.gin")
        with open(config_path, "w") as f:
            f.write(gin.operative_config_str())


def undiscount(value, gamma):
    if gamma == 1.0:
        return value
    else:
        return math.log(max(1e-20,(gamma-1)*value+1),gamma)



def show_gpu_memory(t=None):
    # os.system("nvidia-smi --query-gpu=memory.free --format=csv")
    mb = int((subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])).decode("utf-8").split("\n")[0])
    if t is not None:
        print(f"step {t}: remaining gpu memory: {mb} Mb")
    else:
        print(f"remaining gpu memory: {mb} Mb")



@gin.configurable
def experiment_loop(agent: agent.Agent, saver: Saver, train_env: PDDLEnv, num_steps: int,
                    max_episode_length: int, save_every_n: int, log_every_n: int,
                    logger: Optional[Logger] = None):

    timestep: TimeStep = train_env.reset()
    agent.observe_first(timestep)

    # run for some number of steps
    t = 0
    episode_start = t
    num_updates = 0
    last_update = 0

    # note: heuristics.discount is already set during initialization, thus the value is discounted.
    # See pddlenv.Heuristic ~/miniconda3/envs/pddlrl/lib/python3.8/site-packages/pddlenv/heuristic.py
    h_fn: Heuristic = agent._actor._lookahead.dynamics.heuristic
    gamma = h_fn.discount

    goal_count = []
    reset_count = []

    try:
        while t < num_steps:
            # sample an action
            a_t = agent.select_action(timestep.observation)

            # step environment and log reward
            timestep: TimeStep = train_env.step(a_t)
            t += 1
            goal_reached = (timestep.step_type == dm_env.StepType.LAST)
            goal_count.append(goal_reached)

            # If max episode length is reached, mark this step as a terminal step but keeping the
            # discount unchanged and possibly non-zero.
            if t - episode_start >= max_episode_length:
                timestep: TimeStep = timestep._replace(step_type=dm_env.StepType.LAST)

            reset_count.append(timestep.step_type == dm_env.StepType.LAST)

            def update_and_log(i):
                t = last_update + i
                agent._learner.batch_losses.clear()
                agent._learner.batch_samples.clear()
                agent._learner.batch_targets.clear()
                agent._learner.batch_values.clear()
                agent._learner.batch_heuristics.clear()
                agent.update()
                # -> agent.update()        = Acme.agent.update()
                # -> agent._learner.step() = ApproximateValueIteration.step()
                # -> agent._actor.update() = LookaheadActor.update()
                # -> agent._actor._client.update() = acme.jax.variable_utils.VariableClient.update()
                #    ^^^ doesnt seem to be doing anything
                samples            = np.sum(agent._learner.batch_samples)
                loss               = np.sum(agent._learner.batch_losses)/samples
                correction_targets = np.sum(agent._learner.batch_targets)/samples
                correction_values  = np.sum(agent._learner.batch_values)/samples
                h_gamma            = np.sum(agent._learner.batch_heuristics)/samples
                original_values = correction_values-h_gamma
                learned_h = undiscount(-original_values,gamma)
                h         = undiscount(h_gamma,gamma)
                heuristic_discrepancies = abs(learned_h - h)
                if t % log_every_n == 0 and logger is not None:
                    logger.write({
                        "0-goal": int(goal_count[i]),
                        "0-reset":int(reset_count[i]),
                        "11-loss": loss,
                        "12-E_a Q'(a,s)": correction_targets,
                        "13-V'(s)": correction_values,
                        "21-V(s)": original_values,
                        "22-h_gamma(s)": h_gamma,
                        "31-h^(s)":learned_h,
                        "32-h(s)": h,
                        "33-|h^(s)-h(s)|": heuristic_discrepancies,
                        "timestep": t,
                        "samples":samples,
                    })
                if t % 100 == 0:
                    show_gpu_memory(t)
                if t % save_every_n == 0:
                    saver.save(t, agent)

            agent.observe(a_t, timestep)
            update_and_log(0)
            last_update = t
            goal_count = []
            reset_count = []

            # reset if state is terminal
            if timestep.last():
                episode_start = t
                timestep = train_env.reset()
                agent.observe_first(timestep)

    finally:
        if t >= 2:
            saver.save(t, agent)


@gin.configurable
def run(seed, agent_cls, env_cls, logger):
    agent_seed, env_seed = [
        s.generate_state(2, np.uint32)
        for s in np.random.SeedSequence(seed).spawn(2)
    ]

    # create training and evaluation environments
    train_env = env_cls(env_seed)

    # create dqn agent
    agent = agent_cls(seed=agent_seed)  # type: ignore

    experiment_loop(
        agent=agent,
        train_env=train_env,
        logger=logger,
    )

    return agent


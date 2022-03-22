# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

from typing import Mapping, Tuple

from acme import specs
from acme.agents import agent
from acme.jax import variable_utils
import haiku as hk
import optax

from rlax._src import distributions

import numpy as np

import pddlenv

from pddlrl import adders as nlm_adders
from pddlrl import datasets
from pddlrl import encoding
from pddlrl import replay
from pddlrl.agents.lookahead import actors
from pddlrl.agents.lookahead import learning


class AVIAgent(agent.Agent):

    def __init__(self,
                 seed: int,
                 network: actors.ValueNetwork,
                 optimizer: optax.GradientTransformation,
                 dynamics: pddlenv.PDDLDynamics,
                 encoder: encoding.StateEncoder,
                 environment_specs: Mapping[Tuple[pddlenv.PDDLObject, ...], specs.EnvironmentSpec],
                 learner_policy: distributions.DiscreteDistribution,
                 actor_policy: distributions.DiscreteDistribution,
                 batch_size: int,
                 target_update_period: int,
                 train_after_steps: int,
                 max_replay_size: int,
                 samples_per_insert: int):
        dataset_seed, learner_seed, actor_seed = np.random.SeedSequence(seed).spawn(3)

        self.max_replay_size = max_replay_size
        buffers = {
            objects: replay.SimpleReplayBuffer(max_replay_size)
            for objects in environment_specs
        }
        adder = nlm_adders.MultiObjectSetStateAdder(
            adders={
                objects: nlm_adders.StateAdder(buffer)
                for objects, buffer in buffers.items()
            },
        )
        dataset = datasets.MultiObjectSetDataset(
            rng=np.random.default_rng(dataset_seed),
            buffers=list(buffers.values()),
            batch_size=batch_size,
        )

        lookahead_network = actors.LookaheadValueFunction(
            network=network,
            dynamics=dynamics,
            encoder=encoder
        )

        learner = learning.ApproximateValueIteration(
            rng=hk.PRNGSequence(learner_seed.generate_state(2)),
            obs_spec=list(environment_specs.values())[0].observations,
            lookahead_network=lookahead_network,
            optimizer=optimizer,
            iterator=iter(dataset),
            action_distribution=learner_policy,
            target_update_period=target_update_period,
        )
        actor = actors.LookaheadActor(
            rng=hk.PRNGSequence(actor_seed.generate_state(2)),
            lookahead_network=lookahead_network,
            action_distribution=actor_policy,
            variable_client=variable_utils.VariableClient(learner, ''),
            adder=adder,
        )

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(batch_size, train_after_steps),
            observations_per_step=float(batch_size) / samples_per_insert)

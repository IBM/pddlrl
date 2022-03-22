# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

from typing import Sequence

import gin

import haiku as hk

import rlax

import optax

import nlmax

import jax

import pddlenv
from pddlenv.generators import problems
from pddlenv.generators import literals

from pddlrl import env_utils, datasets, encoding, evaluate
from pddlrl.agents import lookahead
from pddlrl.exceptions import InvalidHyperparameterError

gin.external_configurable(lookahead.AVIAgent)
gin.external_configurable(lookahead.LookaheadValueFunction)
gin.external_configurable(lookahead.LookaheadActor)
gin.external_configurable(pddlenv.PDDLDynamics)
gin.external_configurable(pddlenv.Heuristic)
gin.external_configurable(problems.blocks.SingleTowerSampler)
gin.external_configurable(literals.blocks.ClearSampler)
gin.external_configurable(rlax.softmax, module="rlax")
gin.external_configurable(rlax.greedy, module="rlax")
gin.external_configurable(optax.adam)


@gin.configurable
def make_network(num_nlm_layers, num_hidden_units, final_hidden_units, residual, with_permute=True,
                 max_arity=3, activation="sigmoid"):

    activation = getattr(jax.nn,activation)
    def mlp(x):
        output = hk.Linear(num_hidden_units)(x)
        return activation(output)

    def model(inputs):
        if not (len(inputs) <= num_nlm_layers):
            raise InvalidHyperparameterError(f"number of layers must be > max_arity. Given: max_arity = {len(inputs)}, num_nlm_layers = {num_nlm_layers} ")
        outputs = inputs
        for i in range(num_nlm_layers, 1, -1):
            mode = None
            current_arity = max([x.ndim - 2 for x in outputs])
            if i < current_arity + 1:
                mode = "reduce"
            elif i > current_arity + 1 and current_arity < max_arity:
                mode = "expand"
            outputs = nlmax.nlm_layer(outputs, mlp,
                                      residual=residual,
                                      with_permute=with_permute,
                                      mode=mode)

        outputs = nlmax.nlm_layer(
            outputs,
            hk.Linear(final_hidden_units if final_hidden_units > 0 else 1),
            residual=residual,
            with_permute=with_permute,
            mode="reduce",
        )
        assert len(outputs) == 1, [x.shape for x in outputs]

        outputs = outputs[0]
        if final_hidden_units > 0:
            outputs = hk.Linear(1)(jax.nn.relu(outputs))

        return outputs[:, 0]

    return model


@gin.configurable
def make_encoder():
    return encoding.NLMStateEncoder()


@gin.configurable
def make_training_problems(domain_path, problem_dir) -> Sequence[pddlenv.Problem]:
    domain_path, problem_paths = env_utils.expand_pddl_paths(domain_path, problem_dir)
    return [
        pddlenv.parse_pddl_problem(domain_path, problem_path)[1]
        for problem_path in problem_paths
    ]


@gin.configurable
def make_environment_specs(encoder, training_problems):
    environment_specs = {}
    for problem in training_problems:
        if problem.objects not in environment_specs:
            environment_specs[problem.objects] = datasets.create_environment_spec(problem, encoder)

    return environment_specs

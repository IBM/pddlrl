# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os
import glob
import argparse
import random
import tqdm
import gin
import json
import glob
from acme.utils import loggers
from pddlrl import loop
from pddlrl.exceptions import *

parser = argparse.ArgumentParser()

# common options

parser.add_argument("mode",
                    choices=["train-RL","evaluate-RL","evaluate-nonRL",],
                    help="Evaluation mode.")

parser.add_argument("--runid",
                    help="An ID string for identifying a run. To evaluate a training result, the same ID must be provided.")

# training options

default_gin_file = os.path.join(os.path.dirname(__file__), "blocksworld.gin")

parser.add_argument("--ginfile",
                    default=[default_gin_file],
                    action="append",
                    help="List of paths to the config files. This option can be specified multiple times.")

parser.add_argument("--train-root-dir",
                    help="""
Directory used to store the trained weights. Weights are located in <train-dir>/<hash>/<snapshot-iteration>.
This argument is required in training (when hash value is not available yet)
and when only runid is known (same; hash value is not available).
This argument is ignored when --hyperfile is specified.
""")

parser.add_argument("--domain",
                    help="Pathname to a directory that contains a domain file and a train/ directory containing training problems. The value is mandatory in all modes.")

parser.add_argument("--hyper",
                    action="append",
                    default=[],
                    help="""
A string representation of a python list that specifies a tunable hyperparameter.
Its first value must be a string, a name of a settable place in gin.
The rest is a list of values to be searched from. For example,
 --hyper '["BATCH_SIZE", 10, 20, 50]' specifies the batch size.
This option can be specified multiple times to add more hyperparameters.
See each gin file under pddlrl/experiments/ to see what parameter can be configured.
""")

parser.add_argument("--hyperfile",
                    help="""
Pathname to the hyperparameter file (hyper.json).
--hyperfile and --train-root-dir/--runid are mutually exclusive.
When --hyperfile is specified, --train-root-dir is deduced from the value of --hyperfile.
When --hyperfile is not specified, it requires both --train-root-dir and --runid,
which are then used to deduce the location of hyperfile.
""")


# evaluation options

parser.add_argument("--evaluator",
                    choices=["gbfs","hc","train"],
                    help="The name of the evaluator, which could be \"train\" in the training mode.")

parser.add_argument("--evaluation-weight-snapshot-iteration",
                    type=eval,
                    default=None,
                    help="Specify the iteration in which the snapshot of the weights were taken. Meaningful only in evaluate-RL mode.")

parser.add_argument("--problem",
                    help="Pathname to the problem file. Meaningful only in evaluate-RL/-nonRL. When it is a directory, all files with the extension .pddl will be iterated over.")

parser.add_argument("--time-limit",
                    type=int,
                    default=1000000000,
                    help="Runtime limit for evaluation. Meaningful only in evaluate-RL/-nonRL.")
parser.add_argument("--evaluation-limit",
                    type=int,
                    default=1000000000,
                    help="Evaluation limit for evaluation. Meaningful only in evaluate-RL/-nonRL.")
parser.add_argument("--expansion-limit",
                    type=int,
                    default=1000000000,
                    help="Expansion limit for evaluation. Meaningful only in evaluate-RL/-nonRL.")

parser.add_argument("-f","--force",
                    action="store_true",
                    help="Force re-running experiments even if the log file already exist.")

parser.add_argument("--eval-dir",
                    help="""
Supersedes the evaluation directory where the output is written.
By default, its value is deduced from the value of train_root_dir:
The default location is <train_root_dir>-<evaluator>/<hash>/.
""")

parser.add_argument("-j","--processes",
                    type=int,
                    help="Enable parallel processing (evaluation of non-RL only), and specify the number of CPU cores.")

args = parser.parse_args()


class ProgressBarLogger(loggers.Logger):

    def __init__(self, progbar: tqdm.tqdm):
        self.progbar = progbar

    def write(self, data: loggers.LoggingData):
        loss = data["11-loss"]
        self.progbar.set_description(f"loss: {loss}")
        self.progbar.update(data["timestep"] - self.progbar.n)

import time
from acme.utils.loggers import base
import tensorflow as tf

class TFSummaryLogger(base.Logger):
    # modifed from acme.utils.loggers.tf_summary.TFSummaryLogger
    def __init__(self, logdir: str):
        self._time = time.time()
        self._iter = 0
        self.summary = tf.summary.create_file_writer(logdir)

    def _format_key(self,key: str) -> str:
        """Internal function for formatting keys in Tensorboard format."""
        return key.title().replace('_', '')

    def write(self, values: base.LoggingData):
        with self.summary.as_default():
            for key, value in sorted(values.items()):
                tf.summary.scalar(
                    f'{self._format_key(key)}',
                    value,
                    step=self._iter)
        self._iter += 1


def print_misc_info():
    import os
    for key in ["HOSTNAME", "LSB_JOBID"]:
        try:
            print(f"{key} = {os.environ[key]}")
        except KeyError:
            pass


# training weights are stored in train_dir
# args.train_dir = feb13/domains/<domain>/<heur>-True
# train_dir      = feb13/domains/<domain>/<heur>-True/hash-<hash>/ : (1)
# train_dir      = feb13/domains/<domain>/<heur>-True/run-<runid>/ : (2)
# (1),(2) are symlinks.

# evaluateion resutls are stored in eval_dir
# eval_dir      = feb13/domains/<domain>/<heur>-True-<evaluator>/hash-<hash>/ : (1)
# eval_dir      = feb13/domains/<domain>/<heur>-True-<evaluator>/run-<runid>/ : (2)
# (1),(2) are symlinks.


def select_hyper(require_unique=True):
    for _ in range(100):
        # read the hyperparameter, select one randomly, then append it to the dict
        hyper = {}
        hyper["DOMAIN_PATH"]       = os.path.join(args.domain, "domain.pddl")
        hyper["TRAIN_PROBLEM_DIR"] = os.path.join(args.domain, "train")
        for hyper_string in args.hyper:
            name, *values = eval(hyper_string)
            value = random.choice(values)
            hyper[name] = value

        # compute a unique hash value for a hyperparameter.
        import hashlib, json
        hash = hashlib.md5(str(json.dumps(hyper,sort_keys=True)).encode('utf-8')).hexdigest()

        # each unique hyperparameter corresponds to a unique directory.
        train_dir = os.path.join(args.train_root_dir, f"hash-{hash}")
        if require_unique and os.path.isdir(train_dir):
            # if it exists, it is already tried.
            continue
        else:
            # otherwise, this is new.
            os.makedirs(train_dir,exist_ok=not require_unique)
            run_dir = os.path.join(args.train_root_dir,f"run-{args.runid}")
            if not os.path.exists(run_dir):
                os.symlink(f"hash-{hash}",run_dir)
            return run_dir, hyper
    assert False, "failed to generate a unique hyperparameter"


def save_hyper(train_dir, hyper):
    # record the hyperparameter
    with open(os.path.join(train_dir, "hyper.json"), "w") as f:
        json.dump(hyper,f,indent=2)


def load_hyper():
    if args.hyperfile:
        # location of hyperfile is known; no need to know runid
        train_dir = os.path.dirname(args.hyperfile)
        with open(args.hyperfile, "r") as f:
            hyper = json.load(f)
    else:
        # use the fact that runid directory is a symlink
        train_dir = os.path.join(args.train_root_dir,f"run-{args.runid}")
        with open(os.path.join(train_dir, "hyper.json"), "r") as f:
            hyper = json.load(f)
    return train_dir, hyper


def train_to_eval_dir(train_dir):
    if args.eval_dir:
        eval_dir = args.eval_dir
    else:
        train_root_dir, hash_or_run = os.path.split(train_dir)
        eval_dir = os.path.join(train_root_dir+"-"+args.evaluator, hash_or_run)
    os.makedirs(eval_dir,exist_ok=True)
    return eval_dir


def setup_gin(train_dir, hyper):
    # read one line from config file(s), and append it to the list
    gin_lines = []
    gin_lines.append(f"TRAIN_DIR = '{train_dir}'")
    for ginfile in args.ginfile:
        with open(ginfile, "r") as file:
            gin_lines.append(file.read())

    # similarly copy entries from hyper to the list
    for name, value in hyper.items():
        if isinstance(value, str):
            gin_lines.append(f"{name} = '{value}'")
        else:
            gin_lines.append(f"{name} = {value}")

    gin.clear_config()
    gin.parse_config(gin_lines)
    pass


def train_RL():
    from pddlrl.exceptions import InvalidHyperparameterError
    while True:
        try:
            train_RL_aux()
            break
        except RuntimeError as e:
            print(e)
            continue
        except InvalidHyperparameterError:
            continue


def train_RL_aux():
    train_dir, hyper = select_hyper()
    print("train_dir:",train_dir)
    print("hyper:",json.dumps(hyper,indent=2))
    save_hyper(train_dir, hyper)
    setup_gin(train_dir, hyper)

    import signal
    signal.signal(signal.SIGALRM,SignalInterrupt)
    with tqdm.tqdm(maxinterval=1.,
                   total=hyper["experiment_loop.num_steps"],
                   dynamic_ncols=True) as progbar:
        try:
            loop.run(logger=loggers.Dispatcher([
                ProgressBarLogger(progbar),
                loggers.CSVLogger(train_dir),
                TFSummaryLogger(train_dir),
            ]))
        except SignalInterrupt as e:
            print("received {e}")


def output_path(problem_path, eval_dir, extension=".plan"):
    basename = os.path.basename(problem_path)
    name, _ = os.path.splitext(basename)
    return os.path.join(eval_dir, name+extension)


def evaluate_RL():
    train_dir, hyper = load_hyper()
    eval_dir = train_to_eval_dir(train_dir)
    print("train_dir:",train_dir)
    print("eval_dir:",eval_dir)
    print("hyper:",json.dumps(hyper,indent=2))
    setup_gin(train_dir, hyper)

    iteration = args.evaluation_weight_snapshot_iteration
    if iteration is not None:
        eval_dir = os.path.join(eval_dir, str(iteration))
        os.makedirs(eval_dir, exist_ok=True)

    if os.path.isdir(args.problem):
        problems = glob.glob(os.path.join(args.problem,"*.pddl"))
    else:
        problems = [args.problem]

    import gc
    from pddlrl.evaluate import evaluate,value_heuristic,load_lookahead_actor
    for problem in problems:
        if (not args.force) and os.path.exists(output_path(problem, eval_dir, ".json")):
            print("logfile alrady exist, skipping this problem")
            continue
        gc.collect()
        evaluate(value_heuristic(load_lookahead_actor(train_dir, iteration=iteration)),
                 hyper['DOMAIN_PATH'],
                 eval_dir,
                 problem,
                 args.evaluator,
                 args.time_limit,
                 args.expansion_limit,
                 args.evaluation_limit)
    pass


def evaluate_nonRL():
    train_dir, hyper = select_hyper(require_unique=False)
    eval_dir = train_to_eval_dir(train_dir)
    print("train_dir:",train_dir)
    print("eval_dir:",eval_dir)
    print("hyper:",json.dumps(hyper,indent=2))
    save_hyper(train_dir, hyper)
    setup_gin(train_dir, hyper)

    if os.path.isdir(args.problem):
        problems = glob.glob(os.path.join(args.problem,"*.pddl"))
    else:
        problems = [args.problem]

    import gc
    from pddlrl.evaluate import evaluate
    from pddlenv import Heuristic
    import multiprocessing as mp
    p = mp.Pool(args.processes)
    results = []
    for problem in problems:
        if (not args.force) and os.path.exists(output_path(problem, eval_dir, ".json")):
            print("logfile alrady exist, skipping this problem")
            continue
        gc.collect()
        r = p.apply_async(evaluate,
                          (Heuristic(hyper['SHAPING_HEURISTIC']),
                           hyper['DOMAIN_PATH'],
                           eval_dir,
                           problem,
                           args.evaluator,
                           args.time_limit,
                           args.expansion_limit,
                           args.evaluation_limit))
        results.append(r)
    for r in tqdm.tqdm(results):
        r.get()
    pass


def main():
    print_misc_info()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU") # avoid OOM from tf and jax memory conflict
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".5"
    try:
        {
            "train-RL":       train_RL,
            "evaluate-RL":    evaluate_RL,
            "evaluate-nonRL": evaluate_nonRL,
        }[args.mode]()
    except Exception as e:
        print(f"ERROR: run {args.runid} failed with exception:\n  {e}")
        from pddlrl.stacktrace import format
        format()

if __name__ == "__main__":
    main()

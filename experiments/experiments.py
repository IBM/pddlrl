#!/usr/bin/env python3

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

# usage:
# ./experiments.py RLExperiments    --local-scheduler --workers 200 --log-level INFO --root nov9 --cluster
# ./experiments.py NonRLExperiments --local-scheduler --workers 200 --log-level INFO --root nov9 --cluster



# set the log level to info
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import os
import os.path
import subprocess
import luigi
import glob
import uuid
import stacktrace
import datetime

domain_sets = [
    # "domains",
    "erez-domains2",
    "cherry-pick2",
]
heuristics = [
    "hadd",
    # hmax also showed similar improvments, but was omitted from paper due to the space limitation
    # "hmax",
    "blind",
    "hff",
]

# note: seed is shared with runid in this script,
# but it is not required by the main program

class AbstractSetting(luigi.Task):
    #### meta parameters
    dry = luigi.BoolParameter(default=False,description="dry-run. Just print the command, without running it.")
    cluster = luigi.BoolParameter(default=False,description="prepend a CCC cluster submission command before the command line argument")
    # for the final experiments, this is necessary because
    # the total number of jobs will be more than 4000.
    # CCC does not accept more than 4000 jobs pending from one user.

    root = luigi.Parameter(default="results",description="result directory")

    # Don't automatically retry. Usually failed due to a timeout or disk space issue that retries won't fix.
    retry_count = 0

    pass


class AbstractExperiment(AbstractSetting):
    mem = luigi.IntParameter(default=32)
    maxmem = luigi.IntParameter(default=512)
    cores = luigi.Parameter(default="1+1")
    queue = luigi.Parameter(default="x86_12h")
    requirements = luigi.Parameter(default="'(v100||a100)&&(hname!=cccxc444)&&(hname!=cccxc506)'")
    proj = luigi.Parameter(default="pddlrl",
                           description="project name to be displayed in the LSF job scheduler.")
    uuid4 = luigi.Parameter(default="j"+str(uuid.uuid4()))

    # required parameters
    seed = luigi.IntParameter()
    domain = luigi.Parameter()
    heur = luigi.Parameter()

    @property
    def train_root_dir(self):
        return os.path.join(self.root,self.domain,self.heur+"-True")

    def run(self):
        if self.cluster:
            # TERM_OWNER: job killed by owner.
            # Exited with exit code 130.
            # TERM_RUNLIMIT: job killed after reaching LSF run time limit.
            # Exited with exit code 140.
            # TERM_MEMLIMIT: job killed after reaching LSF memory usage limit.
            # Exited with exit code 137.
            while True:
                print(datetime.datetime.now().isoformat(),self.uuid4," ".join(map(str,self.cmd())))
                if self.dry:
                    return
                exitstatus = subprocess.run(map(str,self.cmd())).returncode
                print(datetime.datetime.now().isoformat(),self.uuid4,f"exitstatus:{exitstatus}")
                if exitstatus == 137:
                    print(datetime.datetime.now().isoformat(),self.uuid4,f"job died due to memory limit. doubling: {self.mem} -> {self.mem*2}")
                    self.mem *= 2
                    if self.mem > self.maxmem:
                        print(datetime.datetime.now().isoformat(),self.uuid4,f"reached the maximum memory {self.maxmem}, aborting.")
                        break
                elif exitstatus == 140:
                    print(datetime.datetime.now().isoformat(),self.uuid4,f"job died due to time limit. rerunning it")
                elif exitstatus == 2:
                    print(datetime.datetime.now().isoformat(),self.uuid4,f"job died due to time limit. rerunning it")
                else:
                    print(datetime.datetime.now().isoformat(),self.uuid4,f"job finished successfully")
                    break
        else:
            print(" ".join(map(str,self.cmd())))
            if self.dry:
                return
            subprocess.run(map(str,self.cmd()))
        return

    def cmd(self):
        cmd = []
        if self.cluster:
            cmd += ["jbsub", "-mem", f"{self.mem}g"]
            cmd += ["-wait"]
            cmd += ["-cores", self.cores]
            if self.requirements:
                cmd += ["-require", self.requirements]
            cmd += ["-proj", self.proj+"-"+self.root]
            cmd += ["-queue", self.queue]
            cmd += ["-name", self.uuid4]
        return cmd

    pass


class RLTrainingExperiment(AbstractExperiment):
    num_steps = luigi.IntParameter(default=10000)
    discount = luigi.FloatParameter(default=0.9999)
    arity = luigi.IntParameter(default=3)
    episode_length = luigi.IntParameter(default=40)
    target_update_period = luigi.IntParameter(default=400)
    arity = luigi.IntParameter(default=3)
    width = luigi.IntParameter(default=8)
    depth = luigi.IntParameter(default=5)
    batch = luigi.IntParameter(default=25)
    temp  = luigi.FloatParameter(default=1.0)
    lr    = luigi.FloatParameter(default=0.001)

    cores = luigi.Parameter(default="1+1")
    queue = luigi.Parameter(default="x86_12h")
    requirements = luigi.Parameter(default="(v100||a100)&&(hname!=cccxc444)&&(hname!=cccxc506)")
    proj = luigi.Parameter(default="pddlrl-train")

    def output(self):
        return luigi.LocalTarget(os.path.join(self.train_root_dir,f"run-{self.seed}",str(self.num_steps),"array_nest"))

    def cmd(self):
        cmd = super().cmd()
        cmd += ["python", "-u", "../pddlrl/main.py"]
        cmd += ["train-RL"]
        cmd += ["--train-root-dir",self.train_root_dir]
        cmd += ["--runid",self.seed]
        cmd += ["--domain",self.domain]
        cmd += ["--hyper",f"['SHAPING_HEURISTIC',                 '{self.heur}']"]
        cmd += ["--hyper",f"['SEED',                              {self.seed}]"]
        cmd += ["--hyper",f"['experiment_loop.num_steps',         {self.num_steps}]"]
        cmd += ["--hyper",f"['experiment_loop.max_episode_length',{self.episode_length}]"]
        cmd += ["--hyper",f"['TARGET_UPDATE_PERIOD',              {self.target_update_period}]"]
        cmd += ["--hyper",f"['LEARNING_RATE',                     {self.lr}]"]
        cmd += ["--hyper",f"['MAX_ARITY',                         {self.arity}]"]
        cmd += ["--hyper",f"['MLP_HIDDEN_UNITS',                  {self.width}]"]
        cmd += ["--hyper",f"['NUM_LAYERS',                        {self.depth}]"]
        cmd += ["--hyper",f"['BATCH_SIZE',                        {self.batch}]"]
        cmd += ["--hyper",f"['TEMPERATURE',                       {self.temp}]"]
        return cmd


class RLEvaluationExperiment(AbstractExperiment):
    search     = luigi.Parameter(default="gbfs")
    eval_limit = luigi.IntParameter(default=100000)
    problem    = luigi.Parameter()

    cores = luigi.Parameter(default="1")
    queue = luigi.Parameter(default="x86_6h")
    requirements = luigi.Parameter(default=None)
    proj = luigi.Parameter(default="pddlrl-eval")

    def requires(self):
        if "logistics" in self.domain or "satellite" in self.domain:
            return RLTrainingExperiment(depth=4,seed=self.seed,domain=self.domain,heur=self.heur,
                                        uuid4=self.uuid4,
                                        cluster=self.cluster, root=self.root, dry=self.dry)
        else:
            return RLTrainingExperiment(depth=5,seed=self.seed,domain=self.domain,heur=self.heur,
                                        uuid4=self.uuid4,
                                        cluster=self.cluster, root=self.root, dry=self.dry)
        pass

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.train_root_dir+"-"+self.search,f"run-{self.seed}",os.path.split(self.problem)[1]+".json"))

    def cmd(self):
        cmd = super().cmd()
        cmd += ["python", "-u", "../pddlrl/main.py"]
        cmd += ["evaluate-RL"]
        cmd += ["--train-root-dir",self.train_root_dir]
        cmd += ["--runid",self.seed]
        cmd += ["--domain",self.domain]
        cmd += ["--evaluator",self.search]
        cmd += ["--evaluation-limit",self.eval_limit]
        cmd += ["--problem",self.problem]
        return cmd


class NonRLEvaluationExperiment(AbstractExperiment):
    search = luigi.Parameter(default="gbfs")
    eval_limit = luigi.IntParameter(default=100000)
    problem = luigi.Parameter()

    cores = luigi.Parameter(default="64")
    mem = luigi.IntParameter(default=512)
    queue = luigi.Parameter(default="x86_12h")
    requirements = luigi.Parameter(default=None)
    proj = luigi.Parameter(default="pddlrl-nonRL")

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.train_root_dir+"-"+self.search,f"run-{self.seed}",os.path.split(self.problem)[1]+".json"))

    def cmd(self):
        cmd = super().cmd()
        cmd += ["python", "-u", "../pddlrl/main.py"]
        cmd += ["evaluate-nonRL"]
        cmd += ["--train-root-dir",self.train_root_dir]
        cmd += ["--runid",self.seed]
        cmd += ["--domain",self.domain]
        cmd += ["--evaluator",self.search]
        cmd += ["--evaluation-limit",self.eval_limit]
        cmd += ["--problem",self.problem]
        cmd += ["--hyper",f"['SEED',              {self.seed}]"]
        cmd += ["--hyper",f"['SHAPING_HEURISTIC', '{self.heur}']"]
        cmd += ["-j",64]        # parallel processing
        return cmd


class RLExperiments(AbstractSetting):
    seeds = luigi.IntParameter(default=20)
    def requires(self):
        for domset in domain_sets:
            for domain in glob.glob(os.path.join(domset,"*")):
                if not os.path.isdir(domain):
                    continue
                problem = os.path.join(domain,"test")
                for seed in range(self.seeds):
                    for heur in heuristics:
                        yield RLEvaluationExperiment(
                            uuid4="j"+str(uuid.uuid4()),
                            domain=domain, problem=problem, seed=seed, heur=heur,
                            cluster=self.cluster, root=self.root, dry=self.dry)
    def run(self):
        pass


class NonRLExperiments(AbstractSetting):
    seeds = luigi.IntParameter(default=1)
    def requires(self):
        for domset in domain_sets:
            for domain in glob.glob(os.path.join(domset,"*")):
                if not os.path.isdir(domain):
                    continue
                problem = os.path.join(domain,"test")
                for seed in range(self.seeds):
                    for heur in heuristics:
                        yield NonRLEvaluationExperiment(
                            uuid4="j"+str(uuid.uuid4()),
                            domain=domain, problem=problem, seed=seed, heur=heur,
                            cluster=self.cluster, root=self.root+"-nonRL", dry=self.dry)
    def run(self):
        pass


class AllExperiments(AbstractSetting):
    seeds = luigi.IntParameter(default=1)
    def requires(self):
        return [
            RLExperiments(seeds=self.seeds),
            NonRLExperiments(seeds=self.seeds),
        ]
    def run(self):
        pass



if __name__ == '__main__':
    try:
        luigi.run()
    except:
        stacktrace.format()

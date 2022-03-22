# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os

import pandas as pd


def load_config_and_run_dirs(results_dir):
    results_dir = os.path.expanduser(results_dir)
    configs_path = os.path.join(results_dir, "configs.csv")
    configs = [config.to_dict() for _, config in pd.read_csv(configs_path).iterrows()]
    run_dirs = [os.path.join(results_dir, str(config["run_id"])) for config in configs]
    return configs, run_dirs


def parse_all_results(dir_path, iteration, filename):
    configs, run_dirs = load_config_and_run_dirs(dir_path)
    dfs = []
    for config, run_path in zip(configs, run_dirs):
        try:
            df = pd.read_csv(os.path.join(run_path, str(iteration), filename))
            for name, value in config.items():
                df[name] = value
            dfs.append(df)
        except FileNotFoundError:
            print(f"Skipping {run_path}")

    return pd.concat(dfs, ignore_index=True)

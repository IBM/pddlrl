#!/bin/bash

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

# this is a script for counting the number of training and testing instances in each domain.

export SHELL=/bin/bash

per-domain (){
    mode=$1
    dir=$2
    ls $dir/$mode/*.pddl | wc -l
}

export -f per-domain

fn (){
    parallel --keep-order per-domain $1 ::: $(domains)
}

domains (){
    basename=${1:-false}
    if $basename
    then
        ls -d {erez-domains2,cherry-pick2}/*/ | sort | parallel --keep-order basename
    else
        ls -d {erez-domains2,cherry-pick2}/*/ | sort
    fi
}

echo "domains	#test	#train"

paste \
    <(domains true) \
    <(fn test) \
    <(fn train) \
    | tee count-problems.csv


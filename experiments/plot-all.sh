#!/bin/bash

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

set -x
set -e

./collect.sh > collect.csv
parallel -v "./count-win-loss-eval.ros {} | column -t | tee wins-{4}-{3}.csv" ::: gbfs ::: "*" ::: blocks ferry gripper logistics satellite miconic parking visitall ::: nov9
parallel -v ./plot-objs.ros ::: gbfs ::: "*" ::: blocks ferry gripper logistics satellite miconic parking visitall ::: nov9
parallel -v ./plot-eval.ros ::: gbfs ::: "*" ::: blocks ferry gripper logistics satellite miconic parking visitall ::: nov9
parallel -v ./plot-cumulative-goal.ros ::: nov9  ::: blocks ferry gripper logistics satellite miconic parking visitall ::: blind hff hadd

# request from Clement for the presentation
# parallel -v ./plot-eval-presentation.ros ::: gbfs ::: hadd ::: blocks logistics ::: nov9


pdfunite objs-*.pdf objs.pdf
pdfunite eval-*.pdf eval.pdf
pdfunite cumulative-goals-*.pdf cumulative-goals.pdf

(
    echo "ignore experiment track domain heuristics seed ignore ignore goalsreached"
    parallel ./plot-cumulative-goal-justnumber.ros ::: nov9  ::: blocks ferry gripper logistics satellite miconic parking visitall ::: blind hff hadd
) > cumulative-goals.csv

./plot-cumulative-goals-rank-sum-test.ros | tee cumulative-goals-rank-sum-test.csv

# cp objs-*.pdf eval-*.pdf cumulative-goals-*.pdf cumulative-goals.csv ~/repos/papers/2021-prl-pddlrl-paper/img/static/


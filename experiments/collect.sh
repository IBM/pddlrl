#!/bin/bash

# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

per-config(){
    dir=$1
    domain=$2
    heur=$3
    root=$dir/*/$domain/$heur-True-gbfs
    i=0
    for d in $root/*/
    do
        count=$(ls $d/*.plan | wc -l)
        echo $dir $domain $heur $count $i
        i=$(($i+1))
    done
}

export -f per-config
SHELL=/bin/bash

(
    parallel --keep-order per-config \
             ::: nov9 nov9-nonRL \
             ::: blocks ferry gripper logistics satellite miconic parking visitall \
             ::: blind hadd hff
) | column -t

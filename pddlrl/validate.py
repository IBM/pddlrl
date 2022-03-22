# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import os.path
import subprocess

directory, basename = os.path.split(__file__)
VAL = os.path.join(directory, "../VAL/build/linux64/Release/bin/Validate")

def validate(domainfile=None, problemfile=None, planfile=None):
    args = [VAL, domainfile, problemfile, planfile]
    args = [ arg for arg in args if arg ]
    return subprocess.run(args)


def arrival(domainfile=None, problemfile=None, planfile=None):
    args = ["arrival", domainfile, problemfile, planfile, "/dev/null"]
    args = [ arg for arg in args if arg ]
    return subprocess.run(args)


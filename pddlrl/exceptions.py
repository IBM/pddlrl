# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

class InvalidHyperparameterError(AssertionError):
    """Raised when the hyperparameter is not valid"""
    pass


import signal
class SignalInterrupt(Exception):
    """Raised when a signal handler was invoked"""
    def __init__(self,signal,frame):
        print("Received Signal",signal)
        self.signal = signal
        self.frame  = frame
        raise self
    pass

signal.signal(signal.SIGUSR2,SignalInterrupt)

class HyperparameterGenerationError(Exception):
    """Raised when the hyperparameter generation failed """
    pass


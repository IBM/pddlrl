# This file is a part of PDDLRL project.
# Copyright (c) 2020 Clement Gehring (clement@gehring.io)
# Copyright (c) 2021 Masataro Asai (guicho2.71828@gmail.com, masataro.asai@ibm.com), IBM Corporation

import collections


class SimpleReplayBuffer(object):

    def __init__(self, capacity):
        self._buffer = collections.deque(maxlen=capacity)
        self._unseen_samples = collections.deque()

    def put(self, data):
        self._unseen_samples.append(data)

    def sample(self, rng, batch_size):
        if len(self._buffer):
            # always include unseen samples
            states = list(self._unseen_samples)
        else:
            assert len(self._unseen_samples)
            # if buffer is empty, fill it with unseen samples and sample random
            # batch from the newly populated buffer
            states = []
            self._flush_unseen()

        # Randomly sample past states in order to have a batch of
        # size `batch_size`.
        num_replay_samples = batch_size - len(states)
        if 0 < num_replay_samples <= len(self._buffer):
            idx = rng.choice(len(self._buffer), num_replay_samples, replace=False)
            states = states + [self._buffer[i] for i in idx]

        self._flush_unseen()
        return states

    def _flush_unseen(self):
        self._buffer.extend(self._unseen_samples)
        self._unseen_samples.clear()

    def __len__(self):
        return len(self._buffer) + len(self._unseen_samples)

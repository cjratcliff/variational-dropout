from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def clip(x):
	return tf.clip_by_value(x, -8, 8)


def get_minibatches_idx(n, minibatch_size, shuffle):
	# Used to shuffle the dataset at each iteration.
	idx_list = np.arange(n, dtype="int32")

	if shuffle:
		np.random.shuffle(idx_list)

	minibatches = []
	minibatch_start = 0
	for i in range(n // minibatch_size):
		minibatches.append(idx_list[minibatch_start : minibatch_start + minibatch_size])
		minibatch_start += minibatch_size

	if minibatch_start != n:
		# Make a mini-batch out of what is left
		minibatches.append(idx_list[minibatch_start:])

	return minibatches

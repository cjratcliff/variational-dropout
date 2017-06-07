from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from utils import clip


def eval_reg(log_sigma2, W):
	# Approximates the negative of the KL-divergence according to eqn 14.
	# This is a key part of the loss function (see eqn 3).
	k1, k2, k3 = 0.63576, 1.8732, 1.48695
	C = -k1
	log_alpha = clip(log_sigma2 - tf.log(W**2))
	mdkl = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.log1p(tf.exp(-log_alpha)) + C
	return -tf.reduce_sum(mdkl)
	

def ell(pred, targets):
	# Expected log-likelihood, the first part of the loss function.
	# Approximated by eqn 4.
	eps = 1e-8
	pred = tf.clip_by_value(pred, eps, 1-eps)
	return -tf.reduce_sum(tf.log(pred)*targets)


def reg(W, log_sigma2):
    return sum([eval_reg(w,s) for (w,s) in zip(W,log_sigma2)])


def sgvlb(predictions, targets, W, log_sigma2, batch_size, rw=None):
	# Stochastic gradient variational lower bound
	# See eqns 3 and 4
	if rw is None:
		rw = tf.Variable(tf.constant(1.0)) ### Variable? trainable?
	num_batches = 60000/batch_size # num_samples / batch_size = N/M, eqn 4 
	loss = num_batches * ell(predictions, targets)
	loss -= rw*reg(W,log_sigma2) # Subtract the KL-divergence term
	### Gets a higher accuracy on MNIST when rw is removed?
	return loss

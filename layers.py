import tensorflow as tf
import numpy as np

eps = 1e-8

def clip(x):
	return tf.clip_by_value(x, -8, 8)
	
	
def weight_matrix(dims):
	with tf.variable_scope("weight"):
		d = np.sqrt(6.0)/np.sqrt(sum(dims))
		v = tf.Variable(tf.random_uniform(shape=dims, minval=-d, maxval=d))
		tf.add_to_collection('weights', v)
		return v


def bias_vector(num_out):
	with tf.variable_scope("bias"):
		v = tf.Variable(tf.zeros(shape=[num_out]))
		tf.add_to_collection('biases', v)
		return v


class FCVarDropout():
	def __init__(self, num_in, num_out, nonlinearity=tf.nn.relu, ard_init=-10):
		self.reg = True
		self.W = weight_matrix([num_in, num_out])
		self.b = bias_vector(num_out)
		self.nonlinearity = nonlinearity
		# ARD is Automatic Relevance Determination
		self.log_sigma2 = tf.Variable(ard_init*tf.ones([num_in, num_out]), 'ls2')


	def get_output(self, x, deterministic, train_clip=False, thresh=3):
		# Alpha is the dropout rate
		log_alpha = clip(self.log_sigma2 - tf.log(self.W**2 + eps))
		
		# Values of log_alpha that are above the threshold
		clip_mask = tf.greater_equal(log_alpha, thresh)

		def true_path(): # For inference
			# If log_alpha >= thresh, return 0
			# If log_alpha < thresh, return tf.matmul(x,self.W)
			return tf.matmul(x, tf.where(clip_mask, tf.zeros_like(self.W), self.W))
		
		def false_path(): # For training
			# Sample from a normal distribution centred on tf.matmul(x,W)
			# and with variance roughly proportional to the size of tf.matmul(x,W)*tf.exp(log_alpha)
			W = self.W
			if train_clip:
				raise NotImplementedError
			mu = tf.matmul(x,W)
			si = tf.matmul(x*x, tf.exp(log_alpha) * self.W * self.W)
			si = tf.sqrt(si + eps)
			return mu + tf.random_normal(tf.shape(mu), mean=0.0, stddev=1.0) * si
			
		h = tf.cond(deterministic, true_path, false_path)
		return self.nonlinearity(h + self.b)
		
		
class Conv2DVarDropOut():
	def __init__(self, kernel_shape, strides=(1,1,1,1), padding='VALID', nonlinearity=tf.nn.relu, ard_init=-10):
		if len(strides) == 2:
			strides = [1,strides[0],strides[1],1]
		
		self.W = weight_matrix(kernel_shape)
		self.strides = strides
		self.padding = padding
		self.nonlinearity = nonlinearity
		self.log_sigma2 = tf.Variable(ard_init*tf.ones(kernel_shape), 'ls2')

	def get_output(self, x, deterministic, train_clip=False, thresh=3):
		# Alpha is the dropout rate
		log_alpha = clip(self.log_sigma2 - tf.log(self.W**2 + eps))
		
		# Values of log_alpha that are above the threshold
		clip_mask = tf.greater_equal(log_alpha, thresh)

		def true_path(): # For inference
			return tf.nn.conv2d(x, tf.where(clip_mask, tf.zeros_like(self.W), self.W), strides=self.strides, padding=self.padding)
		
		def false_path(): # For training
			W = self.W
			if train_clip:
				raise NotImplementedError
			mu = tf.nn.conv2d(x, W, strides=self.strides, padding=self.padding)
			si = tf.nn.conv2d(x*x, tf.exp(log_alpha) * W*W, strides=self.strides, padding=self.padding)
			si = tf.sqrt(si + eps)
			return mu + tf.random_normal(tf.shape(mu), mean=0.0, stddev=1.0) * si
		
		h = tf.cond(deterministic, true_path, false_path)
		return self.nonlinearity(h) ### + self.b?



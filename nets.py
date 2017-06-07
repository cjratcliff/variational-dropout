from __future__ import division
from __future__ import print_function
import time

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dropout

from layers import FCVarDropout, Conv2DVarDropout
from loss import sgvlb
from utils import get_minibatches_idx, clip

batch_size = 32
eps = 1e-8


class Net():
	def fit(self,X,y,sess):
		max_epochs = 20
		
		# Split into training and validation sets
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
		
		for epoch in range(max_epochs):
			start = time.time()
			train_indices = get_minibatches_idx(len(X_train), batch_size, shuffle=True)
			print("\nEpoch %d" % (epoch+1))
			
			train_accs = []
			for c,it in enumerate(train_indices):
				batch_train_x = [X_train[i] for i in it]
				batch_train_y = [y_train[i] for i in it]
				feed_dict = {self.x: batch_train_x, 
							self.y: batch_train_y,
							self.deterministic: False}
				_,acc = sess.run([self.train_step,self.accuracy], feed_dict)
				train_accs.append(acc)
				#print(c,len(train_indices),acc)
			print("Training accuracy: %.3f" % np.mean(train_accs))		
			val_pred = self.predict(X_val,sess)
			y = np.argmax(y_val,axis=1)
			val_acc = np.mean(np.equal(val_pred,y))
			print("Val accuracy: %.3f" % val_acc)
			print("Time taken: %.3fs" % (time.time() - start))
		return
		
		
	def predict(self,X,sess):
		indices = get_minibatches_idx(len(X), batch_size, shuffle=False)
		pred = []
		for i in indices:
			batch_x = [X[j] for j in i]
			feed_dict = {self.x: batch_x, 
						self.deterministic: True}		
			pred_batch = sess.run(self.pred, feed_dict)
			pred.append(pred_batch)
		pred = np.concatenate(pred,axis=0)
		pred = np.argmax(pred,axis=1)
		pred = np.reshape(pred,(-1))
		return pred	


class LeNet(Net):
	def __init__(self, img_size, num_channels, num_classes):
		self.x = tf.placeholder(tf.float32, [None,img_size,img_size,num_channels], 'x')
		self.y = tf.placeholder(tf.float32, [None,num_classes], 'y')
		self.deterministic = tf.placeholder(tf.bool, name='d')	

		h = Conv2D(32, kernel_size=(3,3),
						 activation='relu',
						 input_shape=[None,img_size,img_size,num_channels])(self.x)
		h = Conv2D(64, (3, 3), activation='relu')(h)

		h = MaxPooling2D(pool_size=(2,2))(h)	
		h = Flatten()(h)
		
		h = Dense(500, activation='relu')(h)
		self.pred = Dense(num_classes, activation='softmax')(h)
		pred = tf.clip_by_value(self.pred,eps,1-eps)

		loss = -tf.reduce_sum(tf.log(pred)*self.y)

		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		
		optimizer = tf.train.AdamOptimizer()
		self.train_step = optimizer.minimize(loss)	
	

class LeNetVarDropout(Net):
	def __init__(self, img_size, num_channels, num_classes):
		
		self.x = tf.placeholder(tf.float32, [None,img_size,img_size,num_channels], 'x')
		self.y = tf.placeholder(tf.float32, [None,num_classes], 'y')
		self.deterministic = tf.placeholder(tf.bool, name='d')
		d = self.deterministic

		h = Conv2DVarDropout(num_channels, 32, (3,3), strides=(1,1))(self.x,d)
		h = Conv2DVarDropout(32, 64, (3,3), strides=(1,1))(h,d)	
		h = MaxPooling2D(pool_size=(2,2))(h)
		
		h = Flatten()(h)
		
		if num_channels == 1:
			#h = FCVarDropout(9216,500)(h,d)
			h = Dense(500)(h)
		elif num_channels == 3:
			h = FCVarDropout(12544,500)(h,d)
		else:
			raise NotImplementedError

		#self.pred = FCVarDropout(500,num_classes,tf.nn.softmax)(h,d)
		self.pred = Dense(num_classes,activation='softmax')(h)
		pred = tf.clip_by_value(self.pred,eps,1-eps)
		
		W = tf.get_collection('W')
		log_sigma2 = tf.get_collection('log_sigma2')
		loss = sgvlb(pred, self.y, W, log_sigma2, batch_size, rw=1)
		
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		
		optimizer = tf.train.AdamOptimizer()
		self.train_step = optimizer.minimize(loss)


class VGG(Net):
	def __init__(self, img_size, num_channels, num_classes, dropout_prob=0.0):
		# Based on https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py

		self.x = tf.placeholder(tf.float32, [None,img_size,img_size,num_channels], 'x')
		self.y = tf.placeholder(tf.float32, [None,num_classes], 'y')
		self.deterministic = tf.placeholder(tf.bool, name='d')
		d = self.deterministic
		phase = tf.logical_not(d)
		
		def conv_bn(h, num_filters, phase):
			h = Conv2D(num_filters, (3,3), padding='same')(h) # Linear
			h = tf.contrib.layers.batch_norm(h, center=True, scale=False, is_training=phase)
			return tf.nn.relu(h)
		
		# Block 1
		h = conv_bn(self.x,64,phase)
		h = conv_bn(h,64,phase)
		h = MaxPooling2D((2, 2), strides=(2,2))(h)

		# Block 2
		h = conv_bn(h,128,phase)
		h = conv_bn(h,128,phase)
		h = MaxPooling2D((2, 2), strides=(2,2))(h)

		# Block 3
		h = conv_bn(h,256,phase)
		h = conv_bn(h,256,phase)
		h = conv_bn(h,256,phase)
		h = MaxPooling2D((2,2), strides=(2,2))(h)

		# Block 4
		h = conv_bn(h,512,phase)
		h = conv_bn(h,512,phase)
		h = conv_bn(h,512,phase)
		h = MaxPooling2D((2,2), strides=(2,2))(h)
		
		# Block 5
		h = conv_bn(h,512,phase)
		h = conv_bn(h,512,phase)
		h = conv_bn(h,512,phase)
		h = MaxPooling2D((2,2), strides=(2,2))(h)

		h = Flatten()(h)

		self.pred = Dense(num_classes, activation='softmax')(h)
		pred = tf.clip_by_value(self.pred,eps,1-eps)

		loss = -tf.reduce_sum(tf.log(pred)*self.y)

		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		
		optimizer = tf.train.AdamOptimizer(0.001)
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			# Ensures that we execute the update_ops before performing the train_step
			self.train_step = optimizer.minimize(loss)	
	

class VGGVarDropout(Net):
	def __init__(self, img_size, num_channels, num_classes):
		# Based on https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py

		self.x = tf.placeholder(tf.float32, [None,img_size,img_size,num_channels], 'x')
		self.y = tf.placeholder(tf.float32, [None,num_classes], 'y')
		self.deterministic = tf.placeholder(tf.bool, name='d')
		d = self.deterministic
		phase = tf.logical_not(d)
		
		def conv_bn(h, filters_in, filters_out, d, phase):
			h = Conv2DVarDropout(filters_in, filters_out, (3,3), padding='SAME', nonlinearity=tf.identity)(h,d) # Linear
			h = tf.contrib.layers.batch_norm(h, center=True, scale=False, is_training=phase)
			return tf.nn.relu(h)

		# Block 1
		h = conv_bn(self.x, num_channels, 64, d, phase)
		h = conv_bn(h, 64, 64, d, phase)
		h = MaxPooling2D((2, 2), strides=(2,2))(h)

		# Block 2
		h = conv_bn(h, 64, 128, d, phase)
		h = conv_bn(h, 128, 128, d, phase)
		h = MaxPooling2D((2, 2), strides=(2,2))(h)

		# Block 3
		h = conv_bn(h, 128, 256, d, phase)
		h = conv_bn(h, 256, 256, d, phase)
		h = conv_bn(h, 256, 256, d, phase)
		h = MaxPooling2D((2,2), strides=(2,2))(h)

		# Block 4
		h = conv_bn(h, 256, 512, d, phase)
		h = conv_bn(h, 512, 512, d, phase)
		h = conv_bn(h, 512, 512, d, phase)
		h = MaxPooling2D((2, 2), strides=(2, 2))(h)

		# Block 5
		h = conv_bn(h, 512, 512, d, phase)
		h = conv_bn(h, 512, 512, d, phase)
		h = conv_bn(h, 512, 512, d, phase)
		h = MaxPooling2D((2, 2), strides=(2, 2))(h)
		
		h = Flatten()(h)
		self.pred = FCVarDropout(512, num_classes, tf.nn.softmax)(h,d)

		pred = tf.clip_by_value(self.pred,eps,1-eps)
		
		W = tf.get_collection('W')
		log_sigma2 = tf.get_collection('log_sigma2')
		loss = sgvlb(pred, self.y, W, log_sigma2, batch_size)
		
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		
		optimizer = tf.train.AdamOptimizer(0.0001)
		
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			# Ensures that we execute the update_ops before performing the train_step
			self.train_step = optimizer.minimize(loss)	

from __future__ import division
from __future__ import print_function
import time

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from layers import FCVarDropout, Conv2DVarDropOut
from loss import sgvlb
from utils import get_minibatches_idx, clip

batch_size = 32


class Net():
	def fit(self,X,y,sess):
		max_epochs = 5
		
		# Split into training and validation sets
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
		
		for epoch in range(max_epochs):
			start = time.time()
			train_indices = get_minibatches_idx(len(X_train), batch_size, shuffle=True)
			print("\nEpoch %d" % (epoch+1))
			
			train_accs = []
			for it in train_indices:
				batch_train_x = [X_train[i] for i in it]
				batch_train_y = [y_train[i] for i in it]
				feed_dict = {self.x: batch_train_x, 
							self.y: batch_train_y,
							self.deterministic: False}
				_,acc = sess.run([self.train_step,self.accuracy], feed_dict)
				train_accs.append(acc)
			
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

		vd = Conv2DVarDropOut([3,3,1,32], (1,1))
		h = vd(self.x,self.deterministic)
		
		vd = Conv2DVarDropOut([3,3,32,64], (1,1))
		h = vd(h,self.deterministic)
		
		h = MaxPooling2D(pool_size=(2,2))(h)
		
		h = Flatten()(h)
		
		if num_channels == 1:
			vd = FCVarDropout(9216,500,tf.nn.relu)
		elif num_channels == 3:
			vd = FCVarDropout(12544,500,tf.nn.relu)
		else:
			raise NotImplementedError
			
		h = vd(h,self.deterministic)
		
		vd = FCVarDropout(500,num_classes,tf.nn.softmax)
		self.pred = vd(h,self.deterministic)
		
		eps = 1e-8
		pred = tf.clip_by_value(self.pred,eps,1-eps)
		
		W = tf.get_collection('W')
		log_sigma2 = tf.get_collection('log_sigma2')
		loss = sgvlb(pred, self.y, W, log_sigma2, batch_size)
		
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		
		optimizer = tf.train.AdamOptimizer()
		self.train_step = optimizer.minimize(loss)


class VGG(Net):
	def __init__(self, img_size, num_channels, num_classes):
		pass


from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from sklearn.preprocessing import LabelBinarizer

from nets import LeNet, LeNetVarDropout, VGG, VGGVarDropout

sess = tf.Session()


def main():
	dataset = 'cifar10' # mnist, cifar10, cifar100
	
	# Load the data
	# It will be downloaded first if necessary
	if dataset == 'mnist':
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		img_size = 28
		num_classes = 10
		num_channels = 1
	elif dataset == 'cifar10':
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
		img_size = 32
		num_classes = 10
		num_channels = 3
	elif dataset == 'cifar100':
		(X_train, y_train), (X_test, y_test) = cifar100.load_data()
		img_size = 32
		num_classes = 100
		num_channels = 3	
	
	lb = LabelBinarizer()
	lb.fit(y_train)
	y_train_one_hot = lb.transform(y_train)
	y_test_one_hot = lb.transform(y_test)
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train = np.reshape(X_train,[-1,img_size,img_size,num_channels])
	X_test = np.reshape(X_test,[-1,img_size,img_size,num_channels])
	X_train /= 255
	X_test /= 255
	
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	m = VGGVarDropout(img_size,num_channels,num_classes)
	sess.run(tf.global_variables_initializer())

	m.fit(X_train,y_train_one_hot,sess)
	
	pred = m.predict(X_test,sess)
	y_test = np.squeeze(y_test)
	acc = np.mean(np.equal(y_test,pred))
	print("\nTest accuracy: %.3f" % acc)
	
	
	
if __name__ == "__main__":
	main()

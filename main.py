import tensorflow as tf
import numpy as np
from keras.datasets import mnist, cifar10, cifar100
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time

from layers import FCVarDropout, clip#, Conv2DVarDropOut


batch_size = 32
sess = tf.Session()

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


def sgvlb(predictions, targets, W, log_sigma2, rw=None, train_clip=False, thresh=3):
	# Stochastic gradient variational lower bound
	# See eqns 3 and 4
	if rw is None:
		rw = tf.Variable(tf.constant(1.0)) ### Variable? trainable?
	num_batches = int(60000/batch_size) # num_samples / batch_size = N/M, eqn 4 
	loss = num_batches * ell(predictions, targets)
	loss -= rw*reg(W,log_sigma2) # Subtract the KL-divergence term
	### Gets a higher accuracy on MNIST when rw is removed
	return loss


class LeNet():
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
		
		if num_channels == 1:
			vd = FCVarDropout(9216,128,tf.nn.relu)
		elif num_channels == 3:
			vd = FCVarDropout(12544,128,tf.nn.relu)
		else:
			raise NotImplementedError
			
		h = vd.get_output(h,self.deterministic)
		
		self.pred = Dense(num_classes, activation='softmax')(h)
		
		eps = 1e-8
		pred = tf.clip_by_value(self.pred,eps,1-eps)
		
		W = [vd.W]
		log_sigma2 = [vd.log_sigma2]
		loss = sgvlb(pred, self.y, W, log_sigma2)
		
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.pred, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		
		optimizer = tf.train.AdamOptimizer()
		self.train_step = optimizer.minimize(loss)


	def fit(self,X,y):
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
			val_pred = self.predict(X_val)
			y = np.argmax(y_val,axis=1)
			val_acc = np.mean(np.equal(val_pred,y))
			print("Val accuracy: %.3f" % val_acc)
			print("Time taken: %.3fs" % (time.time() - start))
		return
		
		
	def predict(self,X):
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


def main():
	dataset = 'mnist' # mnist, cifar10, cifar100
	
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

	m = LeNet(img_size,num_channels,num_classes)
	sess.run(tf.global_variables_initializer())

	m.fit(X_train,y_train_one_hot)
	
	pred = m.predict(X_test)
	acc = np.mean(np.equal(y_test,pred))
	print("\nTest accuracy: %.3f" % acc)
	
	
	
if __name__ == "__main__":
	main()

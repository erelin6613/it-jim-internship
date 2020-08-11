import os
import numpy as np
from scipy.spatial import distance
import cv2
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from tqdm import tqdm
import pandas as pd
# from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier #, VotingClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (LogisticRegression, 
	RidgeClassifier, PassiveAggressiveClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, 
	precision_score, recall_score, 
	log_loss, f1_score,)
from sklearn.ensemble import RandomForestClassifier #, HistGradientBoostingClassifier
from sklearn.decomposition import KernelPCA

from utils import *


"""
As always I wish I had more time playing around,
sklearn usually is out of my radar when I deal with
computer vision problems :)

Still, here I tried grayscale images, BGR images, gist
and gistograms. The best performed well... raw
BGR images despite my huge hopes for gist
(I hope I had no mistakes)

Although results are better than shot in the dark,
it for sure would be better with CNNs.

P.S. Multiclass problems I rarely solve with sklearn
API so I am sorry I could not win struggle with
their metrics. I tried roc auc score and log loss
both of which did not work
"""

class Dataset:

	def __init__(self, root_dir, feature=None):
		"""
		Organizing dataset.
		
		:param: root_dir - directory with subfolders named as 
				classes (specified in utils.py)
		:param: feature - wheater to use another function to
				transform images

		"""
		self.root_dir = root_dir
		self.classes = {code: name for name, code in enumerate(list(folders.keys()))}
		self.feature = feature
		self.size, self.input_shape = self.figure_out_input()
		self.set = self.get_set()

	def __len__(self):
		return self.size

	def figure_out_input(self):
		"""
		Method to set helpful attributes
		(not used here)
		"""
		sample_dir = os.path.join(self.root_dir, list(
				self.classes.keys())[0])

		size = len(os.listdir(sample_dir))*len(self.classes)
		input_shape = load_image(
			os.path.join(sample_dir, os.listdir(
				sample_dir)[0]), False).shape
		return size, input_shape

	def get_set(self):
		"""
		Methods giving a tuple of 
		(loaded/transformed images, labels)
		both as a numpy array
		"""
		images = []
		labels = []
		i = 0
		for c, n in self.classes.items():
			files = [os.path.join(
				self.root_dir, c, x) for x in os.listdir(
				os.path.join(self.root_dir, c))]
			for file in files:
				labels.append(n)
				if self.feature is None:
					arr = load_image(file, False)
				else:
					arr = self.feature(file)
				images.append(arr.reshape(-1, ).astype(np.float32))
		c = list(zip(images, labels))
		random.shuffle(c)
		images, labels = zip(*c)
		return np.array(images), np.array(labels).T

	@staticmethod
	def scale(img):
		return img/255.0


def reduce_dims_pca(imgs, components=int(84*84/32)):
	pca = KernelPCA(n_components=components).fit(imgs)
	return pca

def gist(path):
	"""
	I am sorry for `borrowing` the code but
	I have tried to implement this method myself
	and yet I missunderstood a few bits of it.
	Moreover, it still did not work
	"""
	img = load_image(path)
	gabor_pic = gabor(img)
	resized_pic = []
	for gb in gabor_pic:
		f = []
		for i in range(4):
			for j in range(4):
				f.append(np.mean(gb[21 * i: 21 * (i+1), 21 * j: 21 * (j+1)]))
		f = np.reshape(f, (4, 4))
		resized_pic.append(f)

	resized_pic = np.float32(resized_pic)
	resized_pic = np.reshape(resized_pic, (512, ))

	return resized_pic

def get_histogram(path):

	# comparing historgrams of the images
	img = load_image(path)
	img_hist = cv2.calcHist(
		img, [0, 1], None, [255,255], None)
	return img_hist

def gabor(img):

	scale = np.array([3, 7, 11, 15])
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	filtered = []
	# mean amplitude as a feature vector
	mean_amp = []
	# eigenvalues as feature vectors
	eig1 = []
	eig2 = []
	for sc in scale:
		for theta in np.arange(0, np.pi, np.pi / 8):
			kern = cv2.getGaborKernel((sc, sc), 5, theta, 10, 1, 0, cv2.CV_32F)
			kern /= 1.5 * kern.sum()

			filtered.append(cv2.filter2D(img, -1, kern))

	return filtered

def plot_conf_matrix(
	y_true, y_pred, classes=None):
	"""Ploting confusion matrix"""

	conf_matrix = confusion_matrix(
		y_true, y_pred) #, labels=list(classes))
	sns.heatmap(conf_matrix)

	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	# moving and organizing data
	# organize_folders(
	#	os.path.join('..', 'Task4','dataset'),
	#	'dataset')

	classes = {x: i for i, x in enumerate(folders.keys())}

	train_set = Dataset('dataset/train') #, feature=gist)
	val_set =  Dataset('dataset/val') #, feature=gist)
	test_set =  Dataset('dataset/test') #, feature=gist)

	X_train, y_train = train_set.set
	X_val, y_val = val_set.set
	X_test, y_test = test_set.set

	print('squeezing dimentions...')
	pca = reduce_dims_pca(np.vstack([X_train, X_val, X_test]))
	X_train = pca.transform(X_train)
	X_val = pca.transform(X_val)
	X_test = pca.transform(X_test)

	print('training base learners...')
	lr = LogisticRegression(C=100/len(X_train), 
		solver='saga', penalty='l1', tol=0.01,
		multi_class='ovr').fit(X_train, y_train)
	rf = RandomForestClassifier(
		n_estimators=250).fit(X_train, y_train)
	dt = DecisionTreeClassifier(
		criterion='entropy', max_features='sqrt').fit(
		X_train, y_train)

	estimators = [('lr', lr), ('rf', rf), ('dt', dt)]

	print('stacking...')
	clf = StackingClassifier(
		estimators, final_estimator=RandomForestClassifier(
			n_estimators=500))
	clf = clf.fit(X_train, y_train)
	y_val_pred = clf.predict(X_val)
	#print(y_val_pred)

	print('validation: \nf1_score:', 
		f1_score(y_val, y_val_pred, average='macro'),
		'\nprecision score:', precision_score(
			y_val, y_val_pred, average='macro'),
		'\nrecall score:', recall_score(
			y_val, y_val_pred, average='macro'))

	plot_conf_matrix(
		y_val, y_val_pred, classes.keys())

	y_test_pred = clf.predict(X_test)

	print('inference: \nf1_score:', 
		f1_score(y_test, y_test_pred, average='macro'),
		'\nprecision score:', precision_score(
			y_test, y_test_pred, average='macro'),
		'\nrecall score:', recall_score(
			y_test, y_test_pred, average='macro'))

	plot_conf_matrix(
		y_test, y_test_pred, classes.keys())
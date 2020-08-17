import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def accuracy(outputs, labels, return_probs=False):
	"""Calculate accuracy for a batch"""
	preds = torch.argmax(outputs, dim=1)
	if return_probs:
		# for test setting we will need probabilities
		return F.softmax(outputs, dim=1)
	return torch.tensor(
		torch.sum(preds == labels).item()/len(preds))

def conv_block(in_channels, out_channels):
	"""Quick modularization of convolutional
	repeated blocks"""
	layers = [nn.Conv2d(
		in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels)]
	return nn.Sequential(*layers)


def classifier(in_channels, classes=len(folders)):
	"""Module used as classification layer"""
	return nn.Sequential(nn.Flatten(),
		nn.Linear(in_channels, classes))


class BaseModel(nn.Module):
	"""Base model. No matter what architecture we use
	some steps will be repetative (loss calculation,
	checkpointing, metrics tracking etc), hence we 
	can create some model which will preserve attributes, 
	methods of nn.Module but we will define some 
	helper-methods for convinience. From this model
	all our next ones will inheret this functionality."""
	
	def training_step(self, batch):
		try:
			# try/except blocks attributed to my relactance to
			# rewrite __init__ for base model :)
			assert len(self.checkpoint)>0
		except Exception:
			self.checkpoint = {'val_loss': np.inf, 'train_loss': np.inf}
		x, y = batch 
		out = self(x)
		loss = F.cross_entropy(out, y)
		return loss
	
	def validation_step(self, batch):
		x, y = batch 
		out = self(x)
		loss = F.cross_entropy(out, y)
		acc = accuracy(out, y)
		return {'val_loss': loss.detach(), 'val_acc': acc}
		
	def validation_epoch_end(self, outputs):
		batch_losses = [x['val_loss'] for x in outputs]
		epoch_loss = torch.stack(batch_losses).mean()
		batch_accs = [x['val_acc'] for x in outputs]
		epoch_acc = torch.stack(batch_accs).mean()
		return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
	
	def epoch_end(self, epoch, result):

		try:
			if self.checkpoint['val_loss'] > result['val_loss']:
				self.checkpoint = result

		except Exception:
			self.checkpoint = result

		print(f'Epoch: {epoch}, \
			train_loss: {result["train_loss"]}, \
			\nval_loss: {result["val_loss"]}, \
			val_acc: {result["val_acc"]}')
		return result

class FC_model(BaseModel):
	"""Fully-connected model. Simply flatten input,
	drop some features, feed-forward whats left.
	A note on activation: I usually try to use 
	leaky relu to avoid some important (potentially)
	nodes to loose any value just due to negative
	initialization. For this task it actually made
	no difference."""

	def __init__(self, in_channels, image_shape=84, classes=len(folders)):
		super().__init__()

		self.classes = classes
		self.in_channels = in_channels
		self.image_shape = image_shape

		self.input = nn.Linear(in_channels*self.image_shape**2, 
			self.image_shape*5)
		self.drop = nn.Dropout(0.3)
		self.hidden = nn.Linear(self.image_shape*5, self.image_shape*2)
		self.flat = nn.Flatten()
		self.last_lin = nn.Linear(
			self.image_shape*2, self.image_shape)
		self.cl = classifier(self.image_shape, 
			self.classes)

	def forward(self, x):
		x = self.flat(x)
		x = F.leaky_relu(self.input(x))
		x = self.drop(x)
		x = F.leaky_relu(self.hidden(x))
		x = self.flat(x)
		x = F.leaky_relu(self.last_lin(x))
		return self.cl(x)

class ConvModel(BaseModel):
	"""Convolutional model. General flow
	is convolve -> normalize batch -> 
	drop features -> pool, repeat"""

	def __init__(self, in_channels, image_shape=84, classes=len(folders)):
		super().__init__()

		self.classes = classes
		self.input = conv_block(in_channels, in_channels*5)
		self.drop = nn.Dropout(0.3)
		self.conv = conv_block(in_channels*5, in_channels*8)
		self.bottleneck = conv_block(in_channels*8, in_channels)
		self.pool = nn.MaxPool2d(2)
		self.ft_map = conv_block(in_channels, 1)
		self.flat = nn.Flatten()
		flat_size = int(math.floor(image_shape/2**3)**2)
		self.cl = classifier(flat_size, self.classes)

	def forward(self, x):
		x = F.leaky_relu(self.input(x))
		x = self.drop(x)
		x = F.leaky_relu(self.conv(x))
		x= self.pool(x)
		x = F.leaky_relu(self.bottleneck(x))
		x = self.pool(x)
		x = self.drop(x)
		x = F.leaky_relu(self.ft_map(x))
		x = self.pool(x)
		x = self.flat(x)
		return self.cl(x)

class ResidBlock(nn.Module):
	"""Residual block deserves separete module due to its
	non-sequential nature i.e. convolved input should be
	a sum with the original input. As far as I know :)
	"""
	def __init__(self, in_channels):
		super().__init__()

		self.conv = nn.Conv2d(
			in_channels=in_channels, 
			out_channels=in_channels, 
			kernel_size=3, 
			stride=1, 
			padding=1)
		self.norm = nn.BatchNorm2d(in_channels)

	def forward(self, x):
		out = F.leaky_relu(self.conv(x))
		out = self.norm(out)
		out = F.leaky_relu(self.conv(x))
		return out+x

class MiniResNet(BaseModel):
	"""Mini-resnet model. Entire Resnet-18/34/50
	is a killer for humble machines, lighter version
	is defined here"""
	def __init__(self, in_channels, image_size=84, classes=len(folders)):
		super().__init__()

		self.drop = nn.Dropout(0.3)
		self.classes = classes		
		self.conv1 = conv_block(in_channels, in_channels*5)
		self.resid = ResidBlock(in_channels*5)
		self.conv2 = conv_block(in_channels*5, in_channels)
		self.pool = nn.MaxPool2d(2)
		self.flat = nn.Flatten()
		flat_size = math.floor(in_channels*((image_size)/2)**2)
		self.cl = classifier(flat_size, self.classes)
		
	def forward(self, x):
		x = F.leaky_relu(self.conv1(x))
		x = self.drop(x)
		x = F.leaky_relu(self.resid(x))
		x = self.pool(x)
		x = self.drop(x)
		x = F.leaky_relu(self.conv2(x))
		x = self.flat(x)
		return self.cl(x)
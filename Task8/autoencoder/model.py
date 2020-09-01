import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchvision.transforms as transforms

from custom_mnist import CustomMNIST

criterion = nn.MSELoss()
batch_size = 64

train_trans = transforms.Compose([
	transforms.RandomRotation(90),
	transforms.ToTensor()])
val_trans = transforms.Compose([transforms.ToTensor()])

class BaseModel(pl.LightningModule):

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = criterion(y_hat, y)
		result = pl.TrainResult(loss)
		result.log('train_loss', loss, on_epoch=True)
		return result

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = criterion(y_hat, y)
		result = pl.EvalResult(checkpoint_on=loss)
		result.log('val_loss', loss)
		return result

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = criterion(y_hat, y)
		result = pl.EvalResult()
		result.log('test_loss', loss)
		return result


	def setup(self, stage=None):

		if stage == 'fit' or stage is None:
			train_set = CustomMNIST(download=True, 
				train=True, root='./data',
				transform=train_trans)
			train_size = int(0.8*len(train_set))
			self.train_set, self.val_set = random_split(
					train_set, 
					[train_size,
					len(train_set)-train_size])

		if stage == 'test' or stage is None:
			self.test_set = CustomMNIST(download=True, 
				train=False, root='./data',
				transform=val_trans)

	def train_dataloader(self):
		return DataLoader(self.train_set,
			shuffle=True,
			batch_size=batch_size,
			num_workers=2)

	def val_dataloader(self):
		return DataLoader(self.val_set, 
			batch_size=batch_size,
			shuffle=True,
			num_workers=2)

	def test_dataloader(self):
		return DataLoader(self.test_set, 
			batch_size=batch_size,
			shuffle=True,
			num_workers=2)
		

class Autoencoder(BaseModel):
	def __init__(self, in_channels=1, learning_rate=1e-3):

		super(Autoencoder, self).__init__()
		self.learning_rate = learning_rate

		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels, 
				in_channels*8, 3, 
				stride=3, padding=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(in_channels*8, 
				in_channels*4, 3, 
				stride=2, padding=1),
			nn.ReLU(True),
			nn.MaxPool2d(2, stride=1))

		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(in_channels*4, 
				in_channels*8, 3, stride=2),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels*8, 
				in_channels*4, 5, stride=3, padding=1),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels*4, 
				in_channels, 2, stride=2, padding=1))

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(),
			lr=self.learning_rate)
		return optimizer

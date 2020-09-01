import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from dataset import SegDataset, ImgMaskDataset

"""
As a skeleton model for segmentation I was
looking at SegNet and UNet so you will see
similar blocks. Needless to say those are not
original models but some light parts of them.

"""

root_dir = 'foosball_dataset'
batch_size = 8
height = 360
width = 640

filter_scale = 16

class IoULoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1-IoU

criterion = IoULoss()

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
			self.train_set = ImgMaskDataset('train', self.data_dir)
			self.val_set = ImgMaskDataset('val', self.data_dir)

		if stage == 'test' or stage is None:
			self.test_set = ImgMaskDataset('test', self.data_dir)

	def train_dataloader(self):
		return DataLoader(self.train_set,
			shuffle=True,
			batch_size=batch_size,
			num_workers=2)

	def val_dataloader(self):
		return DataLoader(self.val_set, 
			batch_size=batch_size,
			shuffle=False,
			num_workers=2)

	def test_dataloader(self):
		return DataLoader(self.test_set, 
			batch_size=batch_size,
			shuffle=False,
			num_workers=2)


class DownBlock(nn.Module):

	def __init__(self, channels, kernel_size=1, padding=0, activation='relu'):

		super().__init__()
		self.channels = channels

		self.conv = nn.Conv2d(channels, 
			channels*filter_scale, 
			(kernel_size, kernel_size), 
			padding=padding)
		self.norm = nn.BatchNorm2d(channels*filter_scale)
		if activation == 'relu':
			self.activation = nn.ReLU()
		else:
			self.activation = nn.LeakyReLU()
		self.pool = nn.MaxPool2d(2)

	def forward(self, x):
		x = self.conv(x)
		x = self.activation(x)
		x = self.norm(x)
		x= self.pool(x)
		return x


class UpBlock(nn.Module):

	def __init__(self, channels, stride=1,
	padding=None, kernel_size=1, 
	activation='relu'):

		super().__init__()
		self.channels = channels

		if padding is None:
			self.conv = nn.Conv2d(channels, 
			channels//filter_scale, 
			(kernel_size, kernel_size))
		else:
			self.conv = nn.Conv2d(channels, 
				channels//filter_scale, 
				(kernel_size, kernel_size),
				padding=padding)
		self.norm = nn.BatchNorm2d(channels//filter_scale)
		if activation == 'relu':
			self.activation = nn.ReLU()
		else:
			self.activation = nn.LeakyReLU()
		self.upconv = nn.ConvTranspose2d(
			channels//filter_scale,
			channels//filter_scale, 
			kernel_size=(1,1), 
			stride=stride)

	def forward(self, x):
		x = self.conv(x)
		x = self.activation(x)
		x = self.norm(x)
		return self.upconv(x)


class MiniSegNet(BaseModel):

	def __init__(self, channels, 
		height=height, width=width, 
		num_classes=1,
		data_dir=root_dir,
		learning_rate=1e-3):

		super().__init__()

		self.width = width
		self.height = height
		self.data_dir = data_dir
		self.learning_rate = learning_rate

		self.down = DownBlock(channels, padding=1)
		self.up = UpBlock(channels*filter_scale, 2)
		self.out = nn.Conv2d(channels, num_classes, (2, 2))
		self.last = nn.Conv2d(
			num_classes*2, num_classes, (1, 1))

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(
			self.parameters(), lr=self.learning_rate)
		return optimizer

	def forward(self, x):

		y = x.sum(1, keepdim=True)
		x = self.down(x)
		x = self.up(x)
		x = F.relu(self.out(x))
		x = torch.cat([x, y], axis=1)
		x = self.last(x)

		return x
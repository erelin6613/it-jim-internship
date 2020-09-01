import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from model import Autoencoder
from custom_mnist import CustomMNIST

logger = pl.loggers.CSVLogger('logs', 
	name='Autoencoder_logs')

chp_path = os.path.join('logs', 
	'Autoencoder_logs', 'version41', 
	'checkpoints', 'epoch=0.ckpt')

early_stop = pl.callbacks.EarlyStopping(
	monitor='val_loss',
	patience=2,
	strict=False,
	verbose=False,
	mode='min')

def train(pretrained=False):

	model = Autoencoder()
	print(model)
	trainer = pl.Trainer(max_epochs=1,
						gpus=None, 
						progress_bar_refresh_rate=1,
						#fast_dev_run=True,
						early_stop_callback=early_stop,
						logger=logger)

	if not pretrained:
		res = trainer.fit(model)
		torch.save(model.state_dict(), 'model.pth')
	return None, trainer, None

def test(model, trainer):
	if model is None:
		model = Autoencoder()
		model.setup()
		res = trainer.test(model=model, 
			test_dataloaders=model.test_dataloader(), 
			ckpt_path=chp_path)
	return model, trainer, res

def infer(model, trainer, half=True):
	preds = []
	loader = model.test_dataloader()
	half_len = len(loader)//2
	print('Inference...')
	for i, b in tqdm(enumerate(loader)):
		p = model(b[0]).detach().numpy()
		for pred in p:
			preds.append(pred)
	return preds


def show_batch(noisy, original, denoised):
	"""Grid making to visulize images and predictions"""
	fig, ax = plt.subplots(1, 3)
	ax[0].imshow(noisy[0][0], cmap='gray')
	ax[0].set_title('noisy', fontsize=11)
	ax[0].axis('off')

	ax[1].imshow(original[0][0], cmap='gray')
	ax[1].set_title('original', fontsize=11)
	ax[1].axis('off')

	ax[2].imshow(denoised[0][0], cmap='gray')
	ax[2].set_title('denoised', fontsize=11)
	ax[2].axis('off')

	plt.tight_layout()
	plt.show()

def infer(model, sample=True):
	"""Predict images and extract labels.
	Do for entire loader if sample is False"""

	test_loader = model.test_dataloader()
	for batch in test_loader:
		x, original = batch
		print(x.shape)
		y = model(x)
		show_batch(x.detach().numpy(), original, y.detach().numpy())
		if sample:
			return

def main():

	epochs = 5
	pretrained = False
	model, trainer, res = train(pretrained=False)
	print(res)
	model, trainer, res = test(model, trainer)
	preds = infer(model)

if __name__ == '__main__':
	main()

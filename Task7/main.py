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
# import albumentations as A
# from albumentations.pytorch import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import CapsNetForClassification, Encoder, precision_recall_score
from custom_mnist import CustomMNIST

"""

I did not change the pipeline from the previous task,
most of the time being spent on trying to build a
good enough custom version of capsnet. However, a few changes still
have taken place:

* BaseModel recieves log_results method for logging
* accuracy score is replaced with precision, recall
and f1 score, changes reflected in BaseModel too
* Required for the task functionality is done to the
best of my ability and timing (image rotation, partial
freezing etc.)

What I wanted to do but faced issues:

- logging to separate file instead of one. First I
	wanted to keep a few files for ease of accesing
	results by others but it turned out to be trickier
	to reset logging module to set a new logging output
	file. As of now I will leave one file only but
	outputs will have a mark.

- albumentations on MNIST. I did not understand why
	transofrmations of this module throw an error on MNIST
	pictures. It does not seem to be PIL/tensor problem
	but something I will investigate if have some time at
	my disposal.

- build the same model in tensorflow. As per usual I attached
	the notebook `attempt-to-building-capsnet-with-tensorflow.ipynb`
	where I tried to customize model and block but spoiler:
	I could not, perphaps not yet. 

"""


# on rotated: test loss: 0.1796 test prec: 0.9617 test recall: 0.9583 test f1 score: 0.9593
# with frozen encoder: 

torch.manual_seed(10)

@torch.no_grad()
def evaluate(model, val_loader):
	model.eval()
	outputs = [model.validation_step(batch) for batch in val_loader]
	return model.validation_epoch_end(outputs)

def train(model, train_loader, val_loader, 
	epochs=1, max_lr=1e-3, base_lr=1e-4, weight_decay=1e-4, 
	grad_clip=0.1, opt_func=torch.optim.AdamW,
	freeze_mark=None):
	"""Training loop. First we need to tell 
	optimizer what to optimize. Then I used
	one cycle policy for learning rate
	scheduling, corresponding class also
	needs to know which optimizer it is
	being applied to."""
	
	optimizer = opt_func(model.parameters(),
		base_lr, weight_decay=weight_decay)
	sched = optim.lr_scheduler.OneCycleLR(
		optimizer, max_lr, epochs=epochs, 
		steps_per_epoch=len(train_loader))
	
	for epoch in range(epochs):
		model.train()
		train_losses = []
		for batch in tqdm(train_loader):
			#print(batch[0].shape)
			loss = model.training_step(batch)
			train_losses.append(loss)
			loss.backward()
			
			"""Also I applied gradient clipping to
			avoid exploding gradient, without it
			FC model goes off the rails drastically"""
			if grad_clip is not None: 
				clip_grad_value_(model.parameters(), grad_clip)
			
			optimizer.step()
			optimizer.zero_grad()
			sched.step()
		
		result = evaluate(model, val_loader)
		result['train_loss'] = torch.stack(train_losses).mean().item()

		"""Next conditions save model weights of the best model,
		model passed through half epochs and the last one epoch."""

		checkpoints = []

		# uncomment if needed
		"""
		if epoch == epochs//2:
			chp_path = '{}_{}_half.pth'.format(str(model).split('(')[0], str(freeze_mark))
			checkpoints.append(chp_path)
			print('saved checkpoint at {}'.format(chp_path))

		elif epoch == epochs-1:
			chp_path = '{}_{}_last.pth'.format(str(model).split('(')[0], str(freeze_mark))
			checkpoints.append(chp_path)
			print('saved checkpoint at {}'.format(chp_path))
		"""

		if result['val_loss'] < model.checkpoint['val_loss']:
			chp_path = '{}_{}_best.pth'.format(str(model).split('(')[0], str(freeze_mark))
			checkpoints.append(chp_path)
			print('saved checkpoint at {}'.format(chp_path))

		if len(checkpoints)>0:
			for p in checkpoints:
				torch.save(model.state_dict(), p)

		model.epoch_end(epoch, result, freeze_mark)
	return model

def test(model, test_loader):

	model.eval()
	result = evaluate(model, test_loader)
	print('test loss:', result['val_loss'],
		'test prec:', result['val_prec'],
		'test recall:', result['val_recall'],
		'test f1 score:', result['val_fscore']
		)

def show_batch(imgs, probs, ds_classes, grid=(4, 4)):
	"""Grid making to visulize images and predictions"""
	fig, ax = plt.subplots(grid[0], grid[1])
	classes = torch.argmax(probs.detach(), dim=1)
	print(classes)
	max_prob = np.max(probs.detach().numpy(), axis=1)
	j = 0
	i = 0
	for c, inp in enumerate(zip(imgs, classes)):
		img, label = inp
		#for c, im in enumerate(img):
		ax[i][j].imshow(img[0], cmap='gray')
		ax[i][j].set_title('{} {}%'.format(
			label, round(
				max_prob[c].item()*100)), fontsize=11)
		ax[i][j].axis('off')
		
		j += 1
		if j == grid[1]:
			j = 0
			i += 1
	plt.tight_layout()
	plt.show()

def infer(model, test_loader, sample=True):
	"""Predict images and extract labels.
	Do for entire loader if sample is False"""
	model.eval()
	for batch in test_loader:
		x, labels = batch
		print(x.shape)
		img = x #.unsqueeze(0)
		y = model(x)
		probs = F.softmax(y, dim=1)
		# probs = precision_recall_score(y, labels)[0]
		show_batch(img, probs, test_loader.dataset.classes)
		if sample:
			return

def load_mnist(rotated=True):

	"""
	Any idea why albumentations throws an error when
	working with MNIST? Even ToTensor transform does
	not work (is there 3 channels hardcoded somewhere? :D)

	train_trans = A.Compose([
		#A.RandomRotate90(),
		#A.RandomBrightnessContrast(0.3, 0.3),
		ToTensor()])

	val_trans = A.Compose([ToTensor()])"""

	train_trans = transforms.Compose([
		transforms.RandomRotation(90),
		transforms.ToTensor()])
	val_trans = transforms.Compose([transforms.ToTensor()])

	if rotated:

		train_set = CustomMNIST(
			'data', train=True, 
			download=True,
			transform=train_trans)

		train_size = int(0.8*len(train_set))
		train_set, val_set = random_split(
			train_set, [train_size, len(train_set)-train_size])

		test_set = CustomMNIST(
			'data', train=False,
			download=True,
			transform=val_trans)
	else:
		train_set = MNIST(
			'data', train=True, 
			download=True,
			transform=train_trans)

		train_size = int(0.8*len(train_set))
		train_set, val_set = random_split(
			train_set, [train_size, len(train_set)-train_size])

		test_set = MNIST(
			'data', train=False, 
			download=True,
			transform=val_trans)


	train_loader = DataLoader(train_set, 16, shuffle=True)
	val_loader = DataLoader(val_set, 16, shuffle=True)
	test_loader = DataLoader(test_set, 16, shuffle=True)

	return train_loader, val_loader, test_loader

def main():

	epochs = 5
	pretrained = True
	pretrained_path = 'CapsNetForClassification_rotated_train_best.pth'

	train_loader, val_loader, test_loader = load_mnist(
		rotated=True)
	model = CapsNetForClassification()
	print(model)

	if not pretrained:
		# pretrain model on rotated images
		print('Training on rotated images')
		model = train(model, train_loader, 
			val_loader, epochs, freeze_mark='rotated_train')
	else:
		print('Loading pretrained model...')
		model.load_state_dict(torch.load(pretrained_path))
	test(model, test_loader)
	infer(model, test_loader)

	print('Training on original images')
	train_loader, val_loader, test_loader = load_mnist(
		rotated=False)

	print('Training classifier...')
	# freezing encoder part and training classifier only
	for module in model.children():
		if isinstance(module, Encoder):
			for param in module.parameters():
				param.requires_grad = False

	model = train(model, train_loader, 
		val_loader, epochs, freeze_mark='frozen_encoder')
	test(model, test_loader)
	infer(model, test_loader)

	model = CapsNetForClassification()
	model.load_state_dict(torch.load(pretrained_path))

	# freezing classifier and training encoder
	print('Training Encoder...')
	for module in model.children():
		if isinstance(module, Encoder):
			continue
		for param in module.parameters():
			param.requires_grad = False

	model = train(model, train_loader, 
		val_loader, epochs, freeze_mark='frozen_classifier')
	test(model, test_loader)
	infer(model, test_loader)

	# no freezing
	model = CapsNetForClassification()
	model.load_state_dict(torch.load(pretrained_path))
	model = train(model, train_loader, 
		val_loader, epochs, freeze_mark='no_freezing')
	test(model, test_loader)
	infer(model, test_loader)

if __name__ == '__main__':
	main()

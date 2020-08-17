import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_models import FC_model, ConvModel, MiniResNet, accuracy
from utils import load_image
"""
Implementation of three models with pytorch.
Here I trained models on original images. Perphaps
using some other features would be beneficial but
on the behalf of NN I did applied a few 
optimizations.

FC_model - test accuracy bounces around 0.22
ConvModel - test accuracy ~ 0.15 [I am surprised too]
MiniResNet - test accuracy hitting 0.30

Each time trained with 10 epochs.

The keras part of the task I implemented in
kaggle kernel which also can be found in the
same folder.

Online kernel: 
https://www.kaggle.com/erelin6613/task-6-for-it-jim-summer-internship-2020

Dataset:
https://www.kaggle.com/erelin6613/miniimagenet-itjim-internship-2020-task6

Let me know if I should delete/hide those

"""
train_folder = 'dataset/train'
val_folder = 'dataset/val'
test_folder = 'dataset/test'

# some augmentations; we want to extend our
# training images but not validation and test ones.
# Also we do not want here a vertical flip and
# something unrealistic.
train_trans = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Normalize(stats),
	transforms.RandomErasing(scale=(0.02, 0.2)),
	transforms.RandomHorizontalFlip()])
val_trans = transforms.Compose([transforms.ToTensor()])

torch.manual_seed(24)

def get_stats(loader):
	"""I wanted also to normalize images but
	it seems I will not have enough time"""

	mean, var, num_imgs = 0, 0, 0
	for i_batch, batch_target in enumerate(loader):
		img, _ = batch_target
		img = img.view(img.size(0), img.size(1), -1)
		num_imgs += img.size(0)
		mean += img.mean(2).sum(0) 
		var += img.var(2).sum(0)

	mean = mean/num_imgs
	var = var/num_imgs
	std = torch.sqrt(var)

	return mean, std


@torch.no_grad()
def evaluate(model, val_loader):
	model.eval()
	outputs = [model.validation_step(batch) for batch in val_loader]
	return model.validation_epoch_end(outputs)

def train(model, train_loader, val_loader, 
	epochs=1, max_lr=1e-3, base_lr=1e-4, weight_decay=1e-4, 
	grad_clip=0.1, opt_func=torch.optim.AdamW):
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

		if epoch == epochs//2:
			chp_path = '{}_half.pth'.format(str(model).split('(')[0])
			checkpoints.append(chp_path)
			print('saved checkpoint at {}'.format(chp_path))

		elif epoch == epochs-1:
			chp_path = '{}_last.pth'.format(str(model).split('(')[0])
			checkpoints.append(chp_path)
			print('saved checkpoint at {}'.format(chp_path))

		if result['val_loss'] < model.checkpoint['val_loss']:
			chp_path = '{}_best.pth'.format(str(model).split('(')[0])
			checkpoints.append(chp_path)
			print('saved checkpoint at {}'.format(chp_path))

		if len(checkpoints)>0:
			for p in checkpoints:
				torch.save(model.state_dict(), p)

		model.epoch_end(epoch, result)
	return model

def test(model, test_loader):

	model.eval()
	result = evaluate(model, test_loader)
	print('test loss:', result['val_loss'],
		'test acc:', result['val_acc'])

def show_batch(imgs, probs, ds_classes, grid=(4, 4)):
	"""Grid making to visulize images and predictions"""
	fig, ax = plt.subplots(grid[0], grid[1])
	classes = torch.argmax(probs.detach(), dim=1)
	print(classes)
	max_prob = np.max(probs.detach().numpy(), axis=1)
	for img in imgs:
		j = 0
		i = 0
		for c, im in enumerate(img):
			ax[i][j].imshow(im.permute(1, 2, 0))
			ax[i][j].set_title('{} {}%'.format(
				ds_classes[classes[c]], round(
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
		img = x.unsqueeze(0)
		y = model(x)
		probs = accuracy(y, labels, True)
		show_batch(img, probs, test_loader.dataset.classes)
		if sample:
			return


def main():

	batch_size = 32
	test_batch_size = 16
	epochs = 10

	train_set = ImageFolder(train_folder, train_trans, loader=load_image)
	val_set = ImageFolder(val_folder, val_trans, loader=load_image)
	test_set = ImageFolder(test_folder, val_trans, loader=load_image)

	train_loader = DataLoader(train_set, batch_size, shuffle=True)
	val_loder = DataLoader(val_set, batch_size, shuffle=True)
	test_loader = DataLoader(test_set, test_batch_size, shuffle=True)

	# stats = get_stats(train_loader)

	fc_model = FC_model(3)
	cnn_model = ConvModel(3)
	res_model = MiniResNet(3)

	print(str(fc_model))
	print('Training fully-connected model...')
	fc_model = train(fc_model, train_loader, val_loder, epochs)
	test(fc_model, test_loader)
	infer(fc_model, test_loader)

	print(str(cnn_model))
	print('Training convolutional model...')
	cnn_model = train(cnn_model, train_loader, val_loder, epochs)
	test(cnn_model, test_loader)
	infer(cnn_model, test_loader)

	print(str(res_model))
	print('Training ResNet-like model...')
	res_model = train(res_model, train_loader, val_loder, epochs)
	test(res_model, test_loader)
	infer(res_model, test_loader)

if __name__ == '__main__':
	main()
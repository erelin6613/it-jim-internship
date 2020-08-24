import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Of course I could do something like 
`torchvision.models.resnet18()` but I want to 
experiment with capsule network (or at least a
tiny part of it, I am not a Google afterall :)).
Even though actual architecture seems quite bizzare
I will try to impliment at least some parts of it.

https://arxiv.org/pdf/1804.08376.pdf
http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules.pdf
https://www.sciencedirect.com/science/article/pii/S1319157819309322
CapsNet tutorial(s):
https://pechyonkin.me/capsules-1/

Here are some note:

* I followed these papers and tutorials, originally
	CapsNet was designed to generate images, not to
	classify them. For our purpose we can limit ourselves
	to encoder+classifer arcitecture.

* There is math-heavy apparatus which it seems I
	implemented correctly. Good F scores might be not
	the best measure of how correctly I did, moreover
	there is little NN which do not work on MNIST.

* It is still my implimentation of CapsNet so some values,
	details will deviate from original architecture.
	Once again I am just experimenting, not trying to
	win imagenet competition :)

"""

def precision_recall_score(outputs, labels, 
	threshold=0.5, beta=1, epsilon=1e-8):

	"""Calculate precision score, recall score 
	and f-score in one go.
	P.S. Honestly borrowed from kaggle kernels with my minor 
	tweaks for epsilon to prevent zero divition 
	and dimentions adjustments. On the positive
	side now I know what tilde operator does"""

	logits = outputs.argmax(dim=1)
	TP = (logits & labels).sum().float()
	TN = ((~logits) & (~labels)).sum().float()
	FP = (logits & (~labels)).sum().float()
	FN = ((~logits) & labels).sum().float()

	precision = torch.mean(TP / (TP + FP + epsilon))
	recall = torch.mean(TP / (TP + FN + epsilon))
	F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)
	return precision, recall, F2


def classifier(in_channels, classes=10):
	"""Module used as classification layer"""
	return nn.Sequential(nn.Flatten(),
		nn.Linear(in_channels, classes))

class BaseModel(nn.Module):
	"""Base model. As per usual it is going to be
	much simplier with logging and tracking metrics"""
	
	def training_step(self, batch):
		try:
			assert len(self.checkpoint)>0
		except Exception:
			self.checkpoint = {'val_loss': np.inf, 
			'train_loss': np.inf}
		x, y = batch
		out = self(x)
		loss = F.cross_entropy(out, y)
		return loss
	
	def validation_step(self, batch):
		x, y = batch 
		out = self(x)
		loss = F.cross_entropy(out, y)
		f_score = precision_recall_score(out, y)
		return {'val_loss': loss.detach(),
				'val_prec': f_score[0],
				'val_recall': f_score[1],
				'val_fscore': f_score[2]}
		
	def validation_epoch_end(self, outputs):
		batch_losses = [x['val_loss'] for x in outputs]
		epoch_loss = torch.stack(batch_losses).mean()

		batch_pres = [x['val_prec'] for x in outputs]
		epoch_pres = torch.stack(batch_pres).mean()

		batch_rec = [x['val_recall'] for x in outputs]
		epoch_rec = torch.stack(batch_rec).mean()

		batch_f = [x['val_fscore'] for x in outputs]
		epoch_f = torch.stack(batch_f).mean()

		return {'val_loss': round(epoch_loss.item(), 4), 
				'val_prec': round(epoch_pres.item(), 4),
				'val_recall': round(epoch_rec.item(), 4),
				'val_fscore': round(epoch_f.item(), 4)}
	
	def epoch_end(self, epoch, result, freeze_mark=None):

		try:
			if self.checkpoint['val_loss'] > result['val_loss']:
				self.checkpoint = result

		except Exception:
			self.checkpoint = result

		print(f'Epoch: {epoch}, \
			train_loss: {result["train_loss"]}, \
			\nval_loss: {result["val_loss"]}, \
			val_prec: {result["val_prec"]},\
			\nval_recall: {result["val_recall"]},\
			val_fscore: {result["val_recall"]}')
		self.log_results(epoch, result, freeze_mark)
		return result

	def log_results(self, epoch, result, freeze_mark):

		logging.basicConfig(
			filename='{}.log'.format(
				str(self).split('(')[0]), level=logging.INFO)

		if freeze_mark is not None:
			string = '['+freeze_mark+'] Epoch:'+str(epoch)+':'+str(result)
		else:
			string = '[original] Epoch:'+str(epoch)+':'+str(result)
		logging.info(string)
		

class PrimaryCaps(nn.Module):
	"""PrimaryCaps block.
	Original paper has 8 capsules, for my 
	purposes two would be sufficient."""
	def __init__(self, num_capsules=2,
		in_channels=32, out_channels=16, 
		kernel_size=9, num_routes=16*6*6):

		super(PrimaryCaps, self).__init__()
		self.num_routes = num_routes
		self.capsules = nn.ModuleList([
			nn.Conv2d(in_channels=in_channels, 
				out_channels=out_channels, 
				kernel_size=kernel_size, 
				stride=2, padding=0)
			for _ in range(num_capsules)])

	def forward(self, x):
		y = [capsule(x) for capsule in self.capsules]
		y = torch.stack(y, dim=1)
		y = y.view(x.size(0), self.num_routes, -1)
		return self.squash(y)

	def squash(self, input_tensor):
		squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
		output_tensor = squared_norm * input_tensor / (
			(1. + squared_norm) * torch.sqrt(squared_norm))
		return output_tensor


class DigitCaps(nn.Module):
	"""DigitCaps block.
	Here is equations part that follows feature 
	extraction procedures. Original papper also has
	a few iterations for calculating c, s and v
	which I skipped."""
	def __init__(self, num_capsules=10, 
		num_routes=16*6*6, 
		in_channels=2, out_channels=16):

		super().__init__()

		self.num_capsules = num_capsules
		self.num_routes = num_routes
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.W = nn.Parameter(torch.randn(
			1, self.num_routes, 
			self.num_capsules, 
			self.out_channels, 
			self.in_channels))

	def forward(self, x):

		b_size = x.shape[0]
		
		x = torch.stack(
			[x]*self.num_capsules, dim=2).unsqueeze(4)
		W = torch.cat(
			[self.W]*b_size, dim=0)
		u_hat = torch.matmul(W, x).squeeze()
		b_ij = torch.autograd.Variable(torch.zeros(
			1, self.num_routes, self.num_capsules, 1))
		c_ij = F.softmax(b_ij, dim=1)
		s_j = (c_ij*u_hat).sum(dim=1, keepdim=True)
		v_j = self.squash(s_j)

		return v_j.squeeze(1)

	def squash(self, input_tensor):
		squared_norm = (input_tensor ** 2).sum(
			-1, keepdim=True)
		output_tensor = squared_norm * input_tensor / (
			(1. + squared_norm) * torch.sqrt(squared_norm))
		return output_tensor


class Encoder(nn.Module):
	"""Encoder block.
	Combines input convolution layer followed by
	components of CapsNet"""
	def __init__(self):

		super().__init__()

		self.conv_layer = nn.Conv2d(
			in_channels=1,
			out_channels=32,
			kernel_size=9,
			stride=1)

		self.primary_capsules = PrimaryCaps()
		self.digit_capsules = DigitCaps()

	def forward(self, x):
		x = F.relu(self.conv_layer(x))
		x = self.primary_capsules(x)
		return self.digit_capsules(x)

class CapsNetForClassification(BaseModel):
	"""CapsNet comprising architecture
	encoder+classifier."""
	def __init__(self, num_classes=10):
		super().__init__()

		self.encoder = Encoder()
		self.num_classes = num_classes
		self.classifier = classifier(16*10, 10)

	def forward(self, x):
		x = self.encoder(x)
		x = self.classifier(x)
		return x

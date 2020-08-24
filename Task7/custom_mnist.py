import os
from PIL import Image
import torch
from torchvision.datasets import MNIST

"""
Don't blame me for laziness but such an approach 
saves a lot of time to prepare dataset :)
"""


class CustomMNIST(MNIST):

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], int(self.targets[index])

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img.numpy(), mode='L')

		if self.transform is not None:
			#print(img)
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		# img = torch.rot90(img, 1, [0, 1])
		img = img.transpose(2, 1).flip(2) #.flip(3)

		return img, target
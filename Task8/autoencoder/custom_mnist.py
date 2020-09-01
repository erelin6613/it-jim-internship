import os
from PIL import Image
import torch
from torchvision.datasets import MNIST
import numpy as np

"""

Somewhere inside albumentaions code which I am
relactent to fix now

File "/home/val/coding/it-jim-internship/Task8/task8_env/lib/
python3.7/site-packages/albumentations/core/transforms_interface.py", 
line 142, in update_params
    params.update({"cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]})
AttributeError: 'Image' object has no attribute 'shape'


"""

def add_noise(img, noise_type="speckle"):
	#print(img)
	h, w = img.shape[1:] #28,28
	img = img.float().numpy()
	if noise_type=="gaussian":
		mean=0
		var=5
		sigma = var**.5
		noise = np.random.normal(-5.9, 5.9, img.shape)
		noise = noise.reshape(h,w)
		img = img+noise

	if noise_type == "speckle":
		noise = np.random.randn(h,w)
		noise = noise.reshape(h,w)
		img = img + img*noise
	#print(img.shape)	
	return torch.tensor(img) #.unsqueeze(0)

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
			original = self.transform(img)
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)
		img = add_noise(img).float()
		#augmented = self.transforms(image=original, mask=img)
		return img, original #augmented['image'], augmented['mask']
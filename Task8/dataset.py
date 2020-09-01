import os
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import pytorch_lightning as pl
import cv2
import numpy as np

height = 360//4
width = 640//4

augs = {'train': A.Compose([
					A.VerticalFlip(p=0.5),
					A.GridDistortion(p=0.5),
					#A.CLAHE(p=0.5),
					#A.RGBShift(),
					#A.HorizontalFlip(),
					#A.RandomRotate90(p=0.5),
					#A.RandomBrightnessContrast(p=0.7),
					ToTensorV2()]),
		'val': A.Compose([ToTensorV2()]),
		'test': A.Compose([ToTensorV2()])}

root_dir = 'foosball_dataset'


class ImgMaskDataset(Dataset):
	def __init__(self, phase, root_dir=root_dir, resize=True):

		self.root_dir = root_dir
		self.phase = phase

		img_path = os.path.join(root_dir,
			f'{phase}_set')
		self.files = sorted(os.listdir(img_path)) #pd.read_csv(img_path)
		#print(self.files)
		self.transforms = augs[phase]
		self.resize = resize

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):

		im_path = os.path.join(self.root_dir, 
			f'{self.phase}_set',
			self.files[index])
		img = cv2.imread(im_path)/255 #.astype('uint8')
		if self.resize:
			img = cv2.resize(img, (width, height), cv2.INTER_AREA)
		#print(img.shape)
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
		#print(img.shape)
		mask_path = os.path.join(self.root_dir, 
			f'{self.phase}_set_mask', 
			self.files[index]).replace('jpg', 'png')

		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255
		if self.resize:
			mask = cv2.resize(mask, (width, height), cv2.INTER_AREA)
		mask = mask[np.newaxis, ...]
		#print(mask.shape)
		#print(mask)
		#mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		#mask = mask[:, :, -1]
		#print(mask.sum())
		#mask = mask.reshape(1, mask.shape[0], mask.shape[1])
		augmented = self.transforms(image=img, mask=mask)
		return augmented['image'], augmented['mask']


class SegDataset(pl.LightningDataModule):

	def setup(self, stage=None):

		if stage == 'fit' or stage is None:
			self.train_set = ImgMaskDataset('train', self.data_dir)
			self.val_set = ImgMaskDataset('val', self.data_dir)

		if stage == 'test' or stage is None:
			ImgMaskDataset('train', self.data_dir)

	def train_dataloader(self):
		return DataLoader(self.train_set, 
			batch_size=batch_size)

	def val_dataloader(self):
		return DataLoader(self.val_set, 
			batch_size=batch_size)

	def test_dataloader(self):
		return DataLoader(self.test_set, 
			batch_size=batch_size)
import os
import pytorch_lightning as pl
import torch
from tqdm import tqdm
import cv2
import numpy as np

from model import MiniSegNet
from dataset import SegDataset

"""
I hope that will not be much of big deal
but I used pytorch_lightning library to
take care of some things like defining
training loop, logging, saving checkpoints,
etc. I made sure to reflect this
dependancy in requirements.txt

As I am writing these comments my model is
training I hope I will be at least okay.
One thing is for sure: with this pipeline
experiments are way easier.

For now the model is well... not good (I hope it 
all boils down to not enough training :) ) 
that is why I cannot even figure out strategy 
to make a BBox for detected ball. Instead I 
am writing to the file 
`predicted_video_1.avi` a thresholded video.
*Update: it finds something but still far far
from prefect.

This one task is really challanging one
in terms of time but interesting. Thank 
you and I am sorry for concise comments
and humble results :)

As per usual I wish I had more time.

P.S. I am really interested what I am missing if
it is arcitecture related on somthing (in both 
segmentation model and autoencoder). Any
suggestion is highly appriciated.
"""

height = 360
width = 640

logger = pl.loggers.CSVLogger('logs', 
	name='MiniSegNet_logs')

chp_path = os.path.join('logs', 'MiniSegNet_logs',
	'version_78', 'checkpoints', 'epoch=4.ckpt')

early_stop = pl.callbacks.EarlyStopping(
	monitor='val_loss',
	patience=2,
	strict=False,
	verbose=False,
	mode='min')

pl.trainer.seed_everything(66)

def train(pretrained=False):

	model = MiniSegNet(3, 2)
	print(model)
	trainer = pl.Trainer(max_epochs=5,
						gpus=None, 
						progress_bar_refresh_rate=1,
						early_stop_callback=early_stop,
						#fast_dev_run=True,
						logger=logger)

	if not pretrained:
		res = trainer.fit(model.double())
		torch.save(model.state_dict(), 'model.pth')
	return None, trainer, None

def test(model, trainer):
	if model is None:
		model = MiniSegNet(3, 2)
		model.double()
		model.setup()
		res = trainer.test(model=model, 
			test_dataloaders=model.test_dataloader(), 
			ckpt_path=chp_path)
	return model, trainer, res

def infer(model, trainer, half=True):
	preds = []
	loader = model.test_dataloader()
	half_len = len(loader)//2
	print('Making educated guess where the ball is...')
	for i, b in tqdm(enumerate(loader)):
		p = model(b[0]).detach().numpy()
		for pred in p:
			preds.append(pred)
		if half:
			if i > half_len:
				break
	return preds

def record_video(preds, height=height, width=width, out_file='predicted_video_1.avi'):

	fourcc = cv2.VideoWriter_fourcc(*"XVID")
	fps = 5
	image_size = (width, height)
	writer = cv2.VideoWriter(out_file, fourcc, fps, image_size)
	for frame in preds:

		img = np.squeeze(frame, 0)
		img = cv2.resize(img, image_size, cv2.INTER_AREA)*255
		img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
		shot = draw_box(img)
		writer.write(shot)
	writer.release()

def draw_box(shot, thresh=175, kernel_size=3):
	kernel = np.ones((kernel_size, kernel_size), np.uint8)
	original = shot.copy()
	shot = cv2.morphologyEx(shot, cv2.MORPH_CLOSE, kernel)
	_, shot = cv2.threshold(shot, thresh, 255, cv2.THRESH_BINARY)
	return shot


def main():
	model, trainer, res = train(pretrained=False)
	print(res)
	model, trainer, res = test(model, trainer)
	print(res)
	preds = infer(model, trainer)
	record_video(preds)


if __name__ == '__main__':
	main()
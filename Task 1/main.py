import cv2
import numpy as np
from scipy import ndimage
from matplotlib.colors import hsv_to_rgb
from matplotlib import pyplot as plt

video_path = 'input_video.avi'

def read_video(path):

	cap = cv2.VideoCapture(path)
	frm_count = 0
	key = None
	while True:

		ret, shot = cap.read()
		if ret:
			cv2.imshow('original', shot)
			shot = calculate_shot(shot, strategy='color-decompose')
			cv2.imshow('mask', shot)
		else:
			break

		if key == ord('p'):
			wait_period = 0
		else:
			wait_period = 30
		key = cv2.waitKey(wait_period)

	cap.release()
	cv2.destroyAllWindows()

def calculate_shot(shot, 
	strategy='thresh',
	kernel_size=3):

	kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
	if strategy == 'dilate-erode':
		d = cv2.dilate(shot, kernel)
		e = cv2.erode(shot, kernel)
		shot = d-e

	if strategy == 'ad-thresh':
		shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
		shot = cv2.adaptiveThreshold(shot, 255, 
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)

	if strategy == 'gradient':
		shot = cv2.morphologyEx(shot, cv2.MORPH_GRADIENT, kernel)

	if strategy == 'canny':
		shot = cv2.Canny(shot, 25, 25)

	if strategy == 'thresh':
		shot = cv2.cvtColor(shot, cv2.COLOR_BGR2LAB)
		shot = cv2.medianBlur(shot, 5)
		_, shot, _ = cv2.split(shot)
		clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
		shot = clahe.apply(shot)

	if strategy == 'filter':
		shot = cv2.filter2D(shot, cv2.CV_8U, kernel)
		_, shot, _ = cv2.split(shot)

	if strategy == 'color-decompose':
		b = shot[:,:,0]
		g = shot[:,:,1]
		r = shot[:,:,1]
		_, b = cv2.threshold(b, 170, 255, cv2.THRESH_BINARY_INV)
		_, g = cv2.threshold(g, 170, 255, cv2.THRESH_BINARY_INV)
		_, r = cv2.threshold(r, 170, 255, cv2.THRESH_BINARY_INV)
		shot = b+g+r

	return shot


if __name__ == '__main__':
	read_video(video_path)
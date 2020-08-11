import os
import numpy as np
from scipy.spatial import distance
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

"""
I did my best to try many approaches to the given task.
Neither of them gives good results. I am not sure if
any approach would satisfy my intent or probably there is 
something wrong about the metric I implemented (it is
distance but perphaps I am missing something :)).
Still I selected at least those which worked.
"""

dataset_dir = 'dataset'
folders = {'goose': 'n01855672',
			'dog': 'n02091244',
			'wolf': 'n02114548',
			'lemure': 'n02138441',
			'bug': 'n02174001',
			'cannon': 'n02950826',
			'box': 'n02971356',
			'boat': 'n02981792',
			'lock': 'n03075370',
			'truck': 'n03417042',
			'bars': 'n03535780',
			'player': 'n03584254',
			'woman': 'n03770439',
			'rocket': 'n03773504',
			'poncho': 'n03980874',
			'coral': 'n09256479'}

def parse_args():

	# simplifying running pipeline
	parser = argparse.ArgumentParser(
		description='Find similar images in dataset')

	parser.add_argument(
		'--img_path', '-ip', dest='img_path',
		help='Input image path',
		default=os.path.join(
			dataset_dir, folders['truck'], 
			'n0341704200000027.jpg'))

	parser.add_argument(
		'--save', '-s', dest='save',
		help='Put any value other than 0 to save images \
		(saves image of the interest and the last of iteration)',
		default=0)

	return parser.parse_args()

def load_image(img_path, color_space='BGR'):

	# load image in needed color space
	img = cv2.imread(img_path)
	if color_space == 'BGR':
		return img
	elif color_space == 'HVS':
		return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	elif color_space == 'RGB':
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	else:
		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_paths():

	# get list of all images in the dataset
	imgs = []
	for v in folders.values():
		for file in os.listdir(os.path.join(dataset_dir, v)):
			imgs.append(os.path.join(dataset_dir, v, file))
	return imgs

def score_histograms(img_1, img_2, save=0):

	# comparing historgrams of the images
	img1_hist = cv2.calcHist(
		img_1, [0, 1], None, [255,255], None)
	cv2.normalize(img1_hist, img1_hist, 0, 255, cv2.NORM_MINMAX)
	img2_hist = cv2.calcHist(
		img_2, [0, 1], None,[255,255], None)
	cv2.normalize(img2_hist, img2_hist, 0, 255, cv2.NORM_MINMAX)

	if save != 0:
		cv2.imwrite('img_1_hist.png', img1_hist)
		cv2.imwrite('img_2_hist.png', img2_hist)

	score = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
	return score

def build_gist(img_1, img_2, save=0):

	# I am not sure I implemented it correctly but it
	# seems to follow the description I found in papers
	filters = []
	for theta in np.arange(0, np.pi, np.pi / 16):
		# get Gabor 32x32 filters
		kernel = cv2.getGaborKernel(
			(32, 32), 3.0, theta, 45, 0.75, 0, cv2.CV_32F)
		kernel /= 1.5*kernel.sum()
		filters.append(kernel)
	img_1_avg = []
	img_2_avg = []

	# devide image in 4x4 grid
	chunk = img_1.shape[0]//4
	regions = [(0, chunk), (chunk, chunk*2),
				(chunk*2, chunk*3), (chunk*3, chunk*4)]
	for kernel in filters:

		# convolve filters with each region and average them out
		fimg_1 = cv2.filter2D(img_1, cv2.CV_32F, kernel)
		fimg_2 = cv2.filter2D(img_2, cv2.CV_32F, kernel)
		for region in regions:
			decs_avg_1 = np.mean(
				fimg_1[region[0]:region[1], region[0]:region[1], :])
			decs_avg_2 = np.mean(
				fimg_2[region[0]:region[1], region[0]:region[1], :])
			img_1_avg.append(decs_avg_1)
			img_2_avg.append(decs_avg_2)

	if save != 0:
		cv2.imwrite('img_1_gist.png', fimg_1)
		cv2.imwrite('img_2_gist.png', fimg_2)

	return np.linalg.norm(np.array(img_1_avg)-np.array(img_2_avg))

def visualize_images(original, similar, grid=(2, 3), by='histograms'):

	# making grid for pictures
	fig, ax = plt.subplots(2, 3)
	original = load_image(original, 'RGB')
	ax[0][0].imshow(original, interpolation='none')
	ax[0][0].set_xticklabels([])
	ax[0][0].set_yticklabels([])
	ax[0][0].set_xlabel(f'original')

	im = 0
	for i in range(grid[0]):
		for j in range(grid[1]):
			if i == 0 and j==0:
				continue
			img = load_image(similar[im][0], 'RGB')
			ax[i][j].imshow(img, interpolation='none')
			ax[i][j].set_xticklabels([])
			ax[i][j].set_yticklabels([])
			sim = round(similar[im][1], 3)
			ax[i][j].set_xlabel(f'score: {sim}')
			#print()
			
			im += 1
	fig.suptitle(f'Similar by {by}')
	plt.show()

def score_centroids(img_1, img_2, centroids=2):
	"""This function is not implemented, I have 
	been struggling with it for a while 
	but could not make it work"""

	arr_1 = cv2.resize(
		img_1, (100, 100), interpolation = cv2.INTER_AREA)
	arr_2 = cv2.resize(
		img_2, (100, 100), interpolation = cv2.INTER_AREA)
	arr_1 = arr_1.reshape((-1,3)) #.astype(np.float32)
	arr_2 = arr_2.reshape((-1,3)) #.astype(np.float32)
	arr_1 = np.float32(arr_1)
	arr_2 = np.float32(arr_2)

	print(arr_1.ndim)
	print(arr_2.ndim)

	criterion = (
		cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

	arr_1, l_1, center_1 = cv2.kmeans(arr_1, centroids, 
		None, criterion, 10, cv2.KMEANS_RANDOM_CENTERS)
	arr_2, l_2, center_2 = cv2.kmeans(arr_1, centroids, 
		None, criterion, 10, cv2.KMEANS_RANDOM_CENTERS)

	print(center_1, center_2)

def score_corners(img_1, img_2, save=0):

	img_1 = cv2.medianBlur(img_1, 5)
	img_2 = cv2.medianBlur(img_2, 5)
	clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
	img_1 = clahe.apply(img_1)
	img_2 = clahe.apply(img_2)
	img_1 = cv2.morphologyEx(
		img_1, cv2.MORPH_GRADIENT, np.ones((3, 3), 
			dtype=np.uint8))
	img_2 = cv2.morphologyEx(
		img_2, cv2.MORPH_GRADIENT, np.ones((3, 3), 
			dtype=np.uint8))


	img_1 = np.float32(img_1)
	img_2 = np.float32(img_2)
	img_1 = cv2.cornerHarris(img_1, 2, 3, 0.04)
	img_2 = cv2.cornerHarris(img_2, 2, 3, 0.04)

	if save != 0:
		cv2.imwrite('img_1_corners.png', img_1)
		cv2.imwrite('img_2_corners.png', img_2)

	return np.linalg.norm(img_1-img_2)

def score_dct(img_1, img_2, save=0):

	channels_1 = []
	channels_2 = []
	img_1 = cv2.medianBlur(img_1, 5)
	img_2 = cv2.medianBlur(img_2, 5)

	# DCT seems to require grayscale images but I decided
	# to impliment it by channels instead
	for channel in range(img_1.shape[-1]):
		dct_1 = cv2.dct(img_1[:,:,channel].astype(np.float32))
		dct_2 = cv2.dct(img_2[:,:,channel].astype(np.float32))
		channels_1.append(dct_1)
		channels_2.append(dct_2)
	res_1 = np.dstack(channels_1)
	res_2 = np.dstack(channels_2)
	if save != 0:
		cv2.imwrite('img_1_dct.png', res_1)
		cv2.imwrite('img_2_dct.png', res_2)

	return np.linalg.norm(res_1-res_2)

def score_sift(img_1, img_2, save=0):

	# Neither images themselves nor keypoints yielded
	# desirred results but it worth to try
	img_1 = cv2.medianBlur(img_1, 5)
	img_2 = cv2.medianBlur(img_2, 5)

	sift = cv.SIFT_create()
	kp_1, des_1 = sift.detectAndCompute(img_1, None)
	kp_2, des_2 = sift.detectAndCompute(img_2, None)

	if save != 0:
		cv2.imwrite('img_1_sift.png', des_1)
		cv2.imwrite('img_2_sift.png', des_2)

	return np.linalg.norm(kp_1-kp_2)

def compare_images(img_path, func, by, color_space='BGR', save=0):

	top_5 = []
	selected_img = load_image(img_path, color_space)
	paths = get_paths()

	# setting generic pipeline to compare images
	for p in tqdm(paths):
		if p == img_path:
			continue
		another_img = load_image(p, color_space)
		top_5.append((p, func(
			selected_img, another_img, save)))

	# filter the best matches with the lowest distance
	top_5 = sorted(top_5, key=lambda tup: tup[1])[:5]
	visualize_images(img_path, top_5, by=by)


def main():
	args = parse_args()

	print('Comparing histograms...')
	compare_images(args.img_path, score_histograms, by='histograms', save=args.save)

	print('Detecting edges...')
	compare_images(args.img_path, score_corners, by='edges and corners', color_space='gray', save=args.save)

	print('Crunching SIFT points...')
	compare_images(args.img_path, score_corners, by='SIFT', color_space='gray', save=args.save)

	print('Calculating DCT...')
	compare_images(args.img_path, score_dct, by='discrete cosine transformation', save=args.save)

	print('Getting a gist... better make yourself a cup of tea :)')
	compare_images(args.img_path, build_gist, by='GIST', save=args.save)
	""" 
	Clustering I could not impliment, no matter 
	the format, dimentions, variations of 
	criterion it would throw an error:

	cv2.error: OpenCV(4.2.0) 
	/io/opencv/modules/core/src/kmeans.cpp:242: error: 
	(-215:Assertion failed) data0.dims <= 2 && 
	type == CV_32F && K > 0 in function 'kmeans'

	print('Clustering images...')
	compare_images(args.img_path, score_centroids)
	"""


if __name__ == '__main__':
	main()


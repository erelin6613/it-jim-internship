import os
import cv2
import numpy as np

video_path = os.path.join('part1', 'input_video.avi')

"""
This is my humble solution. Yet again there is the room for
improvment.

What did not work?
	* distinqushing between elipse and circle. Many
		literature suggested to use hough circles to
		determine those but I faced problem and
		problem again to implement them
	* working is grayscale, LAB, BGR or RGB failed
		to make clear masks to begin with
	* attempts to adjust bit rate of output video

Outputs:
	output_detected.avi - video with original
		video as a baseline and overlayed contours
		of detected shapes with geometric figure
		name near-ish it
"""


# Values for masks were borrowed from HW_1_review,
# solution 2. Thank you!
masks_lower = {'yellow': np.array([20, 40, 100]),
			'red': np.array([140, 50, 0]),
			'black': np.array([0, 0, 0])}
masks_upper = {'yellow': np.array([80, 100, 255]),
			'red': np.array([179, 150, 255]),
			'black': np.array([179, 255, 40])}


def read_video(path, write_output=False, out_file=None):

	if write_output and out_file is not None:
		cap = cv2.VideoCapture(path)
		ret, frm = cap.read()
		frm_count = 0

		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		fps = 30
		image_size = (width, height)
		writer = cv2.VideoWriter(out_file, fourcc, fps, image_size)
		cap.release()

	elif write_output and out_file is not None:
		raise Exception('`out_file` needed to wite a video')

	cap = cv2.VideoCapture(path)
	frm_count = 0
	key = None
	while True:

		ret, shot = cap.read()
		if ret:
			cv2.imshow('original', shot)
			shot = find_shapes(shot)
			cv2.imshow('with contours', shot)
			if write_output:
				writer.write(shot)
		else:
			break

		if key == ord('q'):
			wait_period = 0
		else:
			wait_period = 30
		key = cv2.waitKey(wait_period)

	cap.release()
	if write_output:
		writer.release()
	cv2.destroyAllWindows()

def find_shapes(frm, kernel_size=3):

	kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
	shot = frm.copy()

	# Convert to HSV colorspace and making empty image
	shot = cv2.cvtColor(shot, cv2.COLOR_BGR2HSV)

	# Blur the image
	shot = cv2.GaussianBlur(
		shot, (kernel_size, kernel_size), 3)

	# Get masks based on masks found for HSV and combining them
	masks = []
	for key in masks_upper.keys():
		masks.append(cv2.inRange(
			shot, masks_lower[key], masks_upper[key]))
	shot = masks[0]+masks[1]+masks[2]

	img = np.zeros(frm.shape)

	# Find all contours on the image keeping only external shapes
	contours, _ = cv2.findContours(
		shot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:

		# Let cv2 approximate what polygon contour resembles
		fig = cv2.approxPolyDP(
			contour, 0.03*cv2.arcLength(contour, True), True)

		# enclose in figure into a virtual rectangle
		px ,py, w, h = cv2.boundingRect(fig)

		# Skip noise and small shapes
		if w<20 or h<20:
			continue

		# Get coordinates to set the text
		x = fig.ravel()[0]+10
		y = fig.ravel()[1]+10

		# Skip lines
		if len(fig) == 2:
			continue

		# If tere are 3 points of polygon, draw it and put `triangle`
		if len(fig) == 3:
			cv2.putText(img, 'triangle', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0))
			img = cv2.drawContours(img, [contour], 0, (255, 0, 0), 3)

		# If there are 4 points calculate width/heigth ratio
		# If ratio is around 1 draw it and put `square`
		# Draw it and put `rectangle` otherwise
		elif len(fig) == 4:
			if (w/h>0.96 and w/h < 1.04) or (h/w>0.96 and h/w<1.04):
				cv2.putText(img, 'square', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255))
				img = cv2.drawContours(img, [contour], 0, (255, 0, 255), 3)
			else:
				cv2.putText(img, 'rectangle', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
				img = cv2.drawContours(img, [contour], 0, (0, 255, 0), 3)

		# If there are more points draw and call it a `circle`
		else:
			cv2.putText(img, 'circle', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
			#print(len(fig))
			img = cv2.drawContours(img, [contour], 0, (0, 0, 255), 3)

	# Overlay text and contours on original image
	img = img.astype('uint8')
	shot = cv2.addWeighted(frm, 0.5, img, 0.5, 0.0)
	return shot

if __name__ == '__main__':
	read_video(video_path, True, 'output_detected.avi')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = 'find_chocolate.mp4'
marker_path = 'marker.jpg'

"""
DISCLAMER: My solution is very poor, despite my efforts
and struggles with points, shapes, lines, in the
opticalflow part I was able only to put a stick on
the chocolate (I might got something wrong even
with diagonal but cannot spot it fast enough; still
if I understood correctly derived box from it
would not translate the homography transformation)

Not only my I do not know many things in opencv I
do not have as much time as I would like to for
exploring and experimenting more and deeper.

I am sorry, but this is what I could do.
Regards, Valentyna
"""


lk_params = dict( winSize  = (50, 50),
				  maxLevel = 4,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def track_marker_op(shot, marker, old_shot, pts=None):

	# back up original images, make grayscale copies
	original = shot.copy()
	original_m = marker.copy()
	original_o = old_shot.copy()

	shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
	marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
	old_shot = cv2.cvtColor(old_shot, cv2.COLOR_BGR2GRAY)

	# for the first frame points are None
	if pts is None:

		# so we need to find them
		orb = cv2.ORB_create()
		m_keys, m_fetures = orb.detectAndCompute(marker, None)
		s_keys, s_features = orb.detectAndCompute(shot, None)

		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		matches = matcher.match(m_fetures, s_features)
		to_track = matches

		pts = np.float32([s_keys[m.trainIdx].pt for m in to_track]).reshape(-1, 1, 2)

	# as I understood our task was to track once defined points,
	# so we are doing so with OpticalFlow
	pts, st, err = cv2.calcOpticalFlowPyrLK(old_shot, shot, pts, None, **lk_params)

	# box diagonal
	x1, x2, y1, y2 = original.shape[1], 0, original.shape[-1], 0
	for p in pts:
		print(p[0])
		if p[0][0]<x1: x1 = p[0][0]
		if p[0][0]>x2: x2 = p[0][0]
		if p[0][1]<y1: y1 = p[0][1]
		if p[0][1]>y2: y2 = p[0][1]
	rect = np.array([[x1, y1], [x2, y2]]).reshape(-1, 1, 2)
	mask = np.zeros(shot.shape)

	# this actually looks funny :)
	# original = cv2.polylines(original, [np.int32(pts)], True, (0, 255, 0),2, cv2.LINE_AA)

	pts_circles = pts.reshape(-1, 1, 2)

	for each in pts_circles:
		# draw the points we are tracking
		original = cv2.circle(original, (each[0][0], each[0][1]), 1, (0, 255, 0), 2)
	original = cv2.polylines(original, [np.int32(rect)], True, (0, 0, 255),2, cv2.LINE_AA)
	return original, pts


def track_marker_orb(shot, marker):
	"""The implementation is far far from perfect, 
	I will leave it as it is for now and come back 
	if I have a time. At this point the second part 
	is still neglected"""
	original = shot.copy()
	original_m = marker.copy()

	shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
	marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create()
	m_keys, m_fetures = orb.detectAndCompute(marker, None)
	s_keys, s_features = orb.detectAndCompute(shot, None)

	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = matcher.match(m_fetures, s_features)

	# I saw filtering actually worsened homograpy transformation
	# so decided not to filter them at all
	to_track = matches

	# points to track for marker and video itself
	dst_pts = np.float32(
		[s_keys[m.trainIdx].pt for m in to_track]).reshape(-1, 1, 2)
	src_pts = np.float32(
		[m_keys[m.queryIdx].pt for m in to_track]).reshape(-1, 1, 2)

	h, w = original_m.shape[0:2]
	pts = np.float32([[0, 0],[w,0], [w, h], [0, h], [0, 0]]).reshape(-1, 1, 2)
	try:
		# finding actual homography transformation
		M, mask = cv2.findHomography(src_pts, dst_pts,  cv2.RANSAC, 1.0)
		dst = cv2.perspectiveTransform(pts, M)

		# drawing actual transformed rectangle
		shot = cv2.polylines(
			original, [np.int32(dst)], True, (0, 255, 0),2, cv2.LINE_AA)
	except Exception as e:
		print(e)

	# draw only 7 matches out of bunch
	shot = cv2.drawMatches(original, s_keys, original_m, m_keys, 
		matches[:7], flags=2, outImg=None)

	# making sure our video is the same resolution as input
	shot = cv2.resize(shot, (original.shape[1], original.shape[0]))
	return shot


def find_roi(shot, marker):
	"""
	This function is not implemented.
	TL;DL: the idea was to select the specific
	region where OpticalFlow should look
	for features to track
	"""

	img = marker.copy()
	out = np.zeros(shot.shape)


	marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
	shot = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
	w, h = marker.shape
	marker = cv2.resize(marker, (int(0.75*w), int(0.75*h)))
	found = cv2.matchTemplate(shot, marker, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(found)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	roi = [top_left[0], bottom_right[0], top_left[1], bottom_right[1]]
	return roi


def read_video(path, write_output=False, out_file=None):

	marker = cv2.imread(marker_path)
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
	old_shot = None
	corners = None
	while True:

		ret, shot = cap.read()
		if ret:

			# this option tracks orb generated each iteration
			# shot = track_marker_orb(shot, marker)
			# cv2.imshow('tracking chocolate', shot)

			# this option only once selected features with OpticalFlow
			if old_shot is None:
				old_shot = shot
			shot, corners = track_marker_op(shot, marker, old_shot, corners)

			old_shot = shot
			cv2.imshow('tracking chocolate', shot)
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


if __name__ == '__main__':
	read_video(video_path)
	# read_video(video_path, write_output=True, out_file='found_chocolate_obr.mp4')
	# read_video(video_path, write_output=True, out_file='found_chocolate_op.mp4')
	# op_attempt_2()
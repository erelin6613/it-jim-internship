import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = os.path.join('part2', 'plan.png')
templates_dir = os.path.join('part2', 'symbols')

"""
Solution as usual is not perfect but the best I could do :)

What did not work?
	* finding contours
	* enchancing images both templates and original 
		(dilate-erode, thresholding, corner detection, etc)
	* feature matching (it propably could work but
		it required a lot of `plumbing` and processing)
	* despite my efforts to not label the same object as
		different once algorithm still does it (could
		be a logic error of mine but I cannot spot it now)

Outputs:
	detected.png (same directory) - image with detected
		templates bounded in the box and written
		file name without extension

	locations.json (same directory) - json file
		listing each detected box as dictionary
		with keys `name`, `x`, `y`, `width`,
		`heigth`, `correlation`

P.S. Picture is a little messy with detected objects.
	I hope json will be more informative file.

Regards, Valentyna Fihurska
"""

# this function is not applied in the code
def dilate_erode(img, kernel_size=3):
	kernel = np.ones((kernel_size, kernel_size), 
		dtype=np.uint8)
	d = cv2.dilate(img, kernel)
	e = cv2.erode(img, kernel)
	return d-e


def find_occurances(img_path, templates_dir, write_json=True):

	# read original image and convert to grayscale
	original = cv2.imread(img_path)
	original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

	# define placeholder for found locations
	loc_list = []

	# take each file in templates directory, read, 
	# convert to grayscale 
	for template in os.listdir(templates_dir):
		name = template.split('.')[0]
		temp_path = os.path.join(templates_dir, template)
		temp = cv2.imread(temp_path)
		temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

		# store height width of template
		temp_h, temp_w = temp.shape

		# find multiple matches of template, filter those
		# regions of which have high correlation values
		found = cv2.matchTemplate(original_gray, temp, cv2.TM_CCOEFF_NORMED)
		locations = np.where(found>=0.8)

		# iterrate over these regions
		for y, x in zip(*locations):
			corr_val = found[y][x]

			# check if we already have surrounding area recorded in list
			# if we do check if its correlation coefficient greater then
			# current one, move on if it is and redefine area otherwise
			for elem in loc_list:
				x_occ, y_occ = int(elem['x']), int(elem['y'])
				if (x in range(x_occ-10, x_occ+10)) and (y in range(
					y_occ-10, y_occ+10)):
					if float(elem['correlation']) > corr_val:
						break

			# log position and correlation to our list
			loc_info = {'name': str(name),
					'x': str(x), 'y': str(y), 
					'width': str(temp_w), 
					'heigth': str(temp_h),
					'correlation': str(corr_val)}
			loc_list.append(loc_info)

			# draw rectangle and text of corresponding object (file name)
			original = cv2.rectangle(original, (x, y), (x+temp_w, y+temp_h), (255, 0, 0), 1)
			cv2.putText(original, name, (x+5, y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

	# write locations to json
	if write_json:
		with open('locations.json', 'w') as f:
			json.dump(loc_list, f)
	plt.imshow(original[:,:,::-1])
	plt.show()
	cv2.imwrite('detected.png', original)


# this function was not implemented to the logical end
# I might come back to it later :)
def find_key_features(img_path, templates_dir):
	original = cv2.imread(img_path)
	original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	print(original.dtype)
	orb = cv2.ORB_create()
	org_keys, original_features = orb.detectAndCompute(original, None)
	print(org_keys[0].pt)
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	for template in os.listdir(templates_dir):
		temp = cv2.imread(os.path.join(templates_dir, template))
		temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
		print(temp.dtype)
		temp_keys, temp_features = orb.detectAndCompute(temp, None)
		matches = matcher.match(temp_features, original_features)
		img3 = cv2.drawMatches(original, 
			temp_keys, temp, [], matches[:10], None,
			flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		plt.imshow(img3)
		plt.show()


if __name__ == '__main__':
	find_occurances(img_path, templates_dir)
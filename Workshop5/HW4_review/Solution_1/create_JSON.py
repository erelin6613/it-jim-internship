import json
import numpy as np
import cv2

# change the path!
from week_4.rename_files import rename_files


def create_all_images_hists(all_images_names):
    """
    To create json file in create_JSON.py

    :param all_images_names:
    :return: all_images_hists:
    """
    all_images_hists = []
    for img in all_images_names:
        image = cv2.imread(img)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        all_images_hists.append(hist)
    return all_images_hists


def create_all_images_hog_hists(all_images_names):
    """
    To create json file in create_JSON.py

    :param all_images_names:
    :return: all_images_hists:
    """
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    all_images_hists = []
    for img in all_images_names:
        image = cv2.imread(img)
        hist = hog.compute(image, winStride=(8, 8), padding=(8, 8), locations=((10, 20),))
        all_images_hists.append(hist)
    return all_images_hists


def create_images_hog_hists_sorted_by_class(sorted_by_class):
    """
    To create json file in create_JSON.py

    :param sorted_by_class:
    :return: sorted_hists_by_class:
    """
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    images_hists_sorted_by_class = [[] for i in range(16)]
    next_class = 0

    for imgs_class in sorted_by_class:
        for img in imgs_class:
            image = cv2.imread(img)
            hist = hog.compute(image, winStride=(8, 8), padding=(8, 8), locations=((10, 20),))
            images_hists_sorted_by_class[next_class].append(hist)
        next_class += 1

    return images_hists_sorted_by_class


def create_images_hists_sorted_by_class(sorted_by_class):
    """
    To create json file in create_JSON.py

    :param sorted_by_class:
    :return: sorted_hists_by_class:
    """
    images_hists_sorted_by_class = [[] for i in range(16)]
    next_class = 0

    for imgs_class in sorted_by_class:
        for img in imgs_class:
            image = cv2.imread(img)
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            images_hists_sorted_by_class[next_class].append(hist)
        next_class += 1

    return images_hists_sorted_by_class


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


sorted_by_class, all_img_names, folder_names = rename_files()

images_hists_sorted_by_class = create_images_hists_sorted_by_class(sorted_by_class)

dict_class_hist = dict(zip(folder_names, images_hists_sorted_by_class))

dumped = json.dumps(dict_class_hist, cls=NumpyEncoder)

with open('images_hists_sorted_by_class.json', 'w') as fp:
    json.dump(dumped, fp, ensure_ascii=False)
    fp.close()

all_images_hists = create_all_images_hists(all_img_names)
dumped = json.dumps(all_images_hists, cls=NumpyEncoder)

with open('all_images_hists.json', 'w') as fp:
    json.dump(dumped, fp, ensure_ascii=False)
    fp.close()


images_hog_hists_sorted_by_class = create_images_hog_hists_sorted_by_class(sorted_by_class)
dict_class_hog_hist = dict(zip(folder_names, images_hog_hists_sorted_by_class))

dumped = json.dumps(dict_class_hog_hist, cls=NumpyEncoder)
with open('images_hog_hists_sorted_by_class.json', 'w') as fp:
    json.dump(dumped, fp, ensure_ascii=False)
    fp.close()

all_images_hog_hists = create_all_images_hog_hists(all_img_names)
dumped = json.dumps(all_images_hog_hists, cls=NumpyEncoder)

with open('all_images_hog_hists.json', 'w') as fp:
    json.dump(dumped, fp, ensure_ascii=False)
    fp.close()
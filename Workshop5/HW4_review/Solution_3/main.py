import cv2
import numpy as np
import os
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt


def canny(img):
    edges = cv2.Canny(img, 200, 250)
    # cv2.imshow('', edges)
    # cv2.waitKey(0)
    return edges


def hist_of_grad(img):
    edgs = canny(img)
    des, hog_image = hog(edgs, orientations=9, pixels_per_cell=(4, 4),
                        cells_per_block=(2, 2), block_norm='L1', visualize=True, multichannel=False)

    # des, hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4),
    #                      cells_per_block=(2, 2), block_norm='L1', visualize=True, multichannel=True)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    # ax1.axis('off')
    # ax1.imshow(img, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    #
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    #
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    hog_image = np.float32(hog_image)
    return hog_image


def load_imgs(fldr, pic):
    pth = os.path.join('dataset/', fldr)
    files = next(os.walk(pth))[2]
    # remove the original pic so not to compare it with itself
    files.remove(pic)
    imgs = []
    for f in files:
        pth = os.path.join('dataset/', fldr, f)
        img = cv2.imread(pth)
        imgs.append(img)
    return imgs, files


def gabor(img):

    scale = np.array([3, 7, 11, 15])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filtered = []
    # mean amplitude as a feature vector
    mean_amp = []
    # eigenvalues as feature vectors
    eig1 = []
    eig2 = []
    for sc in scale:
        for theta in np.arange(0, np.pi, np.pi / 8):
            kern = cv2.getGaborKernel((sc, sc), 5, theta, 10, 1, 0, cv2.CV_32F)
            kern /= 1.5 * kern.sum()

            filtered.append(cv2.filter2D(img, -1, kern))

    return filtered


def eigen_des(img):
    gabor_pic = gabor(img)
    # mean amplitude as a feature vector
    mean_amp = []
    # eigenvalues as feature vectors
    eig1 = []
    eig2 = []
    for gb in gabor_pic:
        mean_amp.append(np.abs(np.linalg.det(gb)))
        # norm of complex eigenvalues
        eig1.append(np.linalg.norm(np.linalg.eigvals(gb)[0]))
        eig2.append(np.linalg.norm(np.linalg.eigvals(gb)[1]))

    mean_amp = np.float32(mean_amp)
    eig1 = np.float32(eig1)
    eig2 = np.float32(eig2)

    # kinda like a gabor descriptor matrix
    # gabor_des = list(zip(mean_amp, eig1, eig2))
    # gabor_des = np.asarray(gabor_des, dtype=np.float32)
    return eig1 * eig2


def gist(img):
    gabor_pic = gabor(img)
    resized_pic = []
    for gb in gabor_pic:
        f = []
        for i in range(4):
            for j in range(4):
                f.append(np.mean(gb[21 * i: 21 * (i+1), 21 * j: 21 * (j+1)]))
        f = np.reshape(f, (4, 4))
        resized_pic.append(f)

    resized_pic = np.float32(resized_pic)
    resized_pic = np.reshape(resized_pic, (512, ))

    return resized_pic


def bf_matcher(des_img, des_img_dtst, lowe_prm):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_img, des_img_dtst, k=2)

    # apply ratio test
    good = []
    for m, n in matches:
        if m.distance < lowe_prm * n.distance:
            good.append(m)
    # print(len(good))

    # we need some 'goodness' parameter -- let's use distance normed by the length of good matches
    # the smaler the parameter -- the better the match
    dstnc = 0
    for g in good:
        dstnc += g.distance
    if len(good) != 0:
        dstnc = dstnc / len(good)
    return dstnc


def flann_matcher(des_img, des_img_dtst, lowe_prm):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_img, des_img_dtst, k=2)

    # apply ratio test
    good = []
    for m, n in matches:
        if m.distance < lowe_prm * n.distance:
            good.append(m)

    # we need some 'goodness' parameter -- let's use distance normed by the length of good matches
    # the smaller the parameter -- the better the match
    dstnc = 0
    for g in good:
        dstnc += g.distance
    if len(good) != 0:
        dstnc = dstnc / len(good)
    return dstnc


if __name__ == '__main__':
    fldr = 'n02971356'
    pic = 'n0297135600000040.jpg'
    flnm = os.path.join('dataset/', fldr, pic)
    img = cv2.imread(flnm)
    # cv2.imshow('original', img)
    # cv2.waitKey(0)
    imgs, file_nms = load_imgs(fldr, pic)

    # choose between hog, gabor eigenvalues and gist
    hog_des_img = hist_of_grad(img)
    gabor_des_img = eigen_des(img)
    gist_des_img = gist(img)

    goodness_prm = []
    for i in imgs:
        # choose between hog, gabor eigenvalues and gist
        # should be the same as for the image
        # also choose brute force or flann matcher

        # hog_des_img_dtst = hist_of_grad(i)
        # dstnc = bf_matcher(hog_des_img, hog_des_img_dtst, 0.86)
        # dstnc = flann_matcher(hog_des_img, hog_des_img_dtst, 0.85)

        # gabor_des_dtst = eigen_des(i)
        # dstnc = bf_matcher(gabor_des_img, gabor_des_dtst, 0.73)
        # dstnc = flann_matcher(gabor_des_img, gabor_des_dtst, 0.75)

        gist_des_dtst = gist(i)
        # dstnc = bf_matcher(gist_des_img, gist_des_dtst, 0.73)
        dstnc = flann_matcher(gist_des_img, gist_des_dtst, 0.69)

        goodness_prm.append(dstnc)

    # combine names of pis in the dataset
    dtst_des = list(zip(file_nms, goodness_prm))

    # sort for the goodness parameter and show the 5 first best matches
    dtst_des.sort(key=lambda x: x[1])

    # get rid of zeros
    new_dtst_des = []
    for el in dtst_des:
        if el[1] != 0:
            new_dtst_des.append(el)

    best_mtch = new_dtst_des[:5]
    for pic in best_mtch:
        pth = os.path.join('dataset/', fldr, pic[0])
        img = cv2.imread(pth)
        cv2.imshow(' ', img)
        cv2.waitKey(0)
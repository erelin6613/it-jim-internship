import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import json
from scipy.spatial import distance as dist


class Plotter:
    """ To understand our classes better from EACH side """

    def __init__(self, sorted_by_class):
        self.sorted_by_class = sorted_by_class

    def plot_1d_hist(self, channel):
        """
        Plot histogram of each class in concrete channel

        :param channel: string, r, g, b or v ( in hsv )
        :return: list of histograms of our concrete channel
        """
        all_hist = []
        fig, axes = plt.subplots(4, 4, figsize=(20, 12))  # grid of 4x4 subplots
        axes = axes.flatten()  # reshape from 4x4 array into 16-element vector

        for i in range(len(self.sorted_by_class)):
            sum_img = np.zeros((84, 1), np.uint8)

            for img in self.sorted_by_class[i]:
                image = cv2.imread(img)
                b, g, r = cv2.split(image)

                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)

                if channel == 'r':
                    sum_img = np.concatenate((r, sum_img), axis=1)
                elif channel == 'b':
                    sum_img = np.concatenate((b, sum_img), axis=1)
                elif channel == 'g':
                    sum_img = np.concatenate((g, sum_img), axis=1)
                elif channel == 'v':
                    sum_img = np.concatenate((v, sum_img), axis=1)

            hist = cv2.calcHist([sum_img], [0], None, [256], [0, 256])
            all_hist.append(hist)
            plt.sca(axes[i])
            axes[i].title.set_text((str(i) + " class, " + str(channel) + " channel"))
            plt.plot(hist)

        plt.show()

        return all_hist

    def plot_3d_hist(self, color_space):
        """
        Plot histogram of each class of concrete color space

        :param color_space: bgr or hsv
        :return: list of histograms of our concrete color space
        """
        all_hist = []
        fig, axes = plt.subplots(4, 4, figsize=(20, 12))  # grid of 4x4 subplots
        axes = axes.flatten()  # reshape from 4x4 array into 12-element vector

        for i in range(len(self.sorted_by_class)):
            sum_img = np.zeros((84, 1, 3), np.uint8)

            for img in self.sorted_by_class[i]:
                image = cv2.imread(img)
                if color_space == 'bgr':
                    sum_img = np.concatenate((image, sum_img), axis=1)
                if color_space == 'hsv':
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    sum_img = np.concatenate((hsv, sum_img), axis=1)

            hist = cv2.calcHist([sum_img], [0], None, [256], [0, 256])
            all_hist.append(hist)
            plt.sca(axes[i])
            axes[i].title.set_text((str(i) + " class, " + str(color_space) + " channels"))
            plt.plot(hist)

        plt.show()

        return all_hist

    def plot_hog_hists(self, hog):
        all_hist = []
        fig, axes = plt.subplots(4, 4, figsize=(20, 12))
        axes = axes.flatten()  # reshape from 4x4 array into 12-element vector

        for i in range(len(self.sorted_by_class)):
            sum_img = np.zeros((84, 1, 3), np.uint8)

            for img in self.sorted_by_class[i]:
                image = cv2.imread(img)
                sum_img = np.concatenate((image, sum_img), axis=1)

            hist = hog.compute(sum_img, winStride=(8, 8), padding=(8, 8), locations=((10, 20),))
            all_hist.append(hist)
            plt.sca(axes[i])
            axes[i].title.set_text((str(i) + " class,  hog features"))
            plt.plot(hist)

        plt.show()

        return all_hist


# Some metrics

OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),  # 1 - base
    # ("Chi-Squared", cv2.HISTCMP_CHISQR),  # 0 - base
    # ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA)  # 0 - base
)

SCIPY_METHODS = (
    ("Euclidean", dist.euclidean),  # 1111 - good
    ("Manhattan", dist.cityblock),  # 9038 - good
    ("Chebysev", dist.chebyshev))  # 384 - good


def np_hist_to_cv(np_histogram_output):
    """
    For compare histograms in right type

    :param np_histogram_output:
    :return:
    """
    counts = np_histogram_output
    return counts.ravel().astype('float32')


def filter_for_hist_method(images_hists_sorted_by_class, folder_names, list_all_hists):
    """
    Go deeper in histograms correlations.
    Understand with which color space or channel better corr

    :param images_hists_sorted_by_class:
    :param folder_names:
    :param list_all_hists:
    :return:
    """
    statistic_dict = {}
    q = 0
    for hist_type in list_all_hists:
        statistic_dict[str(q)] = {}

        for folder in folder_names:
            statistic_dict[str(q)][folder] = {}
            all_cor = []

            for i in range(99):
                hist = images_hists_sorted_by_class.get(folder)[i]
                hist = np.array(hist)
                hist1 = hist_type[folder_names.index(folder)]
                for key, method in OPENCV_METHODS:
                    d = cv2.compareHist(np_hist_to_cv(hist), np_hist_to_cv(hist1), method=method)
                    if key == "Correlation":
                        all_cor.append(d)

            mean_of_corr = statistics.mean(all_cor)
            median_of_corr = statistics.median(all_cor)

            statistic_dict[str(q)][folder] = [mean_of_corr, median_of_corr]
        q += 1
    return statistic_dict


def hog_filter_for_hist_method(images_hog_hists_sorted_by_class, folder_names, list_all_hists):
    """
    Go deeper in histograms correlations.
    Understand with which color space or channel better corr

    :param images_hists_sorted_by_class:
    :param folder_names:
    :param list_all_hists:
    :return:
    """
    statistic_dict = {}
    q = 0
    for hist_type in list_all_hists:
        statistic_dict[str(q)] = {}

        for folder in folder_names:
            statistic_dict[str(q)][folder] = {}
            all_cor = []

            for i in range(99):
                hist = images_hog_hists_sorted_by_class.get(folder)[i]
                hist = np.array(hist)
                hist1 = hist_type[folder_names.index(folder)]
                for key, method in OPENCV_METHODS:
                    d = cv2.compareHist(np_hist_to_cv(hist), np_hist_to_cv(hist1), method=method)
                    if key == "Correlation":
                        all_cor.append(d)

            mean_of_corr = statistics.mean(all_cor)
            median_of_corr = statistics.median(all_cor)

            statistic_dict[str(q)][folder] = [mean_of_corr, median_of_corr]
        q += 1
    return statistic_dict


def get_corr_random_img(all_img_names, folder_names, statistic_dict, classes_hist):
    """
    Compare random images histogram with classes histograms

    :param all_img_names:
    :param folder_names:
    :param statistic_dict:
    :param classes_hist:
    :return:
    """
    random_number = int(np.random.randint(0, 1600, 1))
    random_img = cv2.imread(all_img_names[random_number])
    cv2.imshow('random image', random_img)
    cv2.waitKey(0)

    statistic_of_random_img = {}
    hist = cv2.calcHist([random_img], [0], None, [256], [0, 256])

    for folder in folder_names:
        statistic_dict[folder] = {}
        all_cor = []
        hist1 = classes_hist[folder_names.index(folder)]
        d = cv2.compareHist(np_hist_to_cv(hist), np_hist_to_cv(hist1), method=cv2.HISTCMP_CORREL)
        all_cor.append(d)
        mean_of_corr = statistics.mean(all_cor)

        statistic_of_random_img[folder] = [mean_of_corr]

    return statistic_of_random_img


def method_hist(our_image_for_search, folder_names, statistic_dict, classes_hist):
    """
    Compare our_image_for_search histogram with classes histograms

    :param our_image_for_search:
    :param folder_names:
    :param statistic_dict:
    :param classes_hist:
    :return:
    """
    statistic_of_random_img = {}
    hist = cv2.calcHist([our_image_for_search], [0], None, [256], [0, 256])
    status = False
    potentional_folders = []

    for folder in folder_names:
        statistic_dict[folder] = {}
        all_cor = []
        hist1 = classes_hist[folder_names.index(folder)]
        d = cv2.compareHist(np_hist_to_cv(hist), np_hist_to_cv(hist1), method=cv2.HISTCMP_CORREL)
        all_cor.append(d)
        mean_of_corr = statistics.mean(all_cor)

        statistic_of_random_img[folder] = [mean_of_corr]

        if mean_of_corr > 0.7:
            potentional_folders.append(folder)
            status = True

    return status, potentional_folders


def get_top_5_images_from_class(potentional_folders, image_for_search_hist):
    with open('images_hists_sorted_by_class.json', 'r') as json_file:
        images_hists_sorted_by_class = json.load(json_file)
        images_hists_sorted_by_class = json.loads(images_hists_sorted_by_class)

    all_cor = {}
    all_hists = []
    all_d = []

    for folder in potentional_folders:
        for i in range(99):
            hist1 = images_hists_sorted_by_class.get(folder)[i]
            hist1 = np.array(hist1)
            all_hists.append(hist1)
    index_counter = 0

    for hist in all_hists:
        d = cv2.compareHist(np_hist_to_cv(image_for_search_hist), np_hist_to_cv(hist), method=1)
        all_d.append(d)
        all_cor[index_counter] = d
        index_counter += 1

    images_indexes = sorted(all_cor, key=all_cor.get, reverse=False)[:5]

    return images_indexes


def show_common_images(images_indexes, potentional_folders, folder_names, sorted_by_class):
    images_names = []
    all_images_names = []
    folders_indexes = []

    for folder in potentional_folders:
        folders_indexes.append(folder_names.index(folder))

    for folder_index in folders_indexes:
        for i in range(99):
            img_name = sorted_by_class[folder_index][i]
            all_images_names.append(img_name)

    for i in images_indexes:
        needed_image = all_images_names[i]
        images_names.append(needed_image)

    stack = np.zeros((84, 5, 3), np.uint8)

    for img_name in images_names:
        img = cv2.imread(img_name)
        stack = np.concatenate((stack, img), axis=1)

    cv2.imshow('we found it!', stack)
    cv2.waitKey(0)


def compare_hog_hist(our_image_for_search_hog, all_img_names):

    with open('all_images_hog_hists.json', 'r') as json_file:
        all_images_hog_hists = json.load(json_file)
        all_images_hog_hists = json.loads(all_images_hog_hists)

    all_d = []
    index_counter = 0
    all_cor = {}

    for hog_hist in all_images_hog_hists:
        hog_hist = np.array(hog_hist)
        d = cv2.compareHist(np_hist_to_cv(our_image_for_search_hog), np_hist_to_cv(hog_hist), method=1)
        all_d.append(d)
        all_cor[index_counter] = d
        index_counter += 1

    images_indexes = sorted(all_cor, key=all_cor.get, reverse=False)[:5]

    stack = np.zeros((84, 5, 3), np.uint8)
    for index in images_indexes:
        image = cv2.imread(all_img_names[index])
        stack = np.concatenate((stack, image), axis=1)

    cv2.imshow('we found it', stack)
    cv2.waitKey(0)



import numpy as np
import cv2.cv2
from skimage import feature
import matplotlib.pyplot as plt
import os,glob
from scipy import ndimage as nd
from skimage.filters import gabor_kernel
import sklearn.model_selection as model_selection
import re
from sklearn.preprocessing import minmax_scale
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

"""
You have to implement you own image retrieval solution. There is a dataset with images of a few classes.
Your code should take image filename as an input parameter, search for most similar images over
the whole dataset and visualize input image + 5 top matches. Feel free to use any classic features/descriptors
(histograms, Gabor, HOG etc.) except neural networks stuff. Select matches by the minimal distance.
"""

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

def compute_feats(image, kernels):
    # feats = np.zeros((len(kernels), 2), dtype=np.double)
    descriptors = []
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        descriptors.append(filtered.mean())
        descriptors.append(filtered.var())
        # feats[k, 0] = filtered.mean()
        # feats[k, 1] = filtered.var()
    return descriptors

class Gradient_histogram:
    def __init__(self, numPoints):
        # store the number of points and radius
        self.numPoints = numPoints


    def get_grad_features(self,grad_mag, grad_ang):
        # Init
        angles = grad_ang[grad_mag > 5]
        hist, bins = np.histogram(angles,self.numPoints)

        return hist

    def describe(self, pattern_img, eps=1e-7):
        # Calculate Sobel gradient
        grad_x = cv2.Sobel(pattern_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(pattern_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag, grad_ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        hist = self.get_grad_features(grad_mag, grad_ang).astype(np.float32)
        return hist

def extract_features():

    # Create dataset
    dataset = []

    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)


    for i in range(16):

        folder_path = r'dataset/{}'.format(i + 1)

        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            descriptors = []

            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            feature_extractor = LocalBinaryPatterns(256, 1)
            descriptor_template = feature_extractor.describe(img)
            descriptor_template_norm = minmax_scale(descriptor_template, feature_range=(0, 1))
            descriptors.append(descriptor_template)

            feature_extractor_2 = Gradient_histogram(256)
            descriptor_template_2 = feature_extractor_2.describe(img)
            descriptor_template_2_norm = minmax_scale(descriptor_template_2, feature_range=(0, 1))
            descriptors.append(descriptor_template_2_norm)

            descriptor_template_3 = compute_feats(img, kernels)
            descriptor_template_3_norm = minmax_scale(descriptor_template_3, feature_range=(0, 1))
            descriptors.append(descriptor_template_3_norm)

            # add filename
            d_filename = re.findall(r'\d+', filename)
            descriptors.append([d_filename])

            # add class
            descriptors.append([i + 1])

            flat_descriptors = [item for sublist in descriptors for item in sublist]

            dataset.append(flat_descriptors)

    dataset_arr = np.array([np.array(xi) for xi in dataset], dtype=object)

    y = dataset_arr[:, -1]
    im_names = dataset_arr[:, -2]
    X = dataset_arr[:, :-2]

    df_X = pd.DataFrame.from_records(X) # Shape (1600, 546)
    series_y = pd.Series(y)

    # lasso regression
    ols_mod = sm.OLS(np.asarray(series_y), np.asarray(df_X)).fit_regularized(alpha=0.2, L1_wt=0.01)
    regularized_regression_parameters = ols_mod.params
    regularized_regression_parameters_series = pd.Series(regularized_regression_parameters)
    regularized_regression_parameters_series = regularized_regression_parameters_series[
        regularized_regression_parameters_series == 0.0]

    to_drop_cols = regularized_regression_parameters_series.index.values.tolist()

    df_X_reg = df_X.drop(columns=to_drop_cols).copy() # New shape (1600, 76)

    im_names_df = pd.DataFrame.from_records(im_names)

    # Train Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(df_X_reg)

    # Add test image
    folder = 16
    im_name = 'n0925647900000007'
    path_to_test_im = r'dataset/{}/{}.jpg'.format(folder, im_name)

    test_im_to_show = cv2.imread(path_to_test_im)

    cv2.imshow("Test image", test_im_to_show)

    test_im = cv2.imread(path_to_test_im, cv2.IMREAD_GRAYSCALE)

    test_descriptors = []

    feature_extractor = LocalBinaryPatterns(256, 1)
    descriptor_template = feature_extractor.describe(test_im)
    descriptor_template_norm = minmax_scale(descriptor_template, feature_range=(0, 1))
    test_descriptors.append(descriptor_template)

    feature_extractor_2 = Gradient_histogram(256)
    descriptor_template_2 = feature_extractor_2.describe(test_im)
    descriptor_template_2_norm = minmax_scale(descriptor_template_2, feature_range=(0, 1))
    test_descriptors.append(descriptor_template_2_norm)

    descriptor_template_3 = compute_feats(test_im, kernels)
    descriptor_template_3_norm = minmax_scale(descriptor_template_3, feature_range=(0, 1))
    test_descriptors.append(descriptor_template_3_norm)

    flat_test_descriptors = [item for sublist in test_descriptors for item in sublist]

    test_im_features_list = [i for j, i in enumerate(flat_test_descriptors) if j not in to_drop_cols]
    test_im_features_arr = np.array(test_im_features_list)
    test_im_features_arr = test_im_features_arr.reshape(1, 76)

    test_distances, test_indices = nbrs.kneighbors(test_im_features_arr)

    test_indices_list = test_indices[0].tolist() # Iterate over to open images

    list_of_images = []

    for index_num in test_indices_list:
        folder = im_names_df.iloc[index_num][0]
        im_name = im_names_df.iloc[index_num][1]
        path_to_im = r'dataset/{}/n{}.jpg'.format(folder, im_name)

        im = cv2.imread(path_to_im)
        list_of_images.append(im)

    cv2.imshow("nearest im 1", list_of_images[0])
    cv2.imshow("nearest im 2", list_of_images[1])
    cv2.imshow("nearest im 3", list_of_images[2])
    cv2.imshow("nearest im 4", list_of_images[3])
    cv2.imshow("nearest im 5", list_of_images[4])


    cv2.waitKey(0)
    cv2.destroyAllWindows()

# It takes about 15 minutes to run, but "it works on my machine" =)

if __name__ == '__main__':
    extract_features()
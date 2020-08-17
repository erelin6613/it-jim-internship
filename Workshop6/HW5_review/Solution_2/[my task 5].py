import numpy as np
import cv2
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
import shutil
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing



def create_dir(name):
    try:
        # Create target Directory
        os.mkdir(name)
        print("Directory ", name, " Created ")
    except FileExistsError:
        print("Directory ", name, " already exists")

def split_dataset():
    directories = os.listdir(path="dataset_splitted")
    for dir in directories:
        create_dir("dataset_splitted/" + dir + "/train")
        create_dir("dataset_splitted/" + dir + "/validation")
        create_dir("dataset_splitted/" + dir + "/test")

        i = 1
        for file in os.listdir("dataset_splitted/" + dir):
            if file != "train" and file != "test" and file != "validation":
                if i <= 50: shutil.move("dataset_splitted/" + dir + '/' + file, "dataset_splitted/" + dir + '/train/' + file)
                elif i <= 70: shutil.move("dataset_splitted/" + dir + '/' + file,
                                        "dataset_splitted/" + dir + '/validation/' + file)
                else: shutil.move("dataset_splitted/" + dir + '/' + file,
                                        "dataset_splitted/" + dir + '/test/' + file)
                i += 1
    # shutil.move(source, destination)

def hist_describe(img, mode):

    if mode == 'RGB': pass
    elif mode == 'HSV': img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif mode == 'LAB': img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    bgr_planes = cv2.split(img)
    histSize = 16
    histRange = (0, 256)
    accumulate = False
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

    return (b_hist, g_hist, r_hist)

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

    directories = os.listdir(path="dataset_splitted")

    dataset = []

    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)


    load_full_features = True

    if load_full_features == False:

        for type_name in ('/train/', '/validation/', '/test/'):
            i = 1
            for dir in directories:

                    for file in os.listdir("dataset_splitted/" + dir + type_name):

                        descriptors = []

                        img = cv2.imread("dataset_splitted/" + dir + type_name + file, cv2.IMREAD_GRAYSCALE)
                        img_color = cv2.imread("dataset_splitted/" + dir + type_name + file)

                        feature_extractor = LocalBinaryPatterns(128, 1)
                        descriptor_template = feature_extractor.describe(img)
                        descriptor_template_norm = preprocessing.normalize(np.array(descriptor_template).reshape(1, -1))[0]
                        descriptors.append(descriptor_template_norm)

                        feature_extractor_2 = Gradient_histogram(128)
                        descriptor_template_2 = feature_extractor_2.describe(img)
                        descriptor_template_2_norm = preprocessing.normalize(np.array(descriptor_template_2).reshape(1, -1))[0]
                        descriptors.append(descriptor_template_2_norm)

                        descriptor_template_3 = compute_feats(img, kernels)
                        descriptor_template_3_norm = preprocessing.normalize(np.array(descriptor_template_3).reshape(1, -1))[0]
                        descriptors.append(descriptor_template_3_norm)

                        feature_extractor_4 = LocalBinaryPatterns(128, 2)
                        descriptor_template_4 = feature_extractor_4.describe(img)
                        descriptor_template_4_norm = preprocessing.normalize(np.array(descriptor_template_4).reshape(1, -1))[0]
                        descriptors.append(descriptor_template_4_norm)

                        for method in ('RGB', 'HSV', 'LAB'):
                            for hist in hist_describe(img_color, method):
                                descriptor_template_5_norm = preprocessing.normalize(np.array(hist).reshape(1, -1))[0]
                                descriptors.append(descriptor_template_5_norm)


                        d_filename = file

                        descriptors.append([d_filename])

                        descriptors.append([i])

                        flat_descriptors = []
                        for sublist in descriptors:
                            for item in sublist:
                                flat_descriptors.append(item)



                        dataset.append(flat_descriptors)

                        print(i)

                    i += 1

        dataset_arr = np.array([np.array(xi) for xi in dataset], dtype=object)

        x = dataset_arr[:, :-2]
        y = dataset_arr[:, -1]
        # print(np.unique(y, return_counts=True))
        names = dataset_arr[:, -2]

        np.savetxt('extracted_x.csv', x, fmt='%s', delimiter=',')
        np.savetxt('extracted_y.csv', y, fmt='%s', delimiter=',')
        np.savetxt('extracted_names.csv', names, fmt='%s', delimiter=',')

    else:
        x = np.loadtxt('extracted_x.csv', dtype = 'float', delimiter=',')
        y = np.loadtxt('extracted_y.csv', dtype = 'str', delimiter=',')
        names = np.loadtxt('extracted_names.csv', dtype = 'str', delimiter=',')
    # dtype = 'float'


    # df_X = pd.DataFrame.from_records(X)  # Shape (1600, 546)

    # y = dataset_arr[:, -1]
    # x = dataset_arr[:, :-2]
    # names = dataset_arr[:, -2]

    print(x.shape)

    # pca = PCA(n_components=600).fit(x)
    # print(pca.explained_variance_ratio_)
    # x = pca.transform(x)
    # print(x)

    x_train = x[0:800, :]
    x_validation = x[800:1120, :]
    x_test = x[1120:1600, :]

    print(x_train.shape)

    y_train = y[0:800]
    y_validation = y[800:1120]
    y_test = y[1120:1600]

    classify = True

    if classify == True:

        predictions_val_arr = []
        predictions_test_arr = []

        # test и validation перепутаны

        from sklearn.neighbors import KNeighborsClassifier
        clf1 = KNeighborsClassifier(n_neighbors = 13)

        from sklearn.tree import DecisionTreeClassifier
        clf2 = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=150,
                                     random_state=0)

        from sklearn.ensemble import RandomForestClassifier
        clf3 = RandomForestClassifier(criterion='entropy', max_depth=20, min_samples_leaf=2, min_samples_split=4,
                                     random_state=0)

        from sklearn.ensemble import GradientBoostingClassifier
        clf4 = GradientBoostingClassifier(random_state=0)

        from sklearn.neural_network import MLPClassifier
        clf5 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(128,), random_state=1)

        from sklearn.ensemble import AdaBoostClassifier
        clf6 = AdaBoostClassifier(n_estimators=2000, random_state=0)

        classifiers = (clf1, clf2, clf3, clf4, clf5, clf6)

        # for clf in classifiers:
        #
        #     clf.fit(x_train, y_train)
        #
        #     predictions_val = clf.predict(x_test)
        #     predictions_val_arr.append(predictions_val)
        #
        #     predictions_test = clf.predict(x_validation)
        #     predictions_test_arr.append(predictions_test)
        #
        #     predictions_train = clf.predict(x_train)
        #
        #     print(classification_report(y_train, predictions_train))
        #     print(classification_report(y_test, predictions_val))

        from sklearn.ensemble import VotingClassifier

        voting_clf = VotingClassifier(estimators=[('1', clf1), ('2', clf2), ('3', clf3), ('4', clf4), ('5', clf5), ('6', clf6)],
                                      voting='hard')
        print("Created voting classifier")
        voting_clf.fit(x_train, y_train)

        predictions_val = voting_clf.predict(x_test)
        predictions_val_arr.append(predictions_val)

        predictions_test = voting_clf.predict(x_validation)
        predictions_test_arr.append(predictions_test)

        predictions_train = voting_clf.predict(x_train)

        # print(classification_report(y_train, predictions_train))
        print(classification_report(y_test, predictions_val))
        print(confusion_matrix(y_test, predictions_val))


        print(predictions_val_arr)

        ml_preprocessed_val = []
        ml_preprocessed_test = []

        i = 0
        for j in predictions_val_arr:
            for a in range(16):
                ml_preprocessed_val.append([])
            for prediction in j:
                for a in range(16):
                    ml_preprocessed_val[i * 16 + a].append( int( int(prediction) == a ) )
            i += 1

        i = 0
        for j in predictions_test_arr:
            for a in range(16):
                ml_preprocessed_test.append([])
            for prediction in j:
                for a in range(16):
                    ml_preprocessed_test[i * 16 + a].append(int(int(prediction) == a))
            i += 1

        ml_preprocessed_val = np.array(ml_preprocessed_val)
        ml_preprocessed_test = np.array(ml_preprocessed_test)

        # print(ml_preprocessed_val)
        # print(ml_preprocessed_val.shape)
        #
        # print(ml_preprocessed_test)
        # print(ml_preprocessed_test.shape)

        np.savetxt('ml_preprocessed_val.csv', ml_preprocessed_val, fmt='%s', delimiter=',')
        np.savetxt('ml_preprocessed_test.csv', ml_preprocessed_test, fmt='%s', delimiter=',')

    else:

        ml_preprocessed_val = np.loadtxt('ml_preprocessed_val.csv', dtype='float', delimiter=',')
        ml_preprocessed_test = np.loadtxt('ml_preprocessed_test.csv', dtype='float', delimiter=',')

        # print(x_test)
        # print(x_test.shape)
        #
        # print(ml_preprocessed_val)
        # print(ml_preprocessed_val.shape)
        #
        # print(ml_preprocessed_test)
        # print(ml_preprocessed_test.shape)

    # Final ensembling

    # clf_score = [0.33, 0.08, 0.20, 0.29, 0.09, 0.18, 0.39, 0.60, 0.39, 0.22, 0.32, 0.23, 0.18, 0.05, 0.15, 0.64,
    #              0.18, 0.13, 0.33, 0.28, 0.17, 0.27, 0.15, 0.42, 0.07, 0.21, 0.19, 0.19, 0.21, 0.14, 0.22, 0.49,
    #              0.44, 0.31, 0.45, 0.36, 0.30, 0.21, 0.55, 0.66, 0.35, 0.45, 0.48, 0.25, 0.21, 0.05, 0.34, 0.74,
    #              0.27, 0.34, 0.36, 0.53, 0.25, 0.16, 0.49, 0.57, 0.35, 0.41, 0.45, 0.27, 0.24, 0.10, 0.29, 0.75,
    #              0.33, 0.31, 0.33, 0.37, 0.27, 0.25, 0.51, 0.61, 0.30, 0.44, 0.44, 0.40, 0.30, 0.23, 0.31, 0.79,
    #              0.24, 0.19, 0.18, 0.18, 0.04, 0.19, 0.21, 0.47, 0.12, 0.21, 0.29, 0.08, 0.00, 0.08, 0.10, 0.49]
    #
    # ml_preprocessed_val = ml_preprocessed_val.T
    # ml_preprocessed_test = ml_preprocessed_test.T
    #
    # results_val_max = []

    # for h in range(480):
    #     ml_preprocessed_val[h] = ml_preprocessed_val[h] * clf_score
    #
    #     # results_val_max.append(str(np.argmax(ml_preprocessed_val[h]) % 16 + 1))
    #
    #     for i in

    # for h in range(480):
    #     row = []
    #     for i in range(16):
    #         a = 0
    #         for j in range(6):
    #             a += ml_preprocessed_val[h][j * 16 + i]
    #         row.append(a)
    #     results_val_max.append(str(np.argmax(row) + 1))

    # for h in range(480):
    #     row = []
    #     for i in range(16):
    #         a = 0
    #         for j in range(6):
    #             a += ml_preprocessed_val[h][j * 16 + i]
    #         row.append(a)
    #     results_val_max.append(str(np.argmax(row) + 1))
    #
    #
    # print(y_test)
    # print(results_val_max)
    #
    # print(classification_report(y_test, results_val_max))


    # print(y_test.shape)
    #
    # clf.fit(ml_preprocessed_val, y_test)
    #
    # predictions_test = clf.predict(ml_preprocessed_test)
    # predictions_train = clf.predict(ml_preprocessed_val)
    #
    # print(classification_report(y_test, predictions_train))
    # print(classification_report(y_validation, predictions_test))


if __name__ == '__main__':
    extract_features()
    # split_dataset()
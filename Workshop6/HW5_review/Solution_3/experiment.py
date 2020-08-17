import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

from skimage.io import imread, imshow
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, plot_confusion_matrix

import matplotlib.pyplot as plt

import custom_transform

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import skimage

TRAIN_DIR = "splitted_dataset/train/"
TEST_DIR = "splitted_dataset/test/"
VAL_DIR = "splitted_dataset/val/"

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
val_images = [VAL_DIR+i for i in os.listdir(VAL_DIR)]
test_images =[TEST_DIR+i for i in os.listdir(TEST_DIR)]

classes = os.listdir("dataset/")
class_numbers = range(0,16)


def show_images(X, y, idx) :
  """
  Displays image with corresponding label using pyplot
  :param X: dataset with images
  :param y: dataset with labels
  :param idx: index of image to show
  :return:
  """
  image = X[idx]
  imshow(image)
  plt.title("This is a {}".format(y[idx]))
  plt.show()


def prep_data(images):
    """
     This function preprocesses input images. Firstly, it creates two arrays for images and labels, responsively.
    Then, it read images in the mentioned path, transform them to grayscale and put it to the array.
    Next, based on the image name, we create an appropriate label and store it in another array.
    After all, we return prepared dataset.
    :param images: path to the folder with images that will be processed
    :return: array with grayscale images and array with classes of this images
    """
    m = len(images)
    ROWS, COLS, CHANNELS = 84,84,3

    X = np.ndarray((m,ROWS,COLS))
    y = np.zeros((m,1))

    for i, img_file in enumerate(images):
        image = imread(img_file, as_gray=True)
        X[i,:] = image
        for idx, cls in enumerate(classes):
            if cls in img_file.lower():
                y[i,0] = class_numbers[idx]

    y = y.reshape(-1)

    return X, y

def prep_vectors(images):
    """
    Transfors images into feature vectors
    :param images: path to the folder with images that will be processed
    :return: return array with vectors and array with labels
    """
    m = len(images)
    ROWS, COLS, CHANNELS = 84,84,3

    X = np.ndarray((m,ROWS*COLS))
    y = np.zeros((m,1))

    for i, img_file in enumerate(images):
        image = imread(img_file, as_gray=True)
        image = image.reshape(-1)
        X[i,:] = image
        for idx, cls in enumerate(classes):
            if cls in img_file.lower():
                y[i,0] = class_numbers[idx]

    y = y.reshape(-1)

    return X, y

if __name__ == '__main__':
    x_train, y_train = prep_data(train_images)
    x_train, y_train = shuffle(x_train, y_train)

    x_val, y_val = prep_data(val_images)
    x_val, y_val = shuffle(x_val, y_val)

    x_test, y_test = prep_data(test_images)
    x_test, y_test = shuffle(x_test, y_test)

    hogify = custom_transform.HogTransformer(
        pixels_per_cell=(8, 8),
        cells_per_block=(2,2),
        orientations=9,
        block_norm='L2-Hys'
    )
    scalify = StandardScaler()

    HOG_pipeline = Pipeline([
        ('hogify', custom_transform.HogTransformer(
            pixels_per_cell=(8, 8),
            cells_per_block=(2,2),
            orientations=9,
            block_norm='L2-Hys')
        ),
        ('scalify', StandardScaler()),
        ('classify', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
    ])

    param_grid = [
        {'hogify__orientations': [9],
        'hogify__cells_per_block': [(3, 3)],
        'hogify__pixels_per_cell': [(8, 8), (14, 14)]},
        {'hogify__orientations': [9],
         'hogify__cells_per_block': [(3, 3)],
         'hogify__pixels_per_cell': [(14, 14)],
         'classify': [
             SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
             SVC(kernel='linear'),
             SVC(kernel='rbf', gamma='auto'),
             RandomForestClassifier(),
             AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0),
             AdaBoostClassifier(n_estimators=10, learning_rate=0.01, random_state=0),
         ]}
    ]

    grid_search = GridSearchCV(HOG_pipeline,
                               param_grid,
                               cv=3,
                               n_jobs=-1,
                               scoring='f1_weighted',
                               verbose=1,
                               return_train_score=True)

    grid_res = grid_search.fit(x_train, y_train)

    pp.pprint(grid_res.best_params_)

    # the highscore during the search
    print("the highscore during the search - ", grid_res.best_score_)


    best_pred = grid_res.predict(x_test)
    # metrics
    precision = precision_score(y_test, best_pred, average='weighted')
    recall = recall_score(y_test, best_pred, average='weighted')
    f1 = f1_score(y_test, best_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, best_pred)
    print("Precision - ", precision)
    print("Recall - ", precision)
    print("F1 score - ", f1)
    plot_confusion_matrix(grid_res, x_test, y_test)
    plt.show()
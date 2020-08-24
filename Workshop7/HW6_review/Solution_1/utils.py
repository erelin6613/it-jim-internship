import random
import os
from typing import List, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.feature import hog
from sklearn.metrics import classification_report, plot_confusion_matrix
from tensorflow.keras.models import load_model


SEED = 42


def get_file_paths(root_folder_path: str) -> Dict:
    random.seed(42)
    path2cls = {}
    for cls_dir in os.listdir(root_folder_path):
        dir_path = os.path.join(root_folder_path, cls_dir)
        cls = {os.path.join(dir_path, file): cls_dir for file in os.listdir(dir_path)}
        path2cls = dict(path2cls, **cls)

    return path2cls


def get_sample(sample_path: str) -> np.ndarray:
    img = cv2.imread(sample_path)
    return img


def get_features_vector(img) -> np.ndarray:
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(grey_img, block_norm='L2-Hys', pixels_per_cell=(16, 16), transform_sqrt=True)
    flatted_features = np.hstack([hog_features])
    return flatted_features


def create_dataset(root_folder_path: str = "dataset/", extract_features: bool = False) -> (np.ndarray, List):
    path2cls = get_file_paths(root_folder_path=root_folder_path)
    data = []
    labels = []
    for path, cls in tqdm(path2cls.items()):
        sample = get_sample(path)
        if extract_features:
          flated_features = get_features_vector(sample)
          data.append(flated_features)
        else:
            data.append(sample)
        labels.append(cls)

    return np.array(data), labels


def show_score(cls, X_test, y_test, grid=True):
    if grid:
        print(f'Best score for {cls.best_score_} using {cls.best_params_}')
        means = cls.cv_results_['mean_test_score']
        stds = cls.cv_results_['std_test_score']
        params = cls.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print(f' mean={mean:.3}, std={stdev:.3} using {param}')

    y_pred = cls.predict(X_test)
    if not grid:
        y_pred = y_pred.argmax(axis=1)
    print(classification_report(y_test, y_pred))

    # labels = list(set(y_test))
    # cm = plot_confusion_matrix(grid_result, X_test, y_test, display_labels=labels)
    # plt.xticks(rotation=45)
    # plt.show()


def inference(dataset_path, y_pred, y_test, label_encoder):
    img4check, labels4check = create_dataset(dataset_path)
    _, img4check, _, labels4check = train_test_split(img4check, labels4check, test_size=0.1, random_state=SEED,
                                                     stratify=labels4check)

    index = np.random.choice(y_pred.shape[0], 5, replace=False)
    fig = plt.figure(figsize=(30, 15))
    ax = [fig.add_subplot(1, 5, 1)]
    for i, idx in enumerate(index):
        ax.append(fig.add_subplot(1, 5, i + 1))
        pred_cls = label_encoder.inverse_transform([y_pred[idx]])[0]
        gt_cls = label_encoder.inverse_transform([y_test[idx]])[0]
        ax[-1].set_title(f"Prediction: {pred_cls}\n Ground truth: {gt_cls}")
        ax[-1].imshow(img4check[idx])
    plt.show()

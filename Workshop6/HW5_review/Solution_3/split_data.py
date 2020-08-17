# do not call without need

import os
import numpy as np
import shutil

source_root_dir = "dataset/"

root_dir = 'splitted_dataset'
os.makedirs(root_dir +'/train')
os.makedirs(root_dir +'/val')
os.makedirs(root_dir +'/test')

classes = os.listdir(source_root_dir)

# for dir in ['/train', '/val', '/test']:
#     for cls in classes:
#         os.makedirs(root_dir + dir + "/" + cls)


def split_dataset():
    directories = [os.path.join(source_root_dir, p) for p in sorted(os.listdir(source_root_dir))]
    dir_names = os.listdir(source_root_dir)

    for idx, dir in enumerate(directories):
        dir_name = dir_names[idx]
        images = [os.path.join(dir + "/", p) for p in sorted(os.listdir(dir + "/"))]
        train, val, test = np.split(np.array(images), [int(len(images)*0.7), int(len(images)*0.85)])
        for name in train:
            # shutil.copy(name, root_dir + "/train/" + dir_name)
            shutil.copy(name, root_dir + "/train/")
        for name in val:
            # shutil.copy(name, root_dir + "/val/" + dir_name)
            shutil.copy(name, root_dir + "/val/")
        for name in test:
            # shutil.copy(name, root_dir + "/test/" + dir_name)
            shutil.copy(name, root_dir + "/test/")

if __name__ == '__main__':
    split_dataset()

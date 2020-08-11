import os
import numpy as np
import cv2

folders = {'goose': 'n01855672',
			'dog': 'n02091244',
			'wolf': 'n02114548',
			'lemure': 'n02138441',
			'bug': 'n02174001',
			'cannon': 'n02950826',
			'box': 'n02971356',
			'boat': 'n02981792',
			'lock': 'n03075370',
			'truck': 'n03417042',
			'bars': 'n03535780',
			'player': 'n03584254',
			'woman': 'n03770439',
			'rocket': 'n03773504',
			'poncho': 'n03980874',
			'coral': 'n09256479'}


def organize_folders(src_dir, dst_dir,
	splits=(0.7, 0.15, 0.15)):
	"""
	Organize files into train-val-test structure.
	Even though I tried to make a function as much
	system agnostic as possible, beware it was 
	tested on Debian 10 only.

	:param: src_dir - folder with original images
					splited into classes
	:param: dst_dir - folder with new structure
	:param: splits - train-val-test fractions
	:return: None
	"""

	if not os.path.exists(dst_dir):
		os.system(f'mkdir {dst_dir}')
		os.system(f'mkdir {os.path.join(dst_dir, "train")}')
		os.system(f'mkdir {os.path.join(dst_dir, "val")}')
		os.system(f'mkdir {os.path.join(dst_dir, "test")}')
		for f in ['train', 'val', 'test']:
			for cl in folders.keys():
				os.system(f'mkdir {os.path.join(dst_dir, f, cl)}')

	for cl, folder in folders.items():
		files = np.random.permutation(os.listdir(os.path.join(src_dir, folder)))
		train_idx = int(len(files)*splits[0])
		val_idx = int(len(files)*splits[1])+train_idx
		train_f = files[:train_idx]
		val_f = files[train_idx:val_idx]
		test_f = files[val_idx:]
		src = os.path.join(src_dir, folder)
		target = os.path.join(dst_dir, "train", cl)
		for each in train_f:
			name_t = os.path.join(target, each)
			name_s = os.path.join(src, each)
			os.system(f'cp {name_s} {name_t}')
		target = os.path.join(dst_dir, "val", cl)
		for each in val_f:
			name_t = os.path.join(target, each)
			name_s = os.path.join(src, each)
			os.system(f'cp {name_s} {name_t}')
		target = os.path.join(dst_dir, "test", cl)
		for each in test_f:
			name_t = os.path.join(target, each)
			name_s = os.path.join(src, each)
			os.system(f'cp {name_s} {name_t}')
	print('Done')

def load_image(path, resize=True):
	img = cv2.imread(path)
	if resize:
		img = cv2.resize(img, (100, 100), cv2.INTER_AREA)
	return img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#print(load_set('dataset/train')[0].shape)
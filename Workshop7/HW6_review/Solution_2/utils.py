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

def load_image(path, resize=False):
	img = cv2.imread(path)
	if resize:
		img = cv2.resize(img, (100, 100), cv2.INTER_AREA)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
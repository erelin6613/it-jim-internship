from tkinter import Tk
from tkinter.filedialog import askopenfilename

# change the path!
from week_4.utils import *
from week_4.rename_files import rename_files


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

# change folders and images names
sorted_by_class, all_img_names, folder_names = rename_files()

# create our Plotter obj
plotter = Plotter(sorted_by_class)

# PLOT CLASSES HISTOGRAMS
classes_bgr_hist = plotter.plot_3d_hist(color_space='bgr')
classes_hsv_hist = plotter.plot_3d_hist(color_space='hsv')

classes_r_hist = plotter.plot_1d_hist(channel='r')
classes_g_hist = plotter.plot_1d_hist(channel='g')
classes_b_hist = plotter.plot_1d_hist(channel='b')
classes_hsv_v_hist = plotter.plot_1d_hist(channel='v')

classes_hog_hist = plotter.plot_hog_hists(hog)


list_all_hists = [
    classes_bgr_hist,
    classes_hsv_hist,
    classes_r_hist,
    classes_g_hist,
    classes_b_hist,
    classes_hsv_v_hist,
]

# read json with some useful data
with open('images_hists_sorted_by_class.json', 'r') as json_file:
    images_hists_sorted_by_class = json.load(json_file)
    images_hists_sorted_by_class = json.loads(images_hists_sorted_by_class)

with open('all_images_hists.json', 'r') as json_file:
    all_images_hists = json.load(json_file)
    all_images_hists = json.loads(all_images_hists)

with open('images_hog_hists_sorted_by_class.json', 'r') as json_file:
    images_hog_hists_sorted_by_class = json.load(json_file)
    images_hog_hists_sorted_by_class = json.loads(images_hog_hists_sorted_by_class)

with open('all_images_hog_hists.json', 'r') as json_file:
    all_images_hog_hists = json.load(json_file)
    all_images_hog_hists = json.loads(all_images_hog_hists)


# MEAN AND MEDIAN OF CORRELATIONS OF CLASS HISTOGRAM AND ALL IMAGES HISTOGRAMS IN CLASS
# cv2.HISTCMP_CORREL - works here the best, comparing correlation with other opencv and scipy methods
statistic_dict = filter_for_hist_method(images_hists_sorted_by_class, folder_names, list_all_hists)

list_hog = [classes_hog_hist]
statistic_dict_hog = hog_filter_for_hist_method(images_hog_hists_sorted_by_class, folder_names, list_hog)

print('MEAN AND MEDIAN OF CORRELATIONS OF CLASS HISTOGRAM AND ALL IMAGES HISTOGRAMS IN CLASS')
# print('mean and median for brg hists ', statistic_dict.get('0'))
# print('mean and median for hsv hists ', statistic_dict.get('1'))
# print('mean and median for r channel hists ', statistic_dict.get('2'))
# print('mean and median for g channel hists ', statistic_dict.get('3'))
print('mean and median for b channel hists ', statistic_dict.get('4'))
# print('mean and median for v channel hists ', statistic_dict.get('5'))
print('mean and median for hog hists ', statistic_dict_hog.get('0'))

# Lets explore our results and make the conclusion.
# Hog-histograms have bad correlation.
# We have the biggest correlation in brg colorspace and b channel, they are almost the same.
# Lets chose the b channel for further comparing and for distributing our random image in concrete class

Tk().withdraw()
filename = askopenfilename()


our_image_for_search = cv2.imread(filename)
hist = cv2.calcHist(our_image_for_search, [0], None, [256], [0, 256])

our_image_for_search_hog = hog.compute(our_image_for_search, winStride=(8, 8), padding=(8, 8), locations=((10, 20),))
our_image_for_search_hog = np.array(our_image_for_search_hog)

status_hist_method, potentional_folders = method_hist(our_image_for_search, folder_names, statistic_dict, classes_b_hist)

if status_hist_method:
    # Works good only for images, where color is obviously good feature, for example:
    # perfect for many images from class 780 ,many images from class 672,
    # perfect for : 780/045.jpg, 792/209.jpg, 244/028.jpg, 672/101.jpg, 548/038.jpg, 672/091.jpg
    # not bad for : 356/038.jpg
    # and maybe for many also, but I didn't test
    # I don't know exactly, but maybe for +-30% of all images it works fine
    print('Our method is histogram comparing')
    print('potentional folders for our img is ', potentional_folders)
    images_indexes = get_top_5_images_from_class(potentional_folders, hist)

    show_common_images(images_indexes, potentional_folders, folder_names, sorted_by_class)
else:
    # works perfect for: 244/023.jpg, 874/065.jpg
    # very good for :254/063.jpg, 504/093.jpg
    # not bad for : 479/042.jpg , 504/020.jpg
    compare_hog_hist(our_image_for_search_hog, all_img_names)


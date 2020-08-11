import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


from skimage import feature

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




def feature_extraction_demo():
    img, patt_list = load_example(2)

    feature_extractor = LocalBinaryPatterns(256,1)
    # feature_extractor = Gradient_histogram(8)


    descriptors = []
    for patt in patt_list:


        descriptor_template  = feature_extractor.describe(patt)
        descriptors.append(descriptor_template)



        division = 100
        nx = img.shape[0]//division
        ny = img.shape[1] // division
        outimg = np.zeros((nx,ny))
        for x in range(nx):
            print(x,nx)
            for y in range(ny):
                patch = img[x*division:(x+1)*division,y*division:(y+1)*division]
                descriptor = feature_extractor.describe(patch)
                match = cv2.compareHist(descriptor,descriptor_template,cv2.HISTCMP_KL_DIV)
                outimg[x,y] = match
        plt.plot(descriptor_template)
        plt.figure()
        plt.imshow(patt)
        plt.figure()
        plt.imshow(outimg)
        plt.show()


def load_patterns(patterns_folder, prep):
    onlyfiles = next(os.walk(patterns_folder))[2]
    imgs = []
    for f in onlyfiles:
        img = cv2.imread(patterns_folder + f, cv2.IMREAD_GRAYSCALE)
        img = prep(img)
        imgs.append(img)
    return imgs



def fft_matcher():
    # Load patterns
    img, patt_list = load_example(1)
    print('starting fft')
    imgfft = np.fft.fft2(img[:, :])
    imgfft[0, 0] = 0
    print('finished fft')
    i = 0

    for patt in patt_list:
        tmp = np.zeros((img.shape[0], img.shape[1]))
        tmp[0:patt.shape[0], 0:patt.shape[1]] = patt - np.mean(patt)
        tmpfft = np.abs(np.fft.fft2(tmp))
        tmpfft = tmpfft / np.max(tmpfft)

        filt = (tmpfft >0.2).astype(np.int8)*tmpfft

        filtered = imgfft * filt

        plt.imshow(np.log10(np.abs(filtered)+0.0000001))
        plt.figure()
        plt.imshow(patt)
        result = np.abs(np.fft.ifft2(filtered))
        result = result/(img**2+1)
        plt.figure()
        plt.imshow(result)
        plt.figure()
        plt.imshow(cv2.dilate(result, np.ones((10, 10), dtype=np.int8)))
        plt.show()

    return 0


def load_example(example):
    if example == 1:
        example_folder, example_fname = 'example1/', 'plan.png'
        prep = preprocess_plan
    else:
        example_folder, example_fname = 'example2/', 'plan.jpg'
        prep = preprocess_image
    patterns_folder = example_folder + 'patterns/'
    patt_list = load_patterns(patterns_folder, prep)
    img = cv2.imread(example_folder + example_fname, cv2.IMREAD_GRAYSCALE)
    img = prep(img)
    return img, patt_list


def preprocess_plan(img):
    img = cv2.erode(img,np.ones((4,4)))
    resize_ratio = 3
    img = cv2.resize(img,(img.shape[1]//resize_ratio,img.shape[0]//resize_ratio),interpolation=cv2.INTER_AREA)
    img = cv2.threshold(img,230,255,cv2.THRESH_BINARY)[1]

    return img

def preprocess_image(img):
    resize_ratio = 3
    img = cv2.resize(img,(img.shape[1]//resize_ratio,img.shape[0]//resize_ratio),interpolation=cv2.INTER_AREA)
    return img

if __name__ == '__main__':

    # fft_matcher()
    feature_extraction_demo()
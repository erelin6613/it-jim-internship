import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def k_means_demo():
    import numpy as np
    from sklearn.cluster import KMeans

    from time import time

    n_colors = 256

    # Load the I am back photo
    input_img = cv2.imread("leaves.jpg")
    input_img = cv2.resize(input_img,(input_img.shape[1]//3,input_img.shape[0]//3))

    # Convert to floats instead of the default 8 bits integer coding.
    input_img = np.array(input_img, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    w, h, d = tuple(input_img.shape)

    axis1= np.expand_dims(np.repeat(np.expand_dims(np.linspace(-1,1,w),axis=1),h,axis = 1),axis=2)*0.4
    axis2 = np.expand_dims(np.repeat(np.expand_dims(np.linspace(-1,1,h),axis=0),w,axis = 0),axis=2)*0.4


    input_img = np.concatenate((input_img,axis1,axis2),axis=2)

    w, h, d = tuple(input_img.shape)


    image_array = np.reshape(input_img, (w * h, d))

    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    print("done in %0.3fs." % (time() - t0))

    # Get labels for all points
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = kmeans.predict(image_array)
    print("done in %0.3fs." % (time() - t0))

    def recreate_image(codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image

    # Display all results, alongside original image
    img = recreate_image(kmeans.cluster_centers_, labels, w, h)[:,:,:3]
    img = (img*255).astype(np.uint8)
    cv2.imwrite('quanized.png',img)


def manifold_demo():
    values_to_extract = [1,2,3,4,5,6,7,8,9]
    reduce_dimensions = True
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    x_train, y_train, x_test, y_test = get_part_of_mnist(values_to_extract)

    clf = TSNE()


    if reduce_dimensions:
        dimensioner = PCA(n_components=32)
        print('Reducing dimensions')
        x_train = dimensioner.fit_transform(x_train)
        x_test = dimensioner.transform(x_test)

    print('Creating TSNE')
    part_to_process = 2500
    # x_embedded = x_train[:part_to_process,:2]
    x_embedded = clf.fit_transform(x_train[:part_to_process])
    for v in values_to_extract:
        chosen = y_train[:part_to_process]==v
        plt.plot(x_embedded[chosen,0],x_embedded[chosen,1],'x',label = str(v))
    plt.legend()
    plt.show()


def mnist_demo():
    reduce_dimensions= True
    from sklearn.svm import SVC


    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA, FastICA


    x_train, y_train, x_test, y_test = get_part_of_mnist()
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    clf = SVC(kernel='linear')
    # clf = SVC(kernel='rbf', gamma='auto')

    x_test_nc = x_test.copy()
    if reduce_dimensions:
        dimensioner = PCA(n_components=3)
        # dimensioner = FastICA(n_components=8)
        print('Reducing dimensions')
        x_train = dimensioner.fit_transform(x_train)
        x_test = dimensioner.transform(x_test)
        components = dimensioner.components_
        # for i in range(len(components)):
        #     plt.imshow(np.reshape(components[i], (28, 28)))
        #     plt.show()

        # for i in range(10):
        #     x_restored =dimensioner.inverse_transform(x_test[i])
        #     plt.imshow(np.concatenate((np.reshape(x_restored, (28, 28)),np.reshape(x_test_nc[i], (28, 28))),axis = 1))
        #     plt.show()

    print('Fitting the classifier')
    clf.fit(x_train[1:100], y_train[1:100])
    print('Predicting')
    predicted = clf.predict(x_test)
    print(y_test[0:30])
    print(predicted[0:30])
    print(np.mean((y_test == predicted).astype(np.long)))
    indexes = y_test != predicted
    right_answers = y_test[indexes]
    inferred = predicted[indexes]
    wrong = x_test_nc[indexes, :]
    for i in range(wrong.shape[0]):
        plt.imshow(np.reshape(wrong[i,:],(28,28)))
        plt.title(str(right_answers[i])+'->'+str(inferred[i]))
        plt.show()

def get_mnist_from_idx():
    import idx2numpy
    train_x = np.reshape(idx2numpy.convert_from_file('train-images.idx3-ubyte'),(-1,28*28))
    print(train_x.shape)
    train_y = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    test_x =   np.reshape(idx2numpy.convert_from_file('t10k-images.idx3-ubyte'),(-1,28*28))
    test_y =   idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')


    return test_x, train_x, test_y, train_y

def get_part_of_mnist(values_to_extract=None):
    X_test, X_train, y_test, y_train = get_mnist_from_idx()#

    if values_to_extract is None:
        values_to_extract = [3,5,8]

    mask = np.zeros_like(y_test,dtype=np.uint8)
    mask_train = np.zeros_like(y_train,dtype=np.uint8)
    for v in values_to_extract:
        mask+=(y_test==v).astype(np.uint8)
        mask_train += (y_train == v).astype(np.uint8)
    mask_b = mask>0
    mask_train = mask_train>0
    Test_subX = X_test[mask_b,:]
    Test_subY = y_test[mask_b]

    Train_subX = X_train[mask_train, :]
    Train_subY = y_train[mask_train]
    # Test_subX, Test_subY = shuffle(Test_subX, Test_subY)

    print(Train_subY.shape)
    print(Test_subY.shape)
    return Train_subX, Train_subY, Test_subX, Test_subY

if __name__ == '__main__':
    manifold_demo()
    # k_means_demo()
    # mnist_demo()
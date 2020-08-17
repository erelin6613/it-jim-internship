import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Reshape, Input, Concatenate, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD
import cv2

def save_the_model(model,fname):
    import json
    print('Saving the model')
    json_string = model.to_json()
    with open(fname + '.txt', 'w', encoding='utf-8') as f:
        json.dump(json_string, f, ensure_ascii=False)
    print('Saving the weights')
    model.save_weights(fname + '.hdf5')
    print('Done')

def get_model_from_files(fnm):
    import json
    from tensorflow.keras.models import model_from_json
    with open(fnm + '.txt') as infile:
        json_string = json.load(infile)
    model = model_from_json(json_string)
    model.load_weights(fnm + '.hdf5', by_name=False)
    model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    model.summary()
    return model


def y2indicator(y):
    '''This function is used to make one-hot bit encoding out of integers'''
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def load_mnist_from_local_files(folder = '../'):
    def loadMNIST(prefix, folder):
        intType = np.dtype('int32').newbyteorder('>')
        nMetaDataBytes = 4 * intType.itemsize

        data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte', dtype='ubyte')
        magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
        data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

        labels = np.fromfile(folder + "/" + prefix + '-labels.idx1-ubyte',
                             dtype='ubyte')[2 * intType.itemsize:]

        return data, labels

    x_train, y_train = loadMNIST("train", folder)
    x_test, y_test = loadMNIST("t10k", folder)
    return x_train, y_train, x_test, y_test

def example_of_CNN():
    '''Here we do all the training and watching the results'''
    # First, we load and prepare the inputs of the net
    # x_train, y_train, x_test, y_test = load_mnist_from_local_files()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    y_train_i = y2indicator(y_train)
    y_test_i = y2indicator(y_test)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
    print('Done with preparation of the dataset')

    print('Setting up the network')

    # Here is how you input any data
    input_shape = (28, 28, 1) #ok, shape of the data. Ommiting the first dimension!
    inputs = Input(shape=input_shape)
    #Convolutioinal layer
    x = Conv2D(10, kernel_size=(3, 3), activation='relu')(inputs)
    # Max pooling layer
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(20, kernel_size=(3, 3), activation='relu')(x)
    # Max pooling layer
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    # x = Dense(50)(x)
    predictions = Dense(10, activation='softmax')(x)

    # setting up the model
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # training!
    history = model.fit(x_train, y_train_i, batch_size=1024, epochs=20, validation_split=0.05)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    loss_and_metrics = model.evaluate(x_test, y_test_i, batch_size=256)
    print('Test loss:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])

def example_of_fully_connected():
    '''Here we do all the training and watching the results'''
    # First, we load and prepare the inputs of the net
    # x_train, y_train, x_test, y_test = load_mnist_from_local_files()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    y_train_i = y2indicator(y_train)
    y_test_i = y2indicator(y_test)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
    print('Done with preparation of the dataset')

    print('Setting up the network')

    # Here is how you input any data
    input_shape = (28, 28, 1) # shape of the data. Ommiting the first dimension!
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(100)(x)
    x = Dense(100)(x)
    x = Dense(100)(x)
    x = Dense(100)(x)
    x = Dense(90)(x)
    x = Dense(80)(x)
    x = Dense(50)(x)
    x = Dense(30)(x)
    predictions = Dense(10, activation='softmax')(x)

    # setting up the model
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    # opt = SGD(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # training!
    history = model.fit(x_train, y_train_i, batch_size=1024, epochs=100,validation_split=0.15)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    loss_and_metrics = model.evaluate(x_test, y_test_i, batch_size=256)
    print('Test loss:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])


def get_features(x):
    x_out = []
    for i in range(x.shape[0]):

        bin_example =(x[i,:,:]>127).astype(np.uint8)

        moments = cv2.moments(bin_example,True)
        feature_vector = []
        for key in moments.keys():
            feature_vector.append(moments[key])
        x_out.append(feature_vector)
    x_out = np.asarray(x_out,dtype=np.float32)
    return x_out

def get_hog(x):
    x_out = []
    hog_object = hog_class()
    for i in range(x.shape[0]):
        example = x[i,:,:]
        feature_vector=hog_object.compute(example)
        x_out.append(feature_vector)
    x_out = np.asarray(x_out,dtype=np.float32)
    return x_out

class hog_class():
    def __init__(self):
        winSize = (16, 16)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 2.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        self.winStride = (8, 8)
        self.padding = (14, 14)
        self.locations = ((14, 14),)

    def compute(self,x):
        hist = self.hog.compute(x, self.winStride, self.padding, self.locations)
        return np.squeeze(np.asarray(hist))





def example_of_feature_based():
    '''Here we do all the training and watching the results'''
    # First, we load and prepare the inputs of the net
    # x_train, y_train, x_test, y_test = load_mnist_from_local_files()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshaping the array to 3-dims
    x_train = x_train.reshape(x_train.shape[0], 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)
    x_train=get_hog(x_train)
    x_test=get_hog(x_test)
    # x_train=get_features(x_train)
    # x_test= get_features(x_test)

    # Normalizing the RGB codes by dividing it to the max RGB value.
    y_train_i = y2indicator(y_train)
    y_test_i = y2indicator(y_test)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('y_train one hot shape:', y_train_i.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])
    print('Done with preparation of the dataset')

    print('Setting up the network')

    # Here is how you input any data
    input_shape = x_train.shape[1:] # shape of the data. Ommiting the first dimension!
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(30)(x)
    x = Dense(10)(x)
    predictions = Dense(10, activation='sigmoid')(x)

    # setting up the model
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    opt = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # training!
    history = model.fit(x_train, y_train_i, batch_size=1024, epochs=100,validation_split=0.05)

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    loss_and_metrics = model.evaluate(x_test, y_test_i, batch_size=256)
    print('Test loss:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])


if __name__ == '__main__':
    example_of_CNN()
    # example_of_fully_connected()
    # example_of_feature_based()

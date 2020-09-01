import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)


def rotate_imgs(data):
    return np.rot90(data, 1, (1, 2))


def preproc_data(X_train, y_train, X_test, y_test):
    # Add an additional grayscale channel
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # Covert int array to float32 array
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Data normalization
    X_train /= 255
    X_test /= 255

    # One-Hot encoding
    n_classes = 10
    Y_train = to_categorical(y_train, n_classes)
    Y_test = to_categorical(y_test, n_classes)
    return (X_train, Y_train), (X_test, Y_test)


def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU())
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    BatchNormalization()
    model.add(Dense(512))
    model.add(LeakyReLU())
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])
    return model


def get_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), digits=4))


def train_on_rotated_data(X_train, y_train, X_test, y_test, train_gen, test_gen, prms):
    X_train = rotate_imgs(X_train)
    X_test = rotate_imgs(X_test)
    (X_train, y_train), (X_test, y_test) = preproc_data(X_train, y_train, X_test, y_test)

    batch_size = prms['batch_size']
    epochs = prms['epochs']
    train_step = prms['train_step']
    val_step = prms['val_step']

    train_generator = train_gen.flow(X_train, y_train, batch_size=batch_size)
    test_generator = test_gen.flow(X_test, y_test, batch_size=batch_size)

    model = create_model()
    model.fit(train_generator, steps_per_epoch=train_step, epochs=epochs, validation_data=test_generator,
              validation_steps=val_step)

    get_report(model, X_test, y_test)
    return model


def retrain_model(model_path, X_train, y_train, X_test, y_test, train_gen, test_gen, prms, mode='retrain head'):
    loaded_model = tf.keras.models.load_model(model_path)

    if mode == 'head':
        for layer in loaded_model.layers[:-1]:
            layer.trainable = False
    elif mode == 'classification layers':
        for layer in loaded_model.layers:
            if layer.name == 'flatten':
                break
            layer.trainable = False

    batch_size = prms['batch_size']
    epochs = prms['epochs']
    train_step = prms['train_step']
    val_step = prms['val_step']

    train_generator = train_gen.flow(X_train, y_train, batch_size=batch_size)
    test_generator = test_gen.flow(X_test, y_test, batch_size=batch_size)

    loaded_model.fit(train_generator, steps_per_epoch=train_step, epochs=epochs, validation_data=test_generator,
                     validation_steps = val_step)

    get_report(loaded_model, X_test, y_test)
    return loaded_model


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Create data generators with Data augmentation
    train_gen = ImageDataGenerator(shear_range=0.3, width_shift_range=0.08, height_shift_range=0.08, zoom_range=0.08)
    test_gen = ImageDataGenerator()

    epochs = 5
    batch_size = 128
    train_step = X_train.shape[0] // batch_size
    val_step = X_test.shape[0] // batch_size
    prms = {'epochs': epochs, 'batch_size': batch_size, 'train_step': train_step, 'val_step': val_step}

    # Train a base model
    model = train_on_rotated_data(X_train, y_train, X_test, y_test, train_gen, test_gen, prms)
    model.save('rotated_mnist_model')

    (X_train, y_train), (X_test, y_test) = preproc_data(X_train, y_train, X_test, y_test)
    get_report(tf.keras.models.load_model('rotated_mnist_model'), X_test, y_test)

    # Freeze the entire model except for the last layer and retrain only it
    trained_head_model = retrain_model('rotated_mnist_model', X_train, y_train, X_test, y_test, train_gen, test_gen,
                                       prms, mode='head')
    trained_head_model.save('trained_head_model')

    # Freeze all convolutional layers that extract features and retrain a classifier part
    trained_cls_layers_model = retrain_model('rotated_mnist_model', X_train, y_train, X_test, y_test,
                                             train_gen, test_gen, prms, mode='classification layers')
    trained_cls_layers_model.save('trained_cls_layers_model')

    # Retrain full model with weights of the previous model
    retrained_model = retrain_model('rotated_mnist_model', X_train, y_train, X_test, y_test,
                                    train_gen, test_gen, prms, mode=None)
    retrained_model.save('retrained_model')


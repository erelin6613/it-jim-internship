import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam, Nadam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from Task_6.utils import create_dataset, show_score, SEED, inference

np.random.seed(SEED)


def create_cnn_model(init_mode, optimizer, activation, lr, layers_num):
    model = Sequential()
    for i in range(layers_num-1):
        if i == 0:
            model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer=init_mode, input_shape=(84, 84, 3)))
            model.add(Dropout(0.1))
        else:
            model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='softmax'))
    model.compile(optimizer=optimizer(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    data, labels = create_dataset(root_folder_path="dataset")
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    le_labels = le.transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, le_labels, test_size=0.1,
                                                        random_state=SEED, stratify=le_labels)
    X_train, X_test = X_train / 255.0, X_test / 255.0

    train = False
    if train:
        batch_size = 512
        epochs = 75

        param_grid = dict(init_mode=['he_normal', 'he_uniform'],
                          activation=['relu', 'softplus'],
                          lr=[0.001, 0.005, 0.01, 0.05, 0.1],
                          optimizer=[Adagrad, Adam, Nadam],
                          layers_num=[3, 4, 5])

        # Create a neural network wrapper for grid search
        model_CV = KerasClassifier(build_fn=create_cnn_model, epochs=epochs,
                                   batch_size=batch_size, verbose=1)
        stoper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1)

        grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3, verbose=3)
        grid_result = grid.fit(X_train, y_train, validation_split=0.2, callbacks=[stoper])
        # Save the best model
        grid_result.best_estimator_.model.save('cnn_model.pkl')

        y_pred = grid_result.predict(X_test)
        show_score(grid_result, X_test, y_test)
        inference('dataset', y_pred, y_test, le)
    else:
        # The accuracy is pretty high and I'm sure it's not due to a well-trained network. I trained networks on Colab,
        # in theory, random_state should provide the same data split, but judging by the accuracy, the data from the
        # training set got into the test set while the model loading.
        model = load_model('cnn_model.pkl')
        y_pred = model.predict(X_test).argmax(axis=1)
        show_score(model, X_test, y_test, grid=False)
        inference('dataset', y_pred, y_test, le)

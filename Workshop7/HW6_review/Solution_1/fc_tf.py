import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam, Nadam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from Task_6.utils import create_dataset, show_score, SEED, inference

np.random.seed(SEED)


def create_fc_model(init_mode, optimizer, activation, lr, dropout_rate):
    model = Sequential()
    # for i in range(layers_num-1):
    #   if i == 0:
    #     model.add(Dense(128, kernel_initializer=init_mode, activation=activation, input_dim=729))
    #   else:
    #     model.add(Dense(128, kernel_initializer=init_mode, activation=activation))
    #   model.add(BatchNormalization())
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation, input_dim=729))
    model.add(BatchNormalization())
    if dropout_rate >= 0.5:
        model.add(Dropout(1-dropout_rate))
    model.add(Dense(64, kernel_initializer=init_mode, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, kernel_initializer=init_mode, activation=tf.nn.softmax))

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer(lr=lr),
              metrics=["accuracy"])
    return model


if __name__ == '__main__':
    data, labels = create_dataset(root_folder_path="dataset", extract_features=True)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    le_labels = le.transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, le_labels, test_size=0.1,
                                                        random_state=SEED, stratify=le_labels)

    # Data standardisation
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    train = False
    if train:
        batch_size = 512
        epochs = 35

        # Best parameters: 'activation': 'softplus', 'dropout_rate': 0.68, 'init_mode': 'he_normal', 'lr': 0.001,
        # 'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>
        param_grid = dict(init_mode=['he_normal', 'he_uniform'],
                          activation=['relu', 'softplus'],
                          lr=[0.001, 0.01, 0.05, 0.1],
                          optimizer=[Adagrad, Adam, Nadam],
                          dropout_rate=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        # Create a neural network wrapper for grid search
        model_CV = KerasClassifier(build_fn=create_fc_model, epochs=epochs,
                                   batch_size=batch_size, verbose=1)
        stoper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

        grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=3, verbose=3)
        grid_result = grid.fit(X_train, y_train, validation_split=0.2, callbacks=[stoper])
        # Save the best model
        grid_result.best_estimator_.model.save('fc_model.pkl')

        y_pred = grid_result.predict(X_test)
        show_score(grid_result, X_test, y_test)
        inference('dataset', y_pred, y_test, le)
    else:
        # The accuracy is pretty high and I'm sure it's not due to a well-trained network. I trained networks on Colab,
        # in theory, random_state should provide the same data split, but judging by the accuracy, the data from the
        # training set got into the test set while the model loading.
        model = load_model('fc_model.pkl')
        y_pred = model.predict(X_test).argmax(axis=1)
        show_score(model, X_test, y_test, grid=False)
        inference('dataset', y_pred, y_test, le)

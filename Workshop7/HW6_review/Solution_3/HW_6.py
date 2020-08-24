import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
You have to implement an image classification neural network for the same dataset as before. Feel free to reduce
the number of examples for each class in your dataset, if it takes too long to train on your hardware.

For this task, you should make at least two neural networks: the fully connected one that works on the same extracted
features as before and another one convolutional with class prediction at the end.

You can do it either in Keras or Pytorch. Better to do both.

As an output, you should provide your code, trained model files (2 pcs. at least), your dataset, and the same precision
metrics [calculated on test images] as you did before.

Your code should provide 3 execution modes/functions: train (for training new model), test (for testing the trained
and saved model on the test dataset), and infer (for inference on a particular folder with images or a single image).
"""


def save_the_model(model, fname):
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

def fully_connected_model():

    # Uploaded features from the last homework. The number of features reduced from 546 to 40 with L1 regularization.

    with open('train_dataset.pickle', 'rb') as f:
        train_dataset_df = pickle.load(f)

    target = train_dataset_df.pop('y')
    Y = np.asarray(target).astype(np.float32)
    X = np.asarray(train_dataset_df).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(16)
    features, labels = next(iter(dataset))
    train_dataset = dataset.shuffle(len(train_dataset_df)).batch(1)

    # The code for a moder training

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(40,)),  # input shape required
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(17)
    ])

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loss(model, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss_value, grads = grad(model2, features, labels)
    optimizer.apply_gradients(zip(grads, model2.trainable_variables))

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model2, x, y)
            optimizer.apply_gradients(zip(grads, model2.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model2(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    # Training log:

    """ Epoch 000: Loss: 2.813, Accuracy: 12.292%
        Epoch 050: Loss: 2.133, Accuracy: 51.979%
        Epoch 100: Loss: 1.946, Accuracy: 61.667%
        Epoch 150: Loss: 1.785, Accuracy: 68.750%
        Epoch 200: Loss: 1.673, Accuracy: 73.125%"""

    with open('train_accuracy_results.pickle', 'rb') as f2:
        train_accuracy_results = pickle.load(f2)

    with open('train_loss_results.pickle', 'rb') as f3:
        train_loss_results = pickle.load(f3)

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()

    # save_the_model(model2, 'model_fully_connected')
    #-------------------------------------------------------------------------------------

    with open('test_dataset.pickle', 'rb') as f2:
        test_dataset_df = pickle.load(f2)

    target_test = test_dataset_df.pop('y')
    Y_test = np.asarray(target_test).astype(np.float32)
    X_test = np.asarray(test_dataset_df).astype(np.float32)
    dataset_t = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = dataset_t.shuffle(len(test_dataset_df)).batch(1)

    model3 = get_model_from_files('model_fully_connected')

    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_dataset:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model3(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result())) # Test set accuracy: 15.625%


if __name__ == '__main__':
    fully_connected_model()
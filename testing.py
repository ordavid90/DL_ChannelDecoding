import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

fashion_mnist = keras.datasets.fashion_mnist
(train_data, train_label), (test_data, test_label) = fashion_mnist.load_data()

print(train_data.shape)
print(train_label[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_data = train_data / 255
test_data = test_data / 255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # reformatting the data from 28x28 matrix to 784 vector
    keras.layers.Dense(128, activation=('relu')),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits='true'),
              metrics=['accuracy'])

model.fit(train_data, train_label, epochs=10)

test_loss, test_accuracy = model.evaluate(test_data, test_label, verbose=2)
print('\nTest accuracy:', test_accuracy)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_data)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


#   # Plot the first X test images, their predicted labels, and the true labels.
#   # Color correct predictions in blue and incorrect predictions in red.
# num_cols = 3
# num_rows = 5
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions[i], test_label, test_data)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions[i], test_label)
# plt.tight_layout()
# plt.show()
pred = np.argmax(predictions, axis=1)
model_acc = np.equal(test_label, pred)
print(np.shape(model_acc))
model_acc = sum(model_acc)
print(model_acc)
print(model_acc / len(test_label))

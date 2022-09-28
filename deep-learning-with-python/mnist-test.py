#!/usr/bin/env python3

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# load the mnist dataset
(train_images, train_labels), (test_images, test_data) = mnist.load_data()

# analysing the training data
print(train_images.shape) # show the structure of training data
print(len(train_images)) # show the number of training images
print(train_labels) # sneek peek into the traininig labels

#analysing the test data
print(test_images.shape) # show the structure of the test data
print(len(test_images)) # show the number of test images
print(test_labels) # sneek peek into the test labels

# creating our model
model = Sequential([
    Dense(512, activation='relu'), # layer 1
    Dense(10, activation='softmax') # output layer
])

# compilation - the mechanism through which the model will update
# itself based on the training data it sees so as to improve performance
model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
)

# before training we process the data by reshaping it into the shape the model
# expects and scaling so that values are in the range [0, 1] interval (normalizing)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test-images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# training/fitting the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# using the model to make predictions
test_digits = test_images[0:10] # sample 10 images from the test set
predictions = model.predict(test_digits)
plt.imshow(predictions[0], cmap=plt.cm.binary)
plt.show()
print(f"Predicted: {predictions[0].argmax()}")

# Evaluating the model on new data - how good is our model at classifying 
# never seen before data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"test_accuracy: {test_accuracy}")

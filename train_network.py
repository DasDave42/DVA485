import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime

# Check for a GPU
# Prerequisites for this to work: Python 3.7, CUDA 10.X, Cudnn 3.7
print("Tensorflow built with CUDA: " + str(tf.test.is_built_with_cuda()))
print("List of pysical devices: " + str(tf.config.list_physical_devices('GPU')))

# Read training labels
data_labels = pd.read_csv("training_data.csv", sep=";")

# Create data object from all the images
images = []
for id, path in zip(data_labels["id"], data_labels["path"]):
    images.append(cv2.imread(path))

# Create images and labels array
images = np.array(images)
labels = data_labels["angle"]

# Preprocessing: resizing the images to 64x64 and scaling pixel values from 0-1
images = np.array([cv2.resize(img, (64, 64)) for img in images])
images = images.astype("float32") / 255

# Do train test split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=52)

print(f"\n Train shapes: Images: {train_images.shape}, Labels: {train_labels.shape}")
print(f"\n Training with: {train_images.shape[0]} samples, Testing with: {test_images.shape[0]} samples")

# Determine input and output shapes of network
input_shape = images[0].shape
output_shape = 1

# Build model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(250, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(20, activation="relu"),
        layers.Dense(output_shape),
    ]
)

model.summary()

# Setup batch size and epochs for training
batch_size = 128
epochs = 15

# Compile model
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

# Setup log folder and callback for tensorboard
log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1, callbacks=[tensorboard_callback])

# Testing model on test batch
score = model.evaluate(test_images, test_labels, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Testing prediction with example image
# Load image 0 and label 0
image_path_df = data_labels[data_labels['id'] == 0]
path = image_path_df["path"]
true_angle = image_path_df["angle"]

# Preprocess image
image = cv2.imread(path[0])
image = cv2.resize(image, (64, 64))
image = image.astype("float32") / 255
image = np.array([image])

# Predict on image
result = model.predict(image)

print(f"predicting image {path}, expected: {true_angle}, result: {result}")

# Ask if user wants to save trained mode
print(f"\nTraining finished!")
while True:
    query = input('Save model? \n y/n: ')
    Fl = query[0].lower()
    if query == '' or not Fl in ['y', 'n']:
        print('Please answer with yes or no!')
    else:
        break
if Fl == 'y':
    path = "models\\model-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"
    model.save(path)
    print("Model saved in: " + path)
if Fl == 'n':
    print("OK! exiting")
    exit(0)

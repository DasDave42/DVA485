import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# read training labels
data_labels = pd.read_csv("training_data.csv", sep=";")

# create data object from all the images
images = []
for id, path in zip(data_labels["id"], data_labels["path"]):
    images.append(cv2.imread(path))

images = np.array(images)

labels = data_labels["angle"]

# preprocessing
images = images.astype("float32") / 255

# do train test split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=52)

print(f"\n Train shapes: Images: {train_images.shape}, Labels: {train_labels.shape}")
print(f"\n Training with: {train_images.shape[0]} samples, Testing with: {test_images.shape[0]} samples")

input_shape = (416, 416, 3)
output_shape = 1

# build model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(500, activation="relu"),
        layers.Dense(250, activation="relu"),
        layers.Dense(50, activation="relu"),
        layers.Dense(output_shape, activation="relu"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)

score = model.evaluate(test_images, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

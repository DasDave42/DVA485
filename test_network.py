import os
import cv2
import time
from tensorflow import keras
import numpy as np

# Path where the videos are saved
videopath = ".\\Video_Dataset_Lab#4"

# Load model
model = keras.models.load_model("models/final-model.h5")

cv2.namedWindow("image")

# Step through all files in video folder
for video in os.listdir(videopath):
    # Open video
    cam = cv2.VideoCapture(os.path.join(videopath, video))

    # Use fps for proper video speed
    fps = int(cam.get(cv2.CAP_PROP_FPS))

    print(f"Processing video :{video}, fps: {fps}")

    while cam.isOpened():
        # Step through all images
        ret, image = cam.read()
        if ret is True:
            # Sleeps for seconds/fps to play video in the correct speed
            time.sleep(1/fps)

            # Preprocess image for prediction: resize to 64x64 and scale from 0-1
            prep = cv2.resize(image, (64, 64))
            prep = prep.astype("float32") / 255
            prep = np.array([prep])

            # Make prediction on preprocessed image
            prediction = model.predict(prep)
            # Unpack prediction
            predicted_angle = prediction[0][0]
            # Scale prediction to get from -1 - 1 to 0 - 1
            scaled_angle = (predicted_angle + 1) / 2
            # Calculate point on image x = image width * predicted angle, y = middle of image height
            predicted_point = (int(len(image[0]) * scaled_angle), int(len(image[1]) / 2))

            print(f"Prediction: Angle: {predicted_angle}, Point: {predicted_point}")

            # Draw line and write predicted angle on output image
            cv2.line(image, (int(len(image[0]) / 2), len(image[1])), predicted_point, (0, 255, 0), 1)
            cv2.putText(image,
                        f"Angle: {predicted_angle}",
                        (int(predicted_point[0] - 75),
                         int(predicted_point[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1)

            cv2.imshow("image", image)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                exit(0)

        else:
            break

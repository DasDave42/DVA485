import os
import cv2
import time

# path where the videos are saved
videopath = ".\\Video_Dataset_Lab#4"
# path where the generated images are saved
image_path = ".\\Image_Dataset"

# step: only every x image is saved
offset = 1

# initialize id to number images
id = 0

# initialize counter to track number of processed images
counter = 0

# step through all files in video folder
for video in os.listdir(videopath):
    # open video and step through images
    cam = cv2.VideoCapture(os.path.join(videopath, video))

    fps = int(cam.get(cv2.CAP_PROP_FPS))

    print(f"Processing video :{video}, fps: {fps}, saving every {offset} image")

    while cam.isOpened():
        ret, image = cam.read()
        if ret is True:
            if counter % offset == 0:
                path = os.path.join(image_path, f"image_{id}.png")
                print(f"Saving image: {id}, video: {video}, counter: {counter}, path: {path}")
                cv2.imwrite(path, image)
                id += 1
            counter += 1
        else:
            break

print(f"Generating finished, {id + 1} images saved, {counter + 1} images processed")
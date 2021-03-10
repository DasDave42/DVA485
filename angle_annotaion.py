import os
import cv2
import pandas as pd
import numpy as np

# callback for angle selection
def angle_selection (event, x, y, flags, param):
    global selected_point, image, angle, blank_image
    if event == cv2.EVENT_LBUTTONDOWN:
        # use blank image if a line is already drawn (to avoid duplicated lines)
        if angle is not None:
            image = blank_image.copy()

        selected_point = (x, y)
        # draw line between clicked point and bottom center
        cv2.line(image, (int(len(image[0]) / 2), len(image[1])), selected_point, (0, 255, 0), 1)

        # calculate angle range=(-1, 1) (0 is middle)
        # fraction (range 0 - 1) => * 2 (range 0 - 2) => -1 (range -1 - 1)
        angle = ((selected_point[0] / len(image[0])) * 2) - 1

        # display currently selected angle near selected point
        cv2.putText(image,
                    f"Angle: {angle}",
                    (int(selected_point[0] - 75),
                     int(selected_point[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1)

        cv2.imshow("image", image)


# path where the generated images are saved
image_folder = ".\\Image_Dataset"
# Path where the training data is stored
training_data_path = ".\\training_data.csv"

# try loading a train csv (for continuation)
try:
    train_data = pd.read_csv(training_data_path, sep=";")
except FileNotFoundError:
    red_warning_color = '\033[93m'
    revert_color = '\033[0m'
    print(f"{red_warning_color} Training Data file not found! {revert_color}")
    while True:
        query = input('create new Data file csv? \n y/n: ')
        Fl = query[0].lower()
        if query == '' or not Fl in ['y', 'n']:
            print('Please answer with yes or no!')
        else:
            break
    if Fl == 'y':
        train_data = pd.DataFrame(columns=['id', 'path', 'angle'])
    if Fl == 'n':
        print("OK! exiting")
        exit(0)

# init id param (to continue labeling if it was aborted in the process)
id = train_data['id'].max()

# if id == None assume the csv was newly created and set it to 0
if id is np.nan:
    id = 0

global selected_point, image, angle, blank_image


while True:
    # try to load next image for annotation
    # images HAVE! to be in the format as created by the save_videos_as_images.py
    path = os.path.join(image_folder, f"image_{id}.png")
    image = cv2.imread(path)
    # if the image couldn't be read display an error message and exit
    if image is None:
        print(f"Image couldn't be read, path:{path}")
        exit(1)

    # draw circle in the middle on the bottom from where the angle is calculated
    cv2.circle(image, (int(len(image[0]) / 2), len(image[1])), 5, (0, 255, 0), -1)

    blank_image = image.copy()
    angle = None

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", angle_selection)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(13) & 0xFF
        # Enter key
        if key == 13:
            if angle is not None:
                newline = pd.DataFrame([[id, path, angle]], columns=['id', 'path', 'angle'])
                train_data = train_data.append(newline, ignore_index=True)
                print(f"Image labeled id:{id}, angle:{angle} path: {path}")
                id += 1
                break
            else:
                print("Select a point on the image and press Enter to save")

    # save training data before loading new img
    train_data.to_csv(training_data_path, sep=";", columns=['id', 'path', 'angle'])

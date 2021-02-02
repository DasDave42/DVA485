import numpy as np
import cv2

cap = cv2.VideoCapture('Video_1.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    # set grayscale and apply gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)

    # calc sobel in X and Y direction
    sobelX = cv2.Sobel(gauss, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(gauss, cv2.CV_64F, 0, 1)

    # combine X and Y sobel edges and normalize for displaying
    sobel = np.sqrt(np.square(sobelX) + np.square(sobelY))
    sobel *= (255 / sobel.max())
    sobel = sobel.astype(np.uint8)

    # calc median of image to set upper and lower boundary correctly and calculate canny
    med_val = np.median(gauss)
    canny = cv2.Canny(gauss, 100, 125)

    # detect lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 75  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(canny, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    # set which image shall be displayed
    out = frame

    # display image and print stats
    cv2.imshow('out', out)
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{width} x {height} | fps:{fps} | gray_mean:{med_val}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



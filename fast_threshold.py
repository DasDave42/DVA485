import cv2

img = cv2.imread("./Lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create(threshold=45)
keypoints = fast.detect(gray, None)
print("No kp Detected: " + str(len(keypoints)))

img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('edge', img)
cv2.waitKey()
cv2.destroyAllWindows()
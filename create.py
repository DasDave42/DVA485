def create(path,annpath):
    import cv2
    ref_point = []
    crop = False


    def shape_selection(event, x, y, flags, param):
        global ref_point, crop
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))
            cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
            pr = str(ref_point)
            pr = pr.replace('[', '')
            pr = pr.replace(']', '')
            pr = pr.replace('(', '')
            pr = pr.replace(')', '')
            pr = pr.replace(' ', '')
            print(pr)
            coordinates.append(pr)
            '''class_num = input('enter class num:')
            file = open('C:\\Users\\ALI\\Desktop\\New Text Document (2).txt', 'a+')
            file.write(pr)
            file.close()'''


    coordinates = []

    image = cv2.imread(path)
    file = open(annpath, 'a+')
    file.write(path + ' ')
    file.close()
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # press 'r' to reset the window
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    file = open(annpath, 'a+')
    for i in coordinates:
        print(i + ':')
        temp = input()
        if temp=='x':
            continue
        else:
            file.write(i + ',' + temp + ' ')
    file.write('\n')
    file.close()
    if len(ref_point) == 10:
        crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:
                                                          ref_point[1][0]]
        cv2.imshow("crop_img", crop_img)
        cv2.waitKey(0)

    # close all open windows
    cv2.destroyAllWindows()

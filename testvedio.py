import numpy as np
import cv2
import time

# cap = cv2.VideoCapture(r'C:\Users\Mr.Five_SSD\Desktop\Material\One_Bike_Cross.mp4')
cap = cv2.VideoCapture(r'C:\Users\Mr.Five_SSD\Desktop\Material\Many.mp4')
body_cascade = cv2.CascadeClassifier(
    r'C:\Users\Mr.Five_SSD\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_fullbody.xml')
# cap = cv2.VideoCapture(0)

# fgbg = cv2.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg = cv2.BackgroundSubtractor()

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200

params.minDistBetweenBlobs = 30

# 检测白色
params.filterByColor = True
params.blobColor = 255

# Filter by Area. 面积
params.filterByArea = True
params.minArea = 100
params.maxArea = 800

# Filter by Circularity 圆度
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity 凸性
params.filterByConvexity = False
params.minConvexity = 0.3

# Filter by Inertia 惯性
params.filterByInertia = False
params.minInertiaRatio = 0.01

while (cap.isOpened()):
    # get a frame
    ret, frame = cap.read()

    if ret == True:
        keycode = cv2.waitKey(1) & 0xff
        if keycode != 0xff:
            print(keycode, '\n', chr(keycode))
            break

        # time.sleep(0.1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        bodys = body_cascade.detectMultiScale(frame, 1.1, 5)
        print(bodys)
        for (x, y, w, h) in bodys:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        # cv2.GaussianBlur(frame, (81, 81), 0)
        cv2.imshow('frame0', frame)

        """
        fgmask = fgbg.apply(frame)
        cv2.imshow('frame1', fgmask)

        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        cv2.imshow('frame2', fgmask)

        fgmask = cv2.dilate(fgmask, kernel, iterations=2)
        # fgmask = cv2.dilate(fgmask, kernel, iterations=1)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(fgmask)
        # print(keypoints)

        fgmask = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('frame3', fgmask)
"""
    else:
        break

cap.release()
cv2.destroyAllWindows()

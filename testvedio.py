# -*- coding: utf-8 -*-：

import numpy as np
import cv2
import time
import subprocess

# cap = cv2.VideoCapture(r'C:\Users\Mr.Five_SSD\Desktop\Material\One_Bike_Cross.mp4')
# cap = cv2.VideoCapture(r'C:\Users\Mr.Five_SSD\Desktop\Material\Many.mp4')
cap = cv2.VideoCapture(r'C:\Users\Mr.Five_SSD\Desktop\Material\Three_people_cross.mp4')

body_cascade = cv2.CascadeClassifier(
    r'C:\Users\Mr.Five_SSD\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_fullbody.xml')
# cap = cv2.VideoCapture(0)

# fgbg = cv2.createBackgroundSubtractorMOG()
kernel3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel6x6 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
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
params.maxArea = 2500

# Filter by Circularity 圆度
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity 凸性
params.filterByConvexity = False
params.minConvexity = 0.3

# Filter by Inertia 惯性
params.filterByInertia = False
params.minInertiaRatio = 0.01

time_last = -1.0  # 避免刚运行程序时时间差为0，
frame_count = 0

while (cap.isOpened()):
    # get a frame
    ret, frame = cap.read()

    if ret == True:
        keycode = cv2.waitKey(1) & 0xff
        if keycode != 0xff:
            print(keycode, '\n', chr(keycode))
            break

        # time.sleep(0.1)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # bodys = body_cascade.detectMultiScale(frame, 1.1, 5)
        # print(bodys)
        # for (x, y, w, h) in bodys:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        # cv2.GaussianBlur(frame, (81, 81), 0)

        fgmask = fgbg.apply(gray)
        cv2.imshow('Foreground', fgmask)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel3x3)
        cv2.imshow('Opening', fgmask)

        ret, binary = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)  # 去掉阴影
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel6x6, iterations=2)

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary)

        # _, contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[4]
        frame_RGB = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(frame_RGB, contours, -1, (0, 0, 255), 1)
        # print(contours)

        frame_RGB = cv2.drawKeypoints(frame_RGB, keypoints, np.array([]), (0, 255, 0),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.minEnclosingTriangle()
        for index in range(np.size(keypoints)):
            x = int(keypoints[index].pt[0])
            y = int(keypoints[index].pt[1])
            w = h = int(keypoints[index].size)
            cv2.rectangle(frame, (x - int(0.6 * w), y - int(1.2 * h)), (x + int(0.6 * w), y + int(1.2 * h)),
                          (0, 255, 0), 1)
        # tu = keypoints

        str_t = time.strftime('%Y-%m-%d %H:%M:%S')
        time_now = time.clock()
        time_delta = time_now - time_last
        time_last = time_now
        fps = 1.0 / time_delta
        fps_t = str('fps: %.3f' % fps)
        det_t = str('calculate time:%.3f s' % time_delta)

        frame_count += 1  # 帧数
        fra_t = str('frame count:%d' % frame_count)


        cv2.putText(frame, fra_t, (450, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, fps_t, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, det_t, (0, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, str_t, (450, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Original', frame)
        cv2.imshow('Closing And Blob', frame_RGB)

        if frame_count == 333:
            subprocess.call("pause", shell=True)

    else:
        break

cap.release()
cv2.destroyAllWindows()

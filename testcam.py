# -*- coding: utf-8 -*-：

import cv2 as cv
import cv2
import time
# from cv2 import cv

import numpy as np

print("hello python!")

cap = cv2.VideoCapture(0)
# ret = cap.set(3, 160)
# ret = cap.set(4, 90)

face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Mr.Five_SSD\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    r'C:\Users\Mr.Five_SSD\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier(
    r'C:\Users\Mr.Five_SSD\AppData\Local\Programs\Python\Python36\Lib\site-packages\cv2\data\haarcascade_smile.xml')
time_last = 0.0

while (1):
    # get a frame
    ret, frame = cap.read()
    if ret == False:
        print('get frame false')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)

        roi_gray = gray[y:y + int(h / 2), x:x + w]  # 限制眼睛在脸部上半部分
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # cv2.circle(frame, (x + ex + int(0.5 * ew), y + ey + int(0.5 * eh)), int(0.5 * (ew + eh)), (0, 0, 255), 3)
            cv2.ellipse(frame, (x + ex + int(0.5 * ew), y + ey + int(0.5 * eh)), (int(0.5 * ew), int(0.35 * eh)), 0, 0,
                        360, (0, 0, 255), 3)

        roi_gray_m = gray[y + int(h / 2):y + h, x:x + w]  # 限制 部分
        smile = smile_cascade.detectMultiScale(roi_gray_m, 1.5, 30)
        for (mx, my, mw, mh) in smile:
            cv2.ellipse(frame, (x + mx + int(0.5 * mw), y + int(h / 2) + my + int(0.5 * mh)),
                        (int(0.5 * mw), int(0.35 * mh)), 0, 0,
                        360, (255, 0, 255), 3)

    frame = cv2.flip(frame, 1)  # flip the frame
    str_t = time.strftime('%Y-%m-%d %H:%M:%S')

    time_now = time.clock()
    time_delta = time_now - time_last
    time_last = time_now
    fps = 1.0 / time_delta
    fps_t = str('fps: %.3f' % fps)
    det_t = str('calculate time:%.3f s' % time_delta)

    cv2.putText(frame, fps_t, (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
    cv2.putText(frame, det_t, (0, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)
    cv2.putText(frame, str_t, (0, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
    cv2.imshow("capture", frame)  # show a frame

    keycode = cv2.waitKey(1) & 0xff

    if keycode != 0xff:
        print(keycode, '\n', chr(keycode))
        break

cap.release()
cv2.destroyAllWindows()

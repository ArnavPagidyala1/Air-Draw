import cv2
import numpy as np
from collections import deque
import imutils

# Detection range of color
red_lbound = np.array([161, 155, 84])
red_ubound = np.array([179, 255, 255])

lbound = red_lbound
ubound = red_ubound

# BGR format

color_list = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 255)]

palette = [(255, 0, 0), (0, 0, 255)]

index = 0

bluetrace = [deque(maxlen=1500)]
redtrace = [deque(maxlen=1500)]

blueindex = 0
redindex = 0

webcam = cv2.VideoCapture(0)

while True:
    (rec, frame) = webcam.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=1000)
    feed = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(feed, lbound, ubound)

    (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    t = 2

    frame = cv2.rectangle(frame, (125, 60), (275, 120), (90, 0, 100), -1)
    cv2.putText(frame, "CLEAR", (170, 95), font, font_scale, color_list[4], t, cv2.LINE_AA)

    frame = cv2.rectangle(frame, (425, 60), (575, 120), palette[0], -1)
    cv2.putText(frame, "BLUE", (480, 95), font, font_scale, color_list[4], t, cv2.LINE_AA)

    frame = cv2.rectangle(frame, (725, 60), (875, 120), palette[1], -1)
    cv2.putText(frame, "RED", (795, 95), font, font_scale, color_list[4], t, cv2.LINE_AA)

    if len(contours) > 0:
        cont = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cont)

        cv2.circle(frame, (int(x), int(y)), int(radius), color_list[2], 2)

        M = cv2.moments(cont)

        if M['m00'] != 0:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        else:
            center = (int(M['m10']), int(M['m01']))

        if center[1] <= 120:
            if 125 <= center[0] <= 275:
                bluetrace = [deque(maxlen=1500)]
                redtrace = [deque(maxlen=1500)]

                blueindex = 0
                redindex = 0

            elif 425 <= center[0] <= 575:
                index = 0

            elif 725 <= center[0] <= 875:
                index = 1
        else:
            if index == 0:
                bluetrace[blueindex].appendleft(center)
            elif index == 1:
                redtrace[redindex].appendleft(center)
    else:
        bluetrace.append(deque(maxlen=1500))
        blueindex += 1
        redtrace.append(deque(maxlen=1500))
        redindex += 1
    traced = [bluetrace, redtrace]

    for p in range(len(traced)):
        for m in range(len(traced[p])):
            for n in range(1, len(traced[p][m])):
                if traced[p][m][n] is None:
                    continue
                    
                cv2.line(frame, traced[p][m][n-1], traced[p][m][n], palette[p], 2)

    cv2.imshow("Air Draw", frame)

    if cv2.waitKey(1) & 0xFF == ord("w"):
        break
    
webcam.release()
cv2.destroyAllWindows()
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('Resources/people.mp4')

model = YOLO('WeightS/yolov8n.pt')
mask = cv2.imread('Resources/mask-1.png')

#tracker
tracker = Sort()

countUp = []
countDown= []
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]


while True:
    success, img = cap.read()
    resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imRegion = cv2.bitwise_and(resized_mask, img)
    detections = np.empty((0,5))
    results = model(imRegion,stream =True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1

            #lines
            cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
            cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

            #confidence
            conf = math.ceil(box.conf[0]*100)/100


            #class
            cls = box.cls[0]


            if conf > 0.4 and cls == 0.0:
                detections = np.vstack((detections,np.array([x1,y1,x2,y2,conf])))



    sort_out = tracker.update(detections)


    for ar in sort_out:
        x1, y1, x2, y2, Id = ar
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cvzone.cornerRect(img, (x1, y1, w, h), 10, rt=0, colorR=(255, 0, 255), colorC=(0, 255, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=3, thickness=5, offset=10)

        if (limitsUp[0] < cx < limitsUp[2]) and (limitsUp[1]-15 < cy < limitsUp[3]+15):
            if countUp.count(Id) == 0:
                countUp.append(Id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if (limitsDown[0] < cx < limitsDown[2]) and (limitsDown[1]-15 < cy < limitsDown[3]+15):
            if countDown.count(Id) == 0:
                countUp.append(Id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img, f"{len(countUp)}", (50,50), scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0) )

        cvzone.putTextRect(img, f"{len(countDown)}", (50, 50), scale=3, thickness=3, colorT=(255, 255, 255),
                           colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                           offset=10, border=None, colorB=(0, 255, 0))



    cv2.imshow("Image", img)

    cv2.waitKey(1)



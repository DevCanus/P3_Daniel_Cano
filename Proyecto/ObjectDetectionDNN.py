import fractions
from statistics import mode
import cv2 as cv
from cv2 import mean
import numpy as np

with open('./MSCOCO/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0,255,size=(len(class_names), 3))

model = cv.dnn.readNet(model='./MSCOCO/frozen_inference_graph.pb', config='./MSCOCO/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework='TensorFlow')

cap = cv.VideoCapture('./Videos de Entrada/1.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv.VideoWriter('./Videos de Salida/1.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

contador = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not(ret):
        contador = contador + 1

    if ret:
        image = frame
        image_height, image_width, _ = image.shape

        blob = cv.dnn.blobFromImage(image = image, size = (300,300), mean = (104,117,123), swapRB = True)

        model.setInput(blob)
        output = model.forward()

        for detection in output [0,0,:,:]:
            confidence = detection[2]

            if confidence > .4:
                class_id = detection[1]

                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]

                box_x = detection[3] * image_width
                box_y = detection[4] * image_height

                box_width = detection[5] * image_width
                box_height = detection[6] * image_height

                cv.rectangle(image, (int(box_x), int(box_y)),(int(box_width), int(box_height)), color, thickness = 2)
                out.write(image)
                cv.putText(image, class_name, (int(box_x), int(box_y - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv.imshow('image', image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        elif cv.waitKey(1) & 0xFF == ord('1'):
            contador = 0
            cap = cv.VideoCapture('./Videos de Entrada/1.mp4')
            out = cv.VideoWriter('./Videos de Salida/1.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        elif cv.waitKey(1) & 0xFF == ord('2'):
            contador = 1
            cap = cv.VideoCapture('./Videos de Entrada/2.mp4')
            out = cv.VideoWriter('./Videos de Salida/2.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        elif cv.waitKey(1) & 0xFF == ord('3'):
            contador = 2
            cap = cv.VideoCapture('./Videos de Entrada/3.mp4')
            out = cv.VideoWriter('./Videos de Salida/3.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    elif not(ret) and contador == 1:
        cap = cv.VideoCapture('./Videos de Entrada/2.mp4')
        out = cv.VideoWriter('./Videos de Salida/2.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    elif not(ret) and contador == 2:
        cap = cv.VideoCapture('./Videos de Entrada/3.mp4')
        out = cv.VideoWriter('./Videos de Salida/3.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    else :
        break
    
cap.release()
cv.destroyAllWindows()
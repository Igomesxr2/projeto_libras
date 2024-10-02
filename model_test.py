from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2
import numpy as np
model=YOLO("C:/Users/pires/Downloads/best.pt")

def check_imshow():
    try:
        cv2.imshow('test',np.zeros((1,1,3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WANING: Enviroment does not support cv2.imshow() or PIL Image.')
        return False
    

model.predict(source='0',show=True, conf=0.3)
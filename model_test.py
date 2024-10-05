from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2
import numpy as np
model=YOLO("C:/Users/pires/Downloads/best.pt")

from flask import Flask, render_template
app = Flask(__name__)

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

#INTERFACE

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detectar')
def detectar():
    model.predict(source='0', show=True, conf=0.3)
    return "Detecção em andamento..."

if __name__ == '__main__':
    app.run(debug=True)
from ultralytics import YOLO
import cv2
import time
import webview
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

webview.create_window('SINALIZE', app)

model = YOLO("C:/Users/Admin/Desktop/projeto_libras/best.pt")

palavras = []
ultima_palavra = ""
inicio_tempo_palavra = None
min_tempo_reconhecimento = 1.5

def generate_frames():
    global palavras, ultima_palavra, inicio_tempo_palavra
    
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.4)

        for result in results:
            boxes = result.boxes
            ids_class = result.boxes.cls

            for box, id_class in zip(boxes.xyxy, ids_class):
                #x1, y1, x2, y2 = map(int, box)
                name_class = model.names[int(id_class)]

                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, name_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Verificar se a palavra detectada é a mesma da última detecção
                if name_class == ultima_palavra:
                    # Se for a mesma palavra, verificar se o tempo de detecção é maior que 1.5 segundos
                    if inicio_tempo_palavra and time.time() - inicio_tempo_palavra >= min_tempo_reconhecimento:
                        palavras.append(name_class)
                        inicio_tempo_palavra = time.time()
                else:
                    ultima_palavra = name_class
                    inicio_tempo_palavra = time.time()

        #palavras_str = ", ".join(palavras)
        #cv2.putText(frame, palavras_str, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detectar')
def detectar():
    return render_template('detectar.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/palavras')
def palavras_detectadas():
    return jsonify(palavras)

if __name__ == '__main__':
    webview.start()

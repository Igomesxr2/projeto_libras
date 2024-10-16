from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("C:/Users/pires/OneDrive/Área de Trabalho/projeto_libras/best.pt")

palavras = []
ultima_palavra = ""
inicio_tempo_palavra = None  # Armazena o tempo de início da detecção de uma nova palavra
min_tempo_reconhecimento = 1.5  # Tempo mínimo de reconhecimento da palavra (1.5 segundos)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar a câmera.")
            break

        results = model.predict(source=frame, conf=0.4)

        for result in results:
            boxes = result.boxes
            ids_class = result.boxes.cls

            for box, id_class in zip(boxes.xyxy, ids_class):
                x1, y1, x2, y2 = map(int, box)  # Coordenadas da caixa
                name_class = model.names[int(id_class)]  # Nome da classe

                # Desenhar a caixa
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Adicionar o nome da classe acima da caixa
                cv2.putText(frame, name_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Verificar se a palavra detectada é a mesma da última detecção
                if name_class == ultima_palavra:
                    # Se for a mesma palavra, verificar se o tempo de detecção é maior que 1.5 segundos
                    if inicio_tempo_palavra and time.time() - inicio_tempo_palavra >= min_tempo_reconhecimento:
                        # Se passou mais de 1.5 segundos, adicionar a palavra à lista e reiniciar o temporizador
                        palavras.append(name_class)
                else:
                    # Se for uma nova palavra, reiniciar o temporizador
                    ultima_palavra = name_class
                    inicio_tempo_palavra = time.time()

        # Adicione o texto com as palavras detectadas no frame
        palavras_str = ", ".join(palavras)  # Juntar as palavras em uma única string
        cv2.putText(frame, palavras_str, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)

        # Mostrar o frame da câmera
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

import os
import cv2
import time

# Define o tamanho das imagens
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta com sucesso
if not cap.isOpened():
    print("Erro ao abrir a câmera")
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) 

    # Cria a pasta images para armazenar os frames
    folder = "images"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Intervalo em segundos
    interval = 1
    last_capture_time = time.time()  # Marca o tempo da última captura

    while True:
        ret, frame = cap.read()
        if ret:
            # Exibe o frame capturado em uma janela
            cv2.imshow('Câmera', frame)

            # Verifica se o intervalo de captura passou
            if time.time() - last_capture_time >= interval:
                # Gera o nome do arquivo da imagem
                filename = os.path.join(folder, f"{time.time()}.jpg")
                # Salva a imagem
                cv2.imwrite(filename, frame)
                last_capture_time = time.time()  # Atualiza o tempo da última captura

        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a câmera e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()

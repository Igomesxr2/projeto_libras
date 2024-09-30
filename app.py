import os
import cv2
import time

# Define o tamanho das imagens
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) 

# Cria a pasta images para armazenar o frames
folder = "images"
if not os.path.exists(folder):
    os.makedirs(folder)
    
# Intervalo em segundos 
interval = 5

while True:
    ret, frame = cap.read()
    if ret:
        # Gera o nome do arquivo da imagem
        filename = os.path.join(folder, f"{time.time()}.jpg")
        # Salva a imagem 
        cv2.imwrite(filename,frame)
    
    # Tempo de espera para capturar a próxima imagem
    time.sleep(interval)
    
    # Saí do loop se a tecla q for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()



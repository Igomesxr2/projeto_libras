from roboflow import Roboflow
rf = Roboflow(api_key="crUWM2RAQpP6WUSwQw1W")
project = rf.workspace("gomes-project").project("projeto-libras")
version = project.version(7)
dataset = version.download("yolov8")

#yolo task=detect mode=train model=yolov8s.pt data="C:/Users/pires/OneDrive/Área de Trabalho/projeto_libras/projeto-libras-7/data.yaml" epochs=10 imgsz=640

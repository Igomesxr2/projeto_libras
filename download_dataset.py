import roboflow

# Substitua "YOUR_API_KEY" pela sua chave de API do Roboflow
rf = roboflow.Roboflow(api_key="crUWM2RAQpP6WUSwQw1W")
# Substitua "your-project-name" pelo nome do seu projeto
project = rf.workspace().project("projeto-libras")
# Substitua "1" pela versão do seu dataset
dataset = project.version(1).download("yolov8")

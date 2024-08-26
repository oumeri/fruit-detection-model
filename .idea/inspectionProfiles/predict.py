from ultralytics import YOLO

model = YOLO("C:/Users/hp/AppData/Roaming/JetBrains/PyCharm2024.2/extensions/runs/classify/train4/weights/last.pt") # load a custom model


results = model("C:/Users/hp/Desktop/apple")

print(results)

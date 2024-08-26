from ultralytics import YOLO



model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

model.train(data="C:/Users/hp/Desktop/fruit-detection", epochs=6, imgsz=64)


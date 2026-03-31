from ultralytics import YOLO

# Resume from last trained weights
model = YOLO("runs/detect/train2/weights/last.pt")

model.train(
resume=True)
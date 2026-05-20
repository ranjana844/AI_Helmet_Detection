from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(source=0, show=True)

print(model.names)
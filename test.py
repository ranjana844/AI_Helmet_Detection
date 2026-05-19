from ultralytics import YOLO

model = YOLO(r"C:\Users\ranja\runs\detect\train-2\weights\best.pt")

results = model.predict(source=0, show=True)

print(model.names)
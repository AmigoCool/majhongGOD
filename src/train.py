from ultralytics import YOLO

# 加載預訓練模型
model = YOLO("yolov8x.pt")

# 開始訓練
model.train(
    data="data.yml",
    epochs=200,
    imgsz=512,
    workers=0,
    batch=16,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
)  # 強制使用 GPU device='cuda'

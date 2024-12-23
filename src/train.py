from ultralytics import YOLO

# 加載預訓練模型
model = YOLO('yolov8x.pt')

# 開始訓練
model.train(data='data.yml', epochs=50, imgsz=768, device='cuda', workers=0)  # 強制使用 GPU

model.tune_anchors(data='data.yml')

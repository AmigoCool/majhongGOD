import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO

# 加載訓練好的 YOLO 模型
model = YOLO('runs/detect/train6/weights/best.pt')

# 定義類別名稱映射表（與模型的類別順序一致）
class_names = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1z", "2z", "3z", "4z", "5z", "6z", "7z"
]

# 使用 OpenCV 讓用戶選擇螢幕區域
def select_screen_region():
    # 捕獲整個螢幕的畫面
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])  # 捕捉第二個螢幕，若只有一個螢幕，請使用 sct.monitors[0]

    # 把螢幕畫面轉換為 numpy 陣列
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 使用 OpenCV 的 selectROI 函數讓用戶選擇要偵測的區域
    roi = cv2.selectROI("Select the area to track", img, fromCenter=False, showCrosshair=True)

    # 關閉視窗
    cv2.destroyAllWindows()
    return roi

# 設定要捕獲的螢幕區域
roi = select_screen_region()

# 開始循環捕獲螢幕畫面並進行檢測
with mss.mss() as sct:
    while True:
        # 捕獲指定區域的螢幕畫面
        screenshot = sct.grab({
            'top': roi[1],
            'left': roi[0],
            'width': roi[2],
            'height': roi[3]
        })

        # 把螢幕畫面轉換為 numpy 陣列
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 使用 YOLO 模型進行檢測
        results = model(frame)

        # 確保結果是列表中的第一個元素，並檢查是否有檢測結果
        if len(results) > 0:
            result = results[0]  # 提取第一個檢測結果
            boxes = result.boxes.xywh  # 邊界框資訊
            confidences = result.boxes.conf  # 置信度
            classes = result.boxes.cls  # 類別索引

            if len(boxes) > 0:
                print("Detected objects:")
                for i, box in enumerate(boxes):
                    class_id = int(classes[i])  # 獲取類別索引
                    class_name = class_names[class_id]  # 映射到麻將牌名稱
                    confidence = confidences[i]  # 置信度

                    print(f"牌: {class_name}, 信心值: {confidence:.2f}, 邊界框: {box.cpu().numpy()}")

                # 使用 `plot()` 顯示檢測後的圖像
                result.plot()  # 繪製檢測框並顯示結果圖像

            else:
                print("No detections found.")

        else:
            print("No detections found.")

        # 每秒進行一次偵測
        time.sleep(1)

        # 按 "q" 鍵退出
        if 0xFF == ord('q'):
            break

# 釋放視窗
cv2.destroyAllWindows()

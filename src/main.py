import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO

# 加載訓練好的 YOLO 模型
model = YOLO('runs/detect/other/mahjong.pt')

# 定義類別名稱映射表（與模型的類別順序一致）
class_names = ['1m', '1p', '1s', '1z', '2m', '2p', '2s', '2z', '3m', '3p', '3s', '3z', '4m', '4p', '4s', '4z', '5m', '5p', '5s', '5z', '6m', '6p', '6s', '6z', '7m', '7p', '7s', '7z', '8m', '8p', '8s', '9m', '9p', '9s']
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

        # 確保結果是列表中的第一個元素
        if len(results) > 0:
            result = results[0]  # 提取第一個檢測結果
            boxes = result.boxes.xyxy.cpu().numpy()  # 邊界框 (左上角 x, y, 右下角 x, y)
            confidences = result.boxes.conf.cpu().numpy()  # 置信度
            classes = result.boxes.cls.cpu().numpy()  # 類別索引

            # 在圖像上繪製檢測框和標籤
            for i, box in enumerate(boxes):
                class_id = int(classes[i])  # 獲取類別索引
                class_name = class_names[class_id]  # 映射到麻將牌名稱
                confidence = confidences[i]  # 置信度

                # 繪製邊界框
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 繪製標籤
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 顯示處理後的畫面
        cv2.imshow("YOLO Detection", frame)

        # 每秒進行一次偵測
        time.sleep(1)

        # 按 "q" 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 釋放視窗
cv2.destroyAllWindows()

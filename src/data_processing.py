import cv2
import os


def preprocess_images(input_dir, output_dir, img_size=(128, 128)):
    """
    將input_dir中的圖片調整大小，並保存到output_dir中
    :param input_dir: 原始圖片的文件夾
    :param output_dir: 預處理後圖片的保存文件夾
    :param img_size: 調整後圖片的大小（默認128x128）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is not None:
            # 調整大小
            img_resized = cv2.resize(img, img_size)
            output_path = os.path.join(output_dir, img_file)

            # 保存處理後的圖片
            cv2.imwrite(output_path, img_resized)


# 調用函數對圖片進行預處理
preprocess_images("data/raw/", "data/processed/", img_size=(128, 128))

import cv2
import os
import numpy as np

def preprocess_images(input_dir, output_dir, img_size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is not None:
            # 調整大小
            img_resized = cv2.resize(img, img_size)
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, img_resized)

def split_data(input_dir, train_dir, test_dir, train_size=0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = os.listdir(input_dir)
    np.random.shuffle(images)

    split_index = int(len(images) * train_size)
    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        img_path = os.path.join("data/raw", img)
        cv2.imwrite(os.path.join("data/processed", img), cv2.imread(img_path))

    for img in test_images:
        img_path = os.path.join("data/raw", img)
        cv2.imwrite(os.path.join("data/processed", img), cv2.imread(img_path))

# 調用預處理函數
preprocess_images('data/raw/', 'data/processed/', img_size=(128, 128))
split_data('data/processed/', 'data/train/', 'data/test/')

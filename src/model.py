import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(128, 128, 3)):
    model = models.Sequential()

    # 第一層卷積層
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # 第二層卷積層
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())  # 展開層

    model.add(layers.Dense(64, activation="relu"))  # 全連接層

    model.add(layers.Dense(34, activation="softmax"))  # 輸出層 有34種麻將牌

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model

# 定義資料增強
datagen = ImageDataGenerator(rescale=1./255)

# 訓練集和測試集的生成器
train_generator = datagen.flow_from_directory(
    'data/train/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

test_generator = datagen.flow_from_directory(
    'data/test/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse'
)

# 建立模型
model = build_model()

# 訓練模型
model.fit(train_generator, epochs=10, validation_data=test_generator)

# 儲存模型
model.save('mahjong_model.h5')

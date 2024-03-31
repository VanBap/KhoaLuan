import tensorflow as tf
import pathlib
from tensorflow import keras
#import keras

# Load model
model=tf.keras.models.load_model('C:/saved_model/ResNet50_Weather_epoch20.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert model và lưu vào file .tflite
tflite_model = converter.convert()
with open('C:/CODE_PYCHARM/KhoaLuan/saved_model/ResNet50_Weather_epoch20_optimizing_date14.tflite', 'wb') as f:
    f.write(tflite_model)


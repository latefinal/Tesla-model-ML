import tensorflow as tf
import os

# 檢查和創建輸出目錄
OUTPUT_PATH = "output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Step 1: 加載 Keras 模型
model_path = os.path.join(OUTPUT_PATH, 'tesla_model_classifier.h5')
model = tf.keras.models.load_model(model_path)

# Step 2: 將模型導出為 SavedModel 格式
saved_model_path = os.path.join(OUTPUT_PATH, 'tesla_model_saved_model')
model.export(saved_model_path)  # 使用 export 方法來保存為 SavedModel 格式

# Step 3: 使用 TFLite 轉換器轉換 SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# Step 4: 優化選項，禁用部分不支持的優化
converter.experimental_new_converter = False  # 禁用新轉換器以避免不支持的優化
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 優化以減小模型大小
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,  # 選擇內建的 TensorFlow Lite 操作
                                       tf.lite.OpsSet.SELECT_TF_OPS]    # 支持 TensorFlow 選擇操作集

# Step 5: 進行轉換並保存
try:
    tflite_model = converter.convert()
    # 保存轉換後的 .tflite 文件
    tflite_model_path = os.path.join(OUTPUT_PATH, 'tesla_model_classifier.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted and saved as {tflite_model_path}")
except Exception as e:
    print("Error during conversion:", e)
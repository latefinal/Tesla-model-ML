import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Set model and test image paths
MODEL_PATH = "../output/tesla_model_classifier.h5"  # Path to the saved model
TEST_IMAGE_PATH = "images/"  # Path to folder with test images

# Check if model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Define class labels to match the training order
class_indices = {'Model_3': 0, 'Model_S': 1, 'Model_X': 2, 'Model_Y': 3}
class_labels = list(class_indices.keys())  # Get a list of class names

# Preprocessing function for input images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0             # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)           # Add batch dimension
    return img_array

# Function to predict the class of a single image
def predict_image(img_path):
    img_array = preprocess_image(img_path)                  # Preprocess the image
    predictions = model.predict(img_array)                  # Get model predictions
    predicted_class = np.argmax(predictions, axis=1)[0]     # Get the index of the highest prediction score
    class_name = class_labels[predicted_class]              # Map index to class name
    confidence = predictions[0][predicted_class]            # Confidence of the prediction
    if confidence < 0.5:                                    # Optional: Warn if confidence is low
        print(f"Low confidence ({confidence:.2f}) in prediction for image: {img_path}")
    print(f"Image: {img_path} | Predicted: {class_name} | Confidence: {confidence:.2f}")
    return class_name, confidence

# Function to predict classes for all images in a specified folder
def predict_images_in_folder(folder_path):
    for img_file in os.listdir(folder_path):                # Loop through files in the folder
        img_path = os.path.join(folder_path, img_file)      # Get full path to the image file
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            predict_image(img_path)                         # Predict and print result for each image

# Test prediction on all images in the folder
predict_images_in_folder(TEST_IMAGE_PATH)
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Step 1: Load Pre-trained Vision Transformer (ViT) Model from TensorFlow Hub
vit_url = "https://tfhub.dev/google/vit_base_patch16_224/1"  # ViT pre-trained model
vit_model = hub.KerasLayer(vit_url)

# Step 2: Create a custom model using the ViT layer
model = tf.keras.Sequential([
    vit_model,  # Pre-trained ViT as feature extractor
    tf.keras.layers.Dense(512, activation='relu'),  # Dense layer for more abstraction
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (e.g., healthy or diseased)
])

# Step 3: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Preprocess the image for the ViT model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match ViT input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Step 5: Load and preprocess the image
img_path = 'path_to_your_crop_image.jpg'  # Provide the image path
img_array = preprocess_image(img_path)

# Step 6: Make predictions
def predict_disease(model, img_array):
    predictions = model.predict(img_array)
    return predictions

# Get the prediction (0 = healthy, 1 = diseased)
prediction = predict_disease(model, img_array)
print("Prediction (1 = Diseased, 0 = Healthy):", prediction[0][0])

# Step 7: Convert the model to TensorFlow Lite format for mobile deployment
def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('crop_disease_model.tflite', 'wb') as f:
        f.write(tflite_model)

# Convert the model to TensorFlow Lite
convert_to_tflite(model)
print("Model has been converted to TensorFlow Lite.")


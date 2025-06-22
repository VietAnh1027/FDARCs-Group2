import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

class FruitRipenessPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_indices = self.load_class_indices()
        
    def load_class_indices(self):
        with open('./class_indices.txt', 'r') as f:
            class_indices = eval(f.read())
        return {v: k for k, v in class_indices.items()}  # Reverse mapping
    
    def preprocess_image(self, img_cv2, target_size=(128, 128)):
        # img = Image.open(image_path)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    
    def predict(self, image_path):
        processed_img = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_img)[0][0]
        
        class_idx = 1 if prediction > 0.5 else 0
        class_name = self.class_indices[class_idx]
        confidence = prediction if class_idx == 1 else 1 - prediction
        
        return class_name, confidence
        # {
        #     'class': class_name,
        #     'confidence': float(confidence),
        #     'is_ripe': bool(class_idx)
        # }

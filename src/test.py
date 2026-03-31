import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. Load model
model = load_model('outputs/model.keras')

# 2. get class names
data_dir = 'data_grouped'
class_names = sorted(os.listdir(data_dir))

print("Classes:", class_names)

# 3. Prediction function
def predict_image(img_path):
    try:
        # Check file exists
        if not os.path.exists(img_path):
            print("Error: File not found")
            return

        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]

        
        if confidence < 0.5:
            print("Prediction uncertain (Unknown image)")
        else:
            print(f"Prediction: {predicted_class}")
            print(f"Confidence: {confidence:.2f}")

    except Exception as e:
        print("Error processing image:", str(e))


# 4. Run script
if __name__ == "__main__":
    img_path = input("Enter image path: ")
    predict_image(img_path)
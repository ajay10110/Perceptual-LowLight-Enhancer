import os
import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def enhance_image(image_path, model_path, output_path):
    # 1. Load the model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please download 'best_model_fresh_500.keras' from Google Drive and place it in the correct folder.")
        return

    # Load without compiling since we only need it for prediction, not training
    model = load_model(model_path, compile=False)

    # 2. Load and preprocess the image
    print(f"Loading image from {image_path}...")
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found at {image_path}")
        return

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # U-Nets are fully convolutional, meaning they can process any image size!
    # The only rule is that the width and height must be divisible by 16 
    # (because the network shrinks the image in half 4 times: 2^4 = 16).
    original_size = (img.shape[1], img.shape[0])
    
    # Calculate the nearest dimensions that are divisible by 16
    h, w = img.shape[:2]
    new_h = (h // 16) * 16
    new_w = (w // 16) * 16
    
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Normalize to [0, 1] exactly like we did in training
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension: (1, 256, 256, 3)
    img_input = np.expand_dims(img_normalized, axis=0)

    # 3. Predict (Enhance)
    print("Enhancing image...")
    prediction = model.predict(img_input)[0] # Remove batch dimension

    # 4. Post-process and save
    # Denormalize back to [0, 255]
    prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)
    
    # Resize back to original image size
    prediction_restored_size = cv2.resize(prediction, original_size)
    
    # Convert back to BGR for OpenCV saving
    prediction_bgr = cv2.cvtColor(prediction_restored_size, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, prediction_bgr)
    print(f"✅ Enhanced image successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance a dark image using the trained U-Net model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the dark input image")
    parser.add_argument("--model", type=str, default="models/best_model_fresh_500.keras", help="Path to the .keras model")
    parser.add_argument("--output", type=str, default="enhanced_output.png", help="Path to save the enhanced image")
    
    args = parser.parse_args()
    
    enhance_image(args.image, args.model, args.output)

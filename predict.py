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
    
    original_size = (img.shape[1], img.shape[0])
    
    # The model was hardcoded to accept exactly 256x256 images.
    # To process high-definition images without squishing them (which causes pixelation),
    # we must slice the image into a grid of 256x256 patches, enhance each patch, and stitch them back together!
    patch_size = 256
    h, w, c = img.shape
    
    # Pad the image so it fits perfectly into a 256x256 grid
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    # Normalize to [0, 1]
    img_normalized = img_padded.astype(np.float32) / 255.0
    
    # Create an empty canvas for the enhanced image
    enhanced_padded = np.zeros_like(img_normalized)
    
    rows = img_padded.shape[0] // patch_size
    cols = img_padded.shape[1] // patch_size
    print(f"Enhancing high-res image by processing a {rows}x{cols} grid of patches...")
    
    for i in range(0, img_padded.shape[0], patch_size):
        for j in range(0, img_padded.shape[1], patch_size):
            patch = img_normalized[i:i+patch_size, j:j+patch_size]
            patch_input = np.expand_dims(patch, axis=0)
            
            # Enhance this specific patch
            pred_patch = model.predict(patch_input, verbose=0)[0]
            enhanced_padded[i:i+patch_size, j:j+patch_size] = pred_patch
            
    # Crop the padding off to return to the exact original size
    prediction = enhanced_padded[:h, :w, :]

    # 4. Post-process and save
    # Denormalize back to [0, 255]
    prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)
    
    # Convert back to BGR for OpenCV saving
    prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, prediction_bgr)
    print(f"✅ Enhanced image successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance a dark image using the trained U-Net model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the dark input image")
    parser.add_argument("--model", type=str, default="models/best_model_fresh_500.keras", help="Path to the .keras model")
    parser.add_argument("--output", type=str, default="enhanced_output.png", help="Path to save the enhanced image")
    
    args = parser.parse_args()
    
    enhance_image(args.image, args.model, args.output)

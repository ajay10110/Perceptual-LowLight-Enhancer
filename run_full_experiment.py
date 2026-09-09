import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from functools import partial
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_TRAIN_LOW = "data/train/low"
DATA_TRAIN_HIGH = "data/train/high"
DATA_EVAL_LOW = "data/eval/low"
DATA_EVAL_HIGH = "data/eval/high"
DATA_TEST_LOW = "data/test/low"
DATA_TEST_HIGH = "data/test/high"

SAVE_DIR = "models/training_runs"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- DATA VERIFICATION ---
def verify_dataset(low_dir, high_dir, split_name):
    print(f"\\n--- Verifying {split_name} Dataset ---")
    if not os.path.exists(low_dir) or not os.path.exists(high_dir):
        print(f"ERROR: {low_dir} or {high_dir} does not exist.")
        return False
        
    low_files = sorted(os.listdir(low_dir))
    high_files = sorted(os.listdir(high_dir))
    
    print(f"Low-light images found: {len(low_files)}")
    print(f"High-light images found: {len(high_files)}")
    
    missing_pairs = [f for f in low_files if not os.path.exists(os.path.join(high_dir, f))]
    if missing_pairs:
        print(f"WARNING: Found {len(missing_pairs)} low-light images without high-light pairs!")
        
    if len(low_files) == 0:
        print("ERROR: No images found.")
        return False
        
    # Check dimensions and normalization of the first pair
    sample_low = Image.open(os.path.join(low_dir, low_files[0])).convert('RGB')
    sample_high = Image.open(os.path.join(high_dir, low_files[0])).convert('RGB')
    
    print(f"Sample Dimensions: Low {sample_low.size}, High {sample_high.size}")
    arr_low = np.array(sample_low).astype(np.float32) / 255.0
    arr_high = np.array(sample_high).astype(np.float32) / 255.0
    print(f"Normalization Range: Low [{arr_low.min():.2f}, {arr_low.max():.2f}], High [{arr_high.min():.2f}, {arr_high.max():.2f}]")
    
    return True

# --- DATA LOADING ---
def load_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img = np.array(img).astype(np.float32) / 255.0
    return img

def load_dataset(low_dir, high_dir):
    low_imgs, high_imgs = [], []
    for img_name in sorted(os.listdir(low_dir)):
        low_path = os.path.join(low_dir, img_name)
        high_path = os.path.join(high_dir, img_name)
        if os.path.exists(high_path):
            low_imgs.append(load_image(low_path))
            high_imgs.append(load_image(high_path))
    return np.array(low_imgs, dtype=np.float32), np.array(high_imgs, dtype=np.float32)

# --- MODEL DEFINITION ---
def unet_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    u2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(b)
    concat2 = layers.concatenate([u2, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat2)
    u1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c3)
    concat1 = layers.concatenate([u1, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(c4)
    return Model(inputs, outputs)

# --- METRICS & LOSSES ---
def psnr_metric(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def color_alignment_loss(y_true, y_pred):
    mean_true = tf.reduce_mean(y_true, axis=[1,2])
    mean_pred = tf.reduce_mean(y_pred, axis=[1,2])
    std_true = tf.math.reduce_std(y_true, axis=[1,2])
    std_pred = tf.math.reduce_std(y_pred, axis=[1,2])
    return tf.reduce_mean(tf.square(mean_true - mean_pred)) + tf.reduce_mean(tf.square(std_true - std_pred))

def combined_loss_baseline(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    cal_loss = color_alignment_loss(y_true, y_pred)
    return mse_loss + 0.01 * cal_loss

# VGG Extractor Setup (Lazy init to save memory if not running perceptual)
_vgg_extractor = None
def get_vgg_extractor():
    global _vgg_extractor
    if _vgg_extractor is None:
        base_vgg = VGG19(include_top=False, weights='imagenet')
        base_vgg.trainable = False
        feat_outputs = [base_vgg.get_layer(n).output for n in ['block2_conv2', 'block3_conv4']]
        _vgg_extractor = Model(inputs=base_vgg.input, outputs=feat_outputs)
        _vgg_extractor.trainable = False
    return _vgg_extractor

@tf.function
def perceptual_loss(y_true, y_pred):
    vgg = get_vgg_extractor()
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    y_true_vgg = vgg_preprocess(y_true * 255.0)
    y_pred_vgg = vgg_preprocess(y_pred * 255.0)
    feats_true = vgg(y_true_vgg)
    feats_pred = vgg(y_pred_vgg)
    if not isinstance(feats_true, (list, tuple)):
        feats_true = [feats_true]; feats_pred = [feats_pred]
    loss = 0.0
    for ft, fp in zip(feats_true, feats_pred):
        loss += tf.reduce_mean(tf.square(ft - fp))
    return loss / float(len(feats_true))

@tf.function
def perceptual_loss_normalized(y_true, y_pred):
    vgg = get_vgg_extractor()
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    y_true_vgg = vgg_preprocess(y_true * 255.0)
    y_pred_vgg = vgg_preprocess(y_pred * 255.0)
    feats_true = vgg(y_true_vgg)
    feats_pred = vgg(y_pred_vgg)
    if not isinstance(feats_true, (list, tuple)):
        feats_true = [feats_true]; feats_pred = [feats_pred]
    total = 0.0
    for ft, fp in zip(feats_true, feats_pred):
        diff = ft - fp
        n = tf.cast(tf.size(diff), tf.float32)
        total += tf.reduce_sum(tf.square(diff)) / (n + 1e-12)
    return total / float(len(feats_true))

def combined_loss_perceptual(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + 0.01 * color_alignment_loss(y_true, y_pred) + 0.00001 * perceptual_loss(y_true, y_pred)

def combined_loss_lambda1e4(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + 0.01 * color_alignment_loss(y_true, y_pred) + 0.0001 * perceptual_loss(y_true, y_pred)

def combined_loss_norm1e3(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + 0.01 * color_alignment_loss(y_true, y_pred) + 1e-3 * perceptual_loss_normalized(y_true, y_pred)

def combined_loss_norm5e4(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + 0.01 * color_alignment_loss(y_true, y_pred) + 5e-4 * perceptual_loss_normalized(y_true, y_pred)


# --- HELPERS ---
def save_config(run_dir, config_dict):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

# --- RUNNERS ---
def run_baseline():
    if not verify_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH, "Training"): sys.exit(1)
    X, Y = load_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH)
    model = unet_model()
    model.compile(optimizer='adam', loss=combined_loss_baseline, metrics=['mse', psnr_metric, ssim_metric])
    
    config = {"stage": "baseline", "epochs": 100, "batch_size": 4, "optimizer": "adam (default 1e-3)", "loss": "MSE + 0.01*CAL"}
    save_config(SAVE_DIR, config)
    
    cb_log = CSVLogger(os.path.join(SAVE_DIR, "baseline_log.csv"), append=False)
    model.fit(X, Y, epochs=100, batch_size=4, callbacks=[cb_log], verbose=1)
    model.save(os.path.join(SAVE_DIR, "original_baseline_model.keras"))
    print("✅ Baseline run completed!")

def run_perceptual():
    if not verify_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH, "Training"): sys.exit(1)
    X, Y = load_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH)
    model = unet_model()
    model.compile(optimizer='adam', loss=combined_loss_perceptual, metrics=['mse', psnr_metric, ssim_metric])
    
    config = {"stage": "perceptual", "epochs_part1": 20, "epochs_part2": 100, "batch_size": 4, "loss": "MSE + 0.01*CAL + 1e-5*VGG"}
    save_config(SAVE_DIR, config)
    
    chkpt = os.path.join(SAVE_DIR, "best_model_perceptual.keras")
    cb_ckpt = ModelCheckpoint(chkpt, monitor='loss', save_best_only=True, mode='min', verbose=1)
    cb_log = CSVLogger(os.path.join(SAVE_DIR, "perceptual_log.csv"), append=True)
    
    print("--- Perceptual Stage 1 (20 Epochs) ---")
    model.fit(X, Y, epochs=20, batch_size=4, callbacks=[cb_ckpt, cb_log], verbose=1)
    
    print("--- Perceptual Stage 2 (Resume 100 Epochs) ---")
    model = load_model(chkpt, compile=False)
    model.compile(optimizer='adam', loss=combined_loss_perceptual, metrics=['mse', psnr_metric, ssim_metric])
    model.fit(X, Y, epochs=100, batch_size=4, callbacks=[cb_ckpt, cb_log], verbose=1)
    print("✅ Perceptual run completed!")

def run_experiments():
    if not verify_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH, "Training"): sys.exit(1)
    X, Y = load_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH)
    chkpt = os.path.join(SAVE_DIR, "best_model_perceptual.keras")
    if not os.path.exists(chkpt):
        print(f"ERROR: {chkpt} missing. Run --perceptual first.")
        sys.exit(1)
    
    experiments = [
        ("lambda1e4", combined_loss_lambda1e4, 5),
        ("norm1e3", combined_loss_norm1e3, 1),
        ("norm5e4", combined_loss_norm5e4, 5)
    ]
    
    for name, loss_fn, epochs in experiments:
        print(f"--- Running Experiment: {name} ({epochs} epochs) ---")
        model = load_model(chkpt, compile=False)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['mse', psnr_metric, ssim_metric])
        model.fit(X, Y, epochs=epochs, batch_size=4, verbose=1)
        model.save(os.path.join(SAVE_DIR, f"exp_{name}.keras"))
    print("✅ Experiments run completed!")

def run_long(epochs, val_split, name):
    if not verify_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH, "Training"): sys.exit(1)
    X, Y = load_dataset(DATA_TRAIN_LOW, DATA_TRAIN_HIGH)
    
    chkpt_path = os.path.join(SAVE_DIR, "best_model_perceptual.keras")
    if not os.path.exists(chkpt_path):
        print(f"ERROR: {chkpt_path} missing. Run --perceptual first.")
        sys.exit(1)
        
    model = load_model(chkpt_path, compile=False)
    # Reinitialize weights
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            new_weights = [init(w.shape, dtype=w.dtype) for w, init in zip(layer.get_weights(), [layer.kernel_initializer if hasattr(layer, 'kernel_initializer') else None for _ in layer.get_weights()])]
            if new_weights:
                layer.set_weights(new_weights)
                
    model.compile(optimizer='adam', loss=combined_loss_perceptual, metrics=['mse', psnr_metric, ssim_metric])
    
    run_dir = os.path.join(SAVE_DIR, f"fresh_{name}")
    os.makedirs(run_dir, exist_ok=True)
    out_keras = os.path.join(run_dir, f"best_model_fresh_{name}.keras")
    
    config = {"stage": f"long_{name}", "epochs": epochs, "val_split": val_split, "loss": "MSE + 0.01*CAL + 1e-5*VGG", "weights": "reinitialized"}
    save_config(run_dir, config)
    
    cb_ckpt = ModelCheckpoint(out_keras, monitor='val_loss', save_best_only=True, verbose=1)
    cb_log = CSVLogger(os.path.join(run_dir, "training_log.csv"), append=False)
    
    model.fit(X, Y, epochs=epochs, batch_size=4, validation_split=val_split, callbacks=[cb_ckpt, cb_log], verbose=1)
    print(f"✅ Long run {name} completed!")

def run_evaluate():
    if not verify_dataset(DATA_EVAL_LOW, DATA_EVAL_HIGH, "Evaluation"): sys.exit(1)
    X, Y = load_dataset(DATA_EVAL_LOW, DATA_EVAL_HIGH)
    out_dir = os.path.join(SAVE_DIR, "evaluation_results")
    os.makedirs(out_dir, exist_ok=True)
    
    import glob
    model_files = sorted(glob.glob(os.path.join(SAVE_DIR, "**", "*.keras"), recursive=True))
    
    print(f"Found {len(model_files)} models to evaluate.")
    
    for mf in model_files:
        try:
            mname = os.path.basename(mf)
            print(f"Evaluating {mname}...")
            model = load_model(mf, compile=False)
            preds = model.predict(X, batch_size=4, verbose=0)
            preds = np.clip(preds, 0.0, 1.0)
            
            mse = float(np.mean((preds - Y)**2))
            psnr = float(tf.reduce_mean(tf.image.psnr(preds, Y, max_val=1.0)).numpy())
            ssim = float(tf.reduce_mean(tf.image.ssim(preds, Y, max_val=1.0)).numpy())
            
            print(f"  -> MSE: {mse:.6f}, PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}")
            
            # Save 1 comparison image
            if len(X) > 0:
                concat = np.concatenate([X[0], preds[0], Y[0]], axis=1)
                concat = (concat * 255).astype(np.uint8)
                Image.fromarray(concat).save(os.path.join(out_dir, f"{mname}_sample.png"))
        except Exception as e:
            print(f"  Error on {mf}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOL dataset experiments for Perceptual LowLight Enhancer")
    parser.add_argument("--baseline", action="store_true", help="Run 100 epoch baseline model")
    parser.add_argument("--perceptual", action="store_true", help="Run 20+100 epoch perceptual model")
    parser.add_argument("--experiments", action="store_true", help="Run loss weight experiments")
    parser.add_argument("--long", action="store_true", help="Run 500 epoch fresh training")
    parser.add_argument("--very-long", action="store_true", help="Run 5000 epoch fresh training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all trained models in models/training_runs")
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.baseline: run_baseline()
    if args.perceptual: run_perceptual()
    if args.experiments: run_experiments()
    if args.long: run_long(500, 0.10, "500")
    if args.very_long: run_long(5000, 0.25, "5000")
    if args.evaluate: run_evaluate()

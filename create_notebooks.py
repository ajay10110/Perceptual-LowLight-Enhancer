import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    for cell in cells:
        cell_type = cell.get("type", "code")
        source = cell.get("source", "")
        # Ensure source is a list of strings with newlines
        if isinstance(source, str):
            source = [line + '\n' for line in source.split('\n')]
            if source:
                source[-1] = source[-1].rstrip('\n')
        
        nb_cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source
        }
        if cell_type == "code":
            nb_cell["execution_count"] = None
            nb_cell["outputs"] = []
            
        notebook["cells"].append(nb_cell)
        
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        
# --- BASELINE NOTEBOOK ---
baseline_cells = [
    {
        "type": "markdown",
        "source": "# 01 Baseline Training"
    },
    {
        "type": "code",
        "source": """import os
from PIL import Image
import numpy as np
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)"""
    },
    {
        "type": "code",
        "source": """low_light_dir = "../data/train/low"
enhanced_dir = "../data/train/high"

def load_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img = np.array(img).astype(np.float32) / 255.0
    return img

low_light_images, enhanced_images = [], []
for img_name in sorted(os.listdir(low_light_dir)):
    low_img_path = os.path.join(low_light_dir, img_name)
    enhanced_img_path = os.path.join(enhanced_dir, img_name)
    if os.path.exists(enhanced_img_path):
        low_light_images.append(load_image(low_img_path))
        enhanced_images.append(load_image(enhanced_img_path))

low_light_array = np.array(low_light_images).astype(np.float32)
enhanced_array = np.array(enhanced_images).astype(np.float32)
print("Dataset Loaded. Shape of low-light images:", low_light_array.shape)"""
    },
    {
        "type": "code",
        "source": """def unet_model(input_shape=(256, 256, 3)):
    inputs = tf.keras.layers.Input(input_shape)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    bottleneck = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    upconv2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(bottleneck)
    concat2 = tf.keras.layers.concatenate([conv2, upconv2], axis=-1)
    conv3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat2)
    upconv1 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv3)
    concat1 = tf.keras.layers.concatenate([conv1, upconv1], axis=-1)
    conv4 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
    outputs = tf.keras.layers.Conv2D(3, 1, activation='sigmoid')(conv4)
    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

model = unet_model()
model.summary()"""
    },
    {
        "type": "code",
        "source": """from tensorflow.image import psnr, ssim

def psnr_metric(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(psnr(y_true, y_pred, max_val=1.0))

def ssim_metric(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

def color_alignment_loss(y_true, y_pred):
    mean_true = tf.reduce_mean(y_true, axis=[1,2])
    mean_pred = tf.reduce_mean(y_pred, axis=[1,2])
    std_true = tf.math.reduce_std(y_true, axis=[1,2])
    std_pred = tf.math.reduce_std(y_pred, axis=[1,2])
    return tf.reduce_mean(tf.square(mean_true - mean_pred)) + tf.reduce_mean(tf.square(std_true - std_pred))

def combined_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    cal_loss = color_alignment_loss(y_true, y_pred)
    return mse_loss + 0.01 * cal_loss

model.compile(optimizer='adam', loss=combined_loss, metrics=['mse', psnr_metric, ssim_metric])"""
    },
    {
        "type": "code",
        "source": """history = model.fit(low_light_array, enhanced_array, epochs=100, batch_size=4)
os.makedirs("../models", exist_ok=True)
model.save("../models/original_baseline_model.keras")"""
    }
]

# --- PERCEPTUAL NOTEBOOK ---
perceptual_cells = [
    {
        "type": "markdown",
        "source": "# 02 Perceptual Training (Including Long Training Runs)"
    },
    {
        "type": "code",
        "source": """import os
from PIL import Image
import numpy as np
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

print("TensorFlow version:", tf.__version__)"""
    },
    {
        "type": "code",
        "source": """train_low_dir = "../data/train/low"
train_high_dir = "../data/train/high"

def load_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img = np.array(img).astype(np.float32) / 255.0
    return img

low_light_images, enhanced_images = [], []
for img_name in sorted(os.listdir(train_low_dir)):
    low_path = os.path.join(train_low_dir, img_name)
    high_path = os.path.join(train_high_dir, img_name)
    if os.path.exists(high_path):
        low_light_images.append(load_image(low_path))
        enhanced_images.append(load_image(high_path))

low_light_array = np.array(low_light_images, dtype=np.float32)
enhanced_array = np.array(enhanced_images, dtype=np.float32)
print(f"Loaded training data: {low_light_array.shape}, {enhanced_array.shape}")"""
    },
    {
        "type": "code",
        "source": """def unet_model(input_shape=(256, 256, 3)):
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

model = unet_model()"""
    },
    {
        "type": "code",
        "source": """feature_layer_names = ['block2_conv2', 'block3_conv4']
base_vgg = VGG19(include_top=False, weights='imagenet')
base_vgg.trainable = False
feat_outputs = [base_vgg.get_layer(name).output for name in feature_layer_names]
vgg_feat_extractor = Model(inputs=base_vgg.input, outputs=feat_outputs)
vgg_feat_extractor.trainable = False

@tf.function
def perceptual_loss(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    y_true_vgg = vgg_preprocess(y_true * 255.0)
    y_pred_vgg = vgg_preprocess(y_pred * 255.0)
    feats_true = vgg_feat_extractor(y_true_vgg)
    feats_pred = vgg_feat_extractor(y_pred_vgg)
    if not isinstance(feats_true, (list, tuple)):
        feats_true = [feats_true]; feats_pred = [feats_pred]
    loss = 0.0
    for ft, fp in zip(feats_true, feats_pred):
        loss += tf.reduce_mean(tf.square(ft - fp))
    loss = loss / float(len(feats_true))
    return loss"""
    },
    {
        "type": "code",
        "source": """def psnr_metric(y_true, y_pred):
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

def combined_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    cal_loss = color_alignment_loss(y_true, y_pred)
    p_loss = perceptual_loss(y_true, y_pred)
    return mse_loss + 0.01 * cal_loss + 0.00001 * p_loss"""
    },
    {
        "type": "code",
        "source": """save_dir = "../models"
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, "best_model_perceptual.keras")
log_path = os.path.join(save_dir, "training_log.csv")

checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path, monitor='loss', save_best_only=True, mode='min', verbose=1)
csv_logger = CSVLogger(log_path, append=True)

model.compile(optimizer='adam', loss=combined_loss, metrics=['mse', psnr_metric, ssim_metric])
history = model.fit(low_light_array, enhanced_array, epochs=20, batch_size=4, callbacks=[checkpoint_cb, csv_logger], verbose=1)"""
    },
    {
        "type": "code",
        "source": """# Resume training for 100 epochs
from tensorflow.keras.models import load_model
model = load_model(checkpoint_path, compile=False)
model.compile(optimizer='adam', loss=combined_loss, metrics=['mse', psnr_metric, ssim_metric])
history2 = model.fit(low_light_array, enhanced_array, epochs=100, batch_size=4, callbacks=[checkpoint_cb, csv_logger], verbose=1)"""
    },
    {
        "type": "code",
        "source": """# Fresh 500-epoch training
model = load_model(checkpoint_path, compile=False)
for layer in model.layers:
    if hasattr(layer, 'kernel_initializer'):
        new_weights = [init(w.shape, dtype=w.dtype) for w, init in zip(layer.get_weights(), [layer.kernel_initializer if hasattr(layer, 'kernel_initializer') else None for _ in layer.get_weights()])]
        if new_weights:
            layer.set_weights(new_weights)

model.compile(optimizer='adam', loss=combined_loss, metrics=['mse', psnr_metric, ssim_metric])
run_dir = os.path.join(save_dir, "fresh_training_500")
os.makedirs(run_dir, exist_ok=True)
checkpoint_path_500 = os.path.join(run_dir, "best_model_fresh_500.keras")
checkpoint_cb = ModelCheckpoint(checkpoint_path_500, monitor='val_loss', save_best_only=True, verbose=1)
csv_logger = CSVLogger(os.path.join(run_dir, "training_log_500.csv"), append=False)

history_500 = model.fit(low_light_array, enhanced_array, epochs=500, batch_size=4, validation_split=0.1, callbacks=[checkpoint_cb, csv_logger], verbose=1)"""
    },
    {
        "type": "code",
        "source": """# Fresh 5000-epoch training
model = load_model(checkpoint_path, compile=False)
for layer in model.layers:
    if hasattr(layer, 'kernel_initializer'):
        new_weights = [init(w.shape, dtype=w.dtype) for w, init in zip(layer.get_weights(), [layer.kernel_initializer if hasattr(layer, 'kernel_initializer') else None for _ in layer.get_weights()])]
        if new_weights:
            layer.set_weights(new_weights)

model.compile(optimizer='adam', loss=combined_loss, metrics=['mse', psnr_metric, ssim_metric])
run_dir = os.path.join(save_dir, "fresh_training_5000")
os.makedirs(run_dir, exist_ok=True)
checkpoint_path_5000 = os.path.join(run_dir, "best_model_fresh_5000.keras")
checkpoint_cb = ModelCheckpoint(checkpoint_path_5000, monitor='val_loss', save_best_only=True, verbose=1)
csv_logger = CSVLogger(os.path.join(run_dir, "training_log_5000.csv"), append=False)

history_5000 = model.fit(low_light_array, enhanced_array, epochs=5000, batch_size=4, validation_split=0.25, callbacks=[checkpoint_cb, csv_logger], verbose=1)"""
    }
]

create_notebook("notebooks/01_baseline_training.ipynb", baseline_cells)
create_notebook("notebooks/02_perceptual_training.ipynb", perceptual_cells)
print("Created notebooks.")

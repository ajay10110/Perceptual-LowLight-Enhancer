import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_log(csv_path="training_log.csv", save_path="docs/training_log_plot.png"):
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found!")
        print("Please download it from Google Drive and place it in this folder.")
        return
        
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Use 'epoch' column if it exists, otherwise use the row index
    epochs = df['epoch'] if 'epoch' in df.columns else df.index
    
    # Set up the professional plot style
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    if 'loss' in df.columns:
        plt.plot(epochs, df['loss'], label='Training Loss (Perceptual + Color)', color='#1f77b4', linewidth=2.5)
        
    # Plot validation loss if it exists
    if 'val_loss' in df.columns:
        plt.plot(epochs, df['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2.5)
        
    # Formatting to make it look professional for GitHub/Resume
    plt.title('U-Net Training Convergence (500 Epochs)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
    
    # Clean grid
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # Ensure the docs directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the high-resolution image
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Beautiful, high-resolution plot successfully saved to {save_path}!")
    print("You can now add it to your GitHub repository!")

if __name__ == "__main__":
    plot_training_log()

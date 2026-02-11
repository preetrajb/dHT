"""
Basic Tokenization Example with dHT

This script demonstrates the basic usage of dHT tokenization on images.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from dht.tok.tokenizer import dHTTokenizer


def create_sample_image(size=224):
    """Create a simple test image with distinct color regions."""
    img = torch.zeros(1, 3, size, size)
    
    # Create different colored regions
    img[:, 0, :size//2, :size//2] = 0.8  # Red region (top-left)
    img[:, 1, :size//2, size//2:] = 0.8  # Green region (top-right)
    img[:, 2, size//2:, :size//2] = 0.8  # Blue region (bottom-left)
    img[:, :, size//2:, size//2:] = 0.6  # Gray region (bottom-right)
    
    return img


def visualize_segmentation(seg, original_img, save_path=None):
    """Visualize the segmentation map with colored regions."""
    seg_np = seg[0].cpu().numpy()
    n_segments = seg_np.max() + 1
    
    # Create a colored segmentation map
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_segments, 20)))
    colored_seg = colors[seg_np % 20]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_img[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(colored_seg[:, :, :3])
    axes[1].set_title(f"dHT Segmentation ({n_segments} tokens)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tokenizer
    tokenizer = dHTTokenizer(
        in_ch=3,
        hid_ch=8,
        similarity='gaussian',
        criterion='aicc',
        iota=5.0,
        cmp=0.1,
        compute_grad=True,
    ).to(device)
    
    print("Tokenizer initialized")
    
    # Create or load image
    img = create_sample_image(224).to(device)
    
    # Tokenize
    tokenizer.eval()
    with torch.no_grad():
        result = tokenizer(img)
    
    print(f"\nTokenization Results:")
    print(f"  Number of tokens: {result.nV}")
    print(f"  Token features shape: {result.fV.shape}")
    print(f"  Segmentation shape: {result.seg.shape}")
    
    # Analyze token sizes
    token_sizes = result.seg.view(-1).bincount()
    print(f"\nToken Size Statistics:")
    print(f"  Min token size: {token_sizes.min().item()} pixels")
    print(f"  Max token size: {token_sizes.max().item()} pixels")
    print(f"  Mean token size: {token_sizes.float().mean().item():.1f} pixels")
    
    # Visualize
    visualize_segmentation(result.seg, img, save_path='tokenization_result.png')
    
    print("\nComparison with fixed patches:")
    fixed_patches = (224 // 16) ** 2  # Standard ViT uses 16x16 patches
    print(f"  Fixed ViT patches: {fixed_patches}")
    print(f"  dHT tokens: {result.nV}")
    print(f"  Reduction: {(1 - result.nV / fixed_patches) * 100:.1f}%")


if __name__ == "__main__":
    main()

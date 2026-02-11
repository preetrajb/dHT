"""
Inference Example with dHT

This script demonstrates how to use a trained dHT model for inference.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import time

from dht.nn.transformer import dHTClassifier


# ImageNet class names (subset for demo)
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich"
]


def load_image(image_path, size=224):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = dHTClassifier(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        heads=args.heads,
        depth=args.depth,
        n_classes=args.n_classes,
        channels=3,
        compute_grad=True,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Load image
    print(f"\nLoading image from {args.image}...")
    img_tensor, img_pil = load_image(args.image, size=224)
    img_tensor = img_tensor.to(device)
    
    # Inference
    print(f"Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)
    
    inference_time = time.time() - start_time
    
    # Get top-k predictions
    top_probs, top_indices = probabilities[0].topk(args.top_k)
    
    print(f"\nInference time: {inference_time * 1000:.2f}ms")
    print(f"\nTop-{args.top_k} predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        class_name = IMAGENET_CLASSES[idx] if idx < len(IMAGENET_CLASSES) else f"Class {idx}"
        print(f"  {i+1}. {class_name}: {prob.item() * 100:.2f}%")
    
    # Analyze tokenization
    if args.show_tokens:
        print(f"\nTokenization analysis:")
        with torch.no_grad():
            tok_result = model.tokenizer(img_tensor)
        
        print(f"  Number of tokens: {tok_result.nV}")
        print(f"  Comparison to fixed ViT: {(224//16)**2} patches")
        print(f"  Token reduction: {(1 - tok_result.nV / 196) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with dHT model')
    
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Model architecture (should match training)
    parser.add_argument('--embed-dim', type=int, default=384, help='Embedding dimension')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer blocks')
    parser.add_argument('--n-classes', type=int, default=10, help='Number of classes')
    
    # Inference
    parser.add_argument('--top-k', type=int, default=5, help='Show top-k predictions')
    parser.add_argument('--show-tokens', action='store_true', help='Show tokenization info')
    
    args = parser.parse_args()
    main(args)

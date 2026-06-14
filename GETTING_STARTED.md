# Getting Started with dHT: Complete Guide

## Overview

This guide provides a comprehensive introduction to using Differentiable Hierarchical Tokenization (dHT) with Vision Transformer models and video understanding tasks.

## What You'll Learn

1. **Core Concepts**: Understanding dHT tokenization
2. **ViT Integration**: Using dHT with Vision Transformers
3. **Training**: Setting up training pipelines
4. **Embedding Mechanism**: How the embedding fix works
5. **Model Adaptation**: Retrofitting existing ViT models
6. **Video Applications**: Extending to video models like ViViT

## Repository Structure

```
dHT/
├── dht/                          # Main package
│   ├── tok/                      # Tokenization modules
│   │   ├── tokenizer.py         # dHTTokenizer
│   │   ├── extractor.py         # Feature extraction
│   │   └── embedder.py          # Embedding layer
│   ├── nn/                       # Neural network modules
│   │   └── transformer.py       # dHT Transformer models
│   └── utils/                    # Utility functions
├── notebooks/                    # Tutorial notebooks
│   ├── 01_dht_tokenization_basics.ipynb
│   ├── 02_vit_integration.ipynb
│   ├── 03_training_pipeline.ipynb
│   ├── 04_embedding_fix.ipynb
│   ├── 05_adapting_vit_models.ipynb
│   └── 06_video_models_vivit.ipynb
├── examples/                     # Example scripts
│   ├── basic_tokenization.py
│   ├── train_classifier.py
│   └── inference.py
└── README.md
```

## Quick Start

### Installation

```bash
# Install from GitHub
pip install git+https://github.com/dsb-ifi/dHT.git

# Or clone and install
git clone https://github.com/dsb-ifi/dHT.git
cd dHT
pip install -e .
```

### Basic Usage

```python
import torch
from dht.tok.tokenizer import dHTTokenizer

# Create tokenizer
tokenizer = dHTTokenizer(
    in_ch=3,           # RGB channels
    hid_ch=8,          # Hidden channels
    similarity='gaussian',
    criterion='aicc',
    compute_grad=True
)

# Tokenize an image
img = torch.randn(1, 3, 224, 224)
result = tokenizer(img)

print(f"Number of tokens: {result.nV}")
print(f"Feature shape: {result.fV.shape}")
```

## Learning Path

### Step 1: Understand dHT Basics (30 minutes)

**Start with**: `notebooks/01_dht_tokenization_basics.ipynb`

Learn:
- What dHT is and how it differs from fixed patches
- How to create and use a tokenizer
- Visualizing segmentation results
- Understanding token features

**Try**: `examples/basic_tokenization.py`

### Step 2: Build ViT Models (45 minutes)

**Continue with**: `notebooks/02_vit_integration.ipynb`

Learn:
- Building complete dHT-ViT models
- Understanding the pipeline: Tokenizer → Extractor → Embedder → Transformer
- Handling variable token counts
- Model capacity options (S/M/B/L/H)

**Key concepts**:
- Adaptive token counts based on image complexity
- Attention masking for variable-length sequences
- Backward compatibility with ViT architectures

### Step 3: Training Models (1 hour)

**Work through**: `notebooks/03_training_pipeline.ipynb`

Learn:
- Setting up dataloaders
- Training loops with proper gradient flow
- Freezing/unfreezing components
- Checkpointing and evaluation

**Practice**: `examples/train_classifier.py`

```bash
python examples/train_classifier.py \
    --epochs 10 \
    --batch-size 8 \
    --embed-dim 384 \
    --depth 6
```

### Step 4: Understanding Embeddings (45 minutes)

**Deep dive**: `notebooks/04_embedding_fix.ipynb`

Learn:
- How the embedding mechanism works
- Positional encoding interpolation
- Mean injection technique
- Resolution adaptation

**Key insights**:
- Why dHT is backward compatible with ViT
- How variable-sized tokens are embedded
- Handling different image resolutions

### Step 5: Adapt Existing Models (1 hour)

**Explore**: `notebooks/05_adapting_vit_models.ipynb`

Learn:
- Retrofitting pre-trained ViT models with dHT
- Weight transfer strategies
- Fine-tuning approaches
- Performance comparison

**Use case**: When you have a pre-trained ViT and want to add adaptive tokenization

### Step 6: Video Applications (1 hour)

**Advanced**: `notebooks/06_video_models_vivit.ipynb`

Learn:
- Applying dHT to video models
- Frame-by-frame vs temporal tokenization
- ViViT architecture variants
- Training tips for video

## Common Workflows

### Workflow 1: Image Classification from Scratch

```python
from dht.nn.transformer import dHTClassifier

# Create model
model = dHTClassifier(
    embed_dim=384,
    patch_size=16,
    heads=6,
    depth=12,
    n_classes=1000,
    compute_grad=True
)

# Train with your dataset
# See examples/train_classifier.py
```

### Workflow 2: Fine-tune Pre-trained ViT

```python
from dht.nn.transformer import dHTEncoder

# Create dHT model
model = dHTEncoder.build('B', patch_size=16)

# Transfer weights from pre-trained ViT
# (See notebook 05)

# Freeze transformer, train tokenizer
model.freeze_blocks()

# Fine-tune on your data
```

### Workflow 3: Video Understanding

```python
# Process video frames
B, T, C, H, W = video.shape  # Batch, Time, Channels, Height, Width

for t in range(T):
    frame = video[:, t]
    result = tokenizer(frame)
    # Process tokens...
```

## Key Parameters

### Tokenizer Parameters

- **iota** (default: 5.0): Information criterion weight
  - Higher → fewer tokens
  - Lower → more tokens
  
- **cmp** (default: 0.1): Compression parameter
  - Higher → fewer tokens
  - Lower → more tokens

- **similarity**: 'gaussian', 'cosine', etc.
  - How to measure region similarity

- **criterion**: 'aicc', 'aic', 'bic'
  - Information criterion for merging decisions

### Model Parameters

- **embed_dim**: Transformer embedding dimension
  - 384 (Small), 512 (Medium), 768 (Base), 1024 (Large)

- **patch_size**: Base patch size for feature extraction
  - Common: 16 or 32

- **depth**: Number of transformer layers
  - 12 (Small/Medium/Base), 24 (Large), 32 (Huge)

- **heads**: Number of attention heads
  - Typically embed_dim // 64

## Tips and Best Practices

### Memory Management

```python
# For large images or limited memory:
# 1. Use smaller batch sizes
# 2. Use gradient accumulation
# 3. Control target token count

result = tokenizer(img, target=100)  # Target ~100 tokens
```

### Training Strategy

1. **Start simple**: Train tokenizer first, then transformer
2. **Gradual unfreezing**: Unfreeze layers progressively
3. **Lower learning rates**: Use 1e-4 to 1e-5 for fine-tuning
4. **Warmup**: Use 5-10 epoch warmup for stable training

### Resolution Adaptation

```python
# dHT works with various resolutions
img_224 = torch.randn(1, 3, 224, 224)
img_384 = torch.randn(1, 3, 384, 384)

# Both work without changes
result_224 = model(img_224)
result_384 = model(img_384)
```

## Performance Expectations

### Token Reduction

- Simple images: 50-70% fewer tokens than fixed ViT
- Complex images: 20-40% fewer tokens
- Videos (static scenes): Up to 80% reduction

### Accuracy

- Comparable to fixed ViT on ImageNet
- Better on tasks requiring fine details
- Improved efficiency-accuracy tradeoff

## Troubleshooting

### Issue: Out of memory

**Solution**: Reduce batch size or target token count
```python
result = tokenizer(img, target=50)  # Fewer tokens
```

### Issue: Training unstable

**Solution**: Lower learning rate, add warmup
```python
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5)
```

### Issue: Tokens too coarse

**Solution**: Adjust tokenizer parameters
```python
tokenizer = dHTTokenizer(
    iota=2.0,   # Lower for more tokens
    cmp=0.05    # Lower for more tokens
)
```

## Next Steps

1. **Start with notebooks**: Work through all 6 notebooks
2. **Run examples**: Try the example scripts
3. **Experiment**: Modify parameters and see effects
4. **Integrate**: Add dHT to your projects
5. **Contribute**: Share improvements or use cases

## Resources

- **Paper**: [arXiv:2511.02652](https://arxiv.org/abs/2511.02652)
- **Website**: [dsb-ifi.github.io/dHT/](https://dsb-ifi.github.io/dHT/)
- **Code**: [github.com/dsb-ifi/dHT](https://github.com/dsb-ifi/dHT)
- **Issues**: [GitHub Issues](https://github.com/dsb-ifi/dHT/issues)

## Citation

```bibtex
@inproceedings{aasan2025dht,
  title={Differentiable Hierarchical Visual Tokenization},
  author={Aasan, Marius and Hjelkrem-Tan, Martine and Catalano, Nico and Choi, Changkyu and Ram\'irez Rivera, Ad\'in},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=y8VWYf5cVI}
}
```

## Support

For questions or issues:
1. Check the notebooks for examples
2. Review the troubleshooting section
3. Open an issue on GitHub
4. Refer to the paper for theoretical details

Happy tokenizing! 🚀

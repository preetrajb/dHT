# dHT Examples

This directory contains example scripts and notebooks demonstrating how to use Differentiable Hierarchical Tokenization (dHT).

## Notebooks

Located in `../notebooks/`:

1. **01_dht_tokenization_basics.ipynb**
   - Introduction to dHT tokenization
   - Basic usage and visualization
   - Comparison with fixed patches
   - Parameter exploration

2. **02_vit_integration.ipynb**
   - Integrating dHT with Vision Transformers
   - Understanding the dHT pipeline
   - Building complete models
   - Adaptive token counts

3. **03_training_pipeline.ipynb**
   - Setting up training loops
   - Dataset preparation
   - Optimization strategies
   - Model checkpointing

4. **04_embedding_fix.ipynb**
   - Understanding the embedding mechanism
   - Positional encoding interpolation
   - Mean injection technique
   - Resolution adaptation

5. **05_adapting_vit_models.ipynb**
   - Retrofitting pre-trained ViT models
   - Weight transfer strategies
   - Fine-tuning approaches
   - Efficiency analysis

6. **06_video_models_vivit.ipynb**
   - Applying dHT to video models
   - Temporal tokenization
   - ViViT architecture variants
   - Training tips for video

## Scripts

### Basic Usage

**basic_tokenization.py**
```bash
python examples/basic_tokenization.py
```
Demonstrates basic tokenization on a sample image.

### Training

**train_classifier.py**
```bash
python examples/train_classifier.py \
    --epochs 10 \
    --batch-size 8 \
    --embed-dim 384 \
    --depth 6 \
    --n-classes 10
```

Options:
- `--train-samples`: Number of training samples (default: 1000)
- `--val-samples`: Number of validation samples (default: 200)
- `--embed-dim`: Model embedding dimension (default: 384)
- `--patch-size`: Base patch size (default: 16)
- `--heads`: Number of attention heads (default: 6)
- `--depth`: Number of transformer layers (default: 6)
- `--epochs`: Training epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--save-path`: Where to save checkpoints (default: dht_checkpoint.pth)

### Inference

**inference.py**
```bash
python examples/inference.py \
    --image path/to/image.jpg \
    --checkpoint dht_checkpoint.pth \
    --show-tokens
```

Options:
- `--image`: Path to input image (required)
- `--checkpoint`: Path to model checkpoint (required)
- `--top-k`: Number of top predictions to show (default: 5)
- `--show-tokens`: Display tokenization statistics

## Quick Start

1. **Explore tokenization:**
   ```bash
   python examples/basic_tokenization.py
   ```

2. **Train a model:**
   ```bash
   python examples/train_classifier.py --epochs 5
   ```

3. **Run inference:**
   ```bash
   python examples/inference.py \
       --image your_image.jpg \
       --checkpoint dht_checkpoint.pth
   ```

## Using Notebooks

Start Jupyter:
```bash
jupyter notebook notebooks/
```

Or use JupyterLab:
```bash
jupyter lab notebooks/
```

Recommended order:
1. Start with `01_dht_tokenization_basics.ipynb` to understand the core concepts
2. Progress through `02_vit_integration.ipynb` to see how to build models
3. Learn training in `03_training_pipeline.ipynb`
4. Deep dive into embeddings with `04_embedding_fix.ipynb`
5. Adapt existing models in `05_adapting_vit_models.ipynb`
6. Explore video applications in `06_video_models_vivit.ipynb`

## Requirements

All examples require:
- PyTorch >= 2.7.0
- torchvision >= 0.22.0
- numpy >= 2.2.0
- pillow >= 12.0.0
- matplotlib (for visualization)

For notebooks:
- jupyter
- ipykernel

Install with:
```bash
pip install torch torchvision numpy pillow matplotlib jupyter ipykernel
```

## Common Use Cases

### Image Classification
See `train_classifier.py` and `02_vit_integration.ipynb`

### Fine-tuning Pre-trained Models
See `05_adapting_vit_models.ipynb`

### Video Understanding
See `06_video_models_vivit.ipynb`

### Custom Tokenization
See `01_dht_tokenization_basics.ipynb` and `04_embedding_fix.ipynb`

## Tips

1. **Memory Management**: Use smaller batch sizes or gradient accumulation for large models
2. **Token Count**: Adjust `iota` and `cmp` parameters in tokenizer to control token count
3. **Training**: Start with frozen tokenizer, then fine-tune end-to-end
4. **Resolution**: dHT works with various image sizes, not just 224x224
5. **Visualization**: Use the visualization functions to understand tokenization behavior

## Support

For issues or questions:
- GitHub Issues: https://github.com/dsb-ifi/dHT/issues
- Paper: https://arxiv.org/abs/2511.02652
- Website: https://dsb-ifi.github.io/dHT/

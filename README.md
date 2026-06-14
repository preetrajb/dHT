<div align="center">

# $\text{Differentiable Hierarchical Visual Tokenization}$

**[Marius Aasan](https://www.mn.uio.no/ifi/english/people/aca/mariuaas/)$^1$, [Martine Hjelkrem-Tan](https://www.mn.uio.no/ifi/english/people/aca/matan/)$^1$, [Nico Catalano](https://nicocatalano.github.io/)$^2$, [Chankyu Choi](https://en.uit.no/ansatte/person?p_document_id=617277)$^3$, [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)$^1$** <br>


**${}^1\underset{\text{Department of Informatics}}{\text{University of Oslo}}$** $\hspace{1em}$ 
**${}^2\underset{\text{Artificial Intelligence and Robotics Lab}}{\text{Polytechnic University of Milan}}$** $\hspace{1em}$ 
**${}^3\underset{\text{Department of Physics and Technology}}{\text{UiT The Arctic University of Norway}}$** 
<br>

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/dHT/)
[![PaperArxiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2511.02652)
[![PaperNeurIPS](https://img.shields.io/badge/Paper-NeurIPS_2025-blue)](https://openreview.net/forum?id=y8VWYf5cVI)
[![SpotlightNeurIPS](https://img.shields.io/badge/Spotlight-NeurIPS_2025-cyan)](https://neurips.cc/virtual/2025/poster/115180)
[![NotebookR2V](https://img.shields.io/badge/Notebook-R2V-orange)](https://nbviewer.jupyter.org/github/dsb-ifi/dHT/blob/main/demo_raster2svg.ipynb)<br>

![dHT Figure 1](/assets/dht_teaser_transparent.png#gh-light-mode-only "Examples of feature trajectoreis with SPoT-ON")
![dHT Figure 1](/assets/dht_teaser_transparent_dark.png#gh-dark-mode-only "Examples of feature trajectoreis with SPoT-ON")

## $\text{Abstract}$
</div>
<div style="font-family: serif;">
Vision Transformers rely on fixed patch tokens that ignore the spatial and semantic structure of images. In this work, we introduce an end-to-end differentiable tokenizer that adapts to image content with pixel-level granularity while remaining backward-compatible with existing architectures for retrofitting pretrained models. Our method uses hierarchical model selection with information criteria to provide competitive performance in both image-level classification and dense-prediction tasks, and even supports out-of-the-box raster-to-vector conversion.
</div>


## $\partial\text{HT}$: Differentiable Hierarchical Visual Tokenization

This repo contains code for **Differentiable Hierarchical Visual Tokenization**, accepted as a spotlight paper for NeurIPS 2025.

For an introduction to our work, visit the [project webpage](https://dsb-ifi.github.io/dHT/). 

## Installation

📚 **New to dHT?** Check out our [**Getting Started Guide**](GETTING_STARTED.md) for a complete tutorial!

The repo can currently be installed as a package via:

```bash
# HTTPS
pip install git+https://github.com/dsb-ifi/dHT.git

# SSH
pip install git+ssh://git@github.com/dsb-ifi/dHT.git
```

## Loading models

You can load the Superpixel Transformer models easily via `torch.hub`:

```python
# Example with raster-to-vector model
model = torch.hub.load(
    'dsb-ifi/dht', 
    'dht_ras2vec',
    pretrained=True,
    source='github',
)
```

This will load the model and downloaded the pretrained weights, stored in your local `torch.hub` directory.

## Documentation & Examples

### 📓 Notebooks

Comprehensive Jupyter notebooks covering all aspects of dHT:

1. **[Tokenization Basics](notebooks/01_dht_tokenization_basics.ipynb)** - Introduction to dHT tokenization
2. **[ViT Integration](notebooks/02_vit_integration.ipynb)** - Using dHT with Vision Transformers
3. **[Training Pipeline](notebooks/03_training_pipeline.ipynb)** - Complete training examples
4. **[Embedding Fix](notebooks/04_embedding_fix.ipynb)** - Understanding positional embeddings and mean injection
5. **[Adapting ViT Models](notebooks/05_adapting_vit_models.ipynb)** - Retrofit pre-trained models with dHT
6. **[Video Models (ViViT)](notebooks/06_video_models_vivit.ipynb)** - Applying dHT to video understanding

### 🚀 Example Scripts

Located in the `examples/` directory:

- **basic_tokenization.py** - Basic tokenization demonstration
- **train_classifier.py** - Complete training script
- **inference.py** - Run inference with trained models

See [examples/README.md](examples/README.md) for detailed usage instructions.

### Quick Start

```bash
# Run basic tokenization example
python examples/basic_tokenization.py

# Train a model
python examples/train_classifier.py --epochs 10

# Explore notebooks
jupyter notebook notebooks/
```

## Key Features

- **Adaptive Tokenization**: Creates variable-sized tokens based on image content
- **Backward Compatible**: Works with existing Vision Transformer architectures
- **Differentiable**: End-to-end trainable
- **Efficient**: Fewer tokens for simple regions, more for complex areas
- **Flexible**: Supports various image sizes and video inputs

## Citation

If you find our work useful, please consider citing our paper.

```
@inproceedings{aasan2025dht,
  title={Differentiable Hierarchical Visual Tokenization},
  author={Aasan, Marius and Hjelkrem-Tan, Martine and Catalano, Nico and Choi, Changkyu and Ram\'irez Rivera, Ad\'in},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=y8VWYf5cVI}
}
```

---

🚧 **NOTE:** The `hubconf.py` is still under construction, and will be updated with classification models soon.
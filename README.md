# X-Ray Pneumonia Classification with Noise Robustness Analysis

A deep learning project that classifies chest X-ray images as normal or pneumonia, and evaluates model robustness to Gaussian and Poisson noise.

## Project Overview

This project trains convolutional neural networks (CNNs) to detect pneumonia from chest X-rays and analyzes how different types of imaging noise affect model predictions. We implement noise injection simulating real-world X-ray imaging artifacts and compare multiple CNN architectures.

## File Structure

### Core Modules

- **`classifier.py`** - Defines the PyTorch Lightning training module with the training loop, validation, testing procedures, and metrics computation.

- **`dataset.py`** - Custom dataset class that loads X-ray images and optionally applies noise during data loading for robustness testing.

- **`models.py`** - Contains all CNN architecture definitions, adapted from pretrained ImageNet models to work with grayscale X-ray images.

- **`noise.py`** - Implements Poisson and Gaussian noise injection functions to simulate real-world X-ray imaging artifacts.

- **`main.py`** - Command-line interface for training models from scratch and evaluating trained models on clean or noisy test sets.

### Utility Scripts

- **`split.py`** - Generates train/validation/test splits and creates a CSV file mapping image paths to labels for reproducible experiments.

- **`experiments.sh`** - Automates running multiple training and evaluation experiments across different models and noise configurations.

- **`summary.py`** - Aggregates results from multiple experiment runs into consolidated summary statistics and reports.

- **`plot.py`** - Creates visualization plots comparing model performance across different noise levels and architectures.

### Demonstration

- **`project.ipynb`** - **Interactive Jupyter notebook demonstrating the complete workflow** with a small data subset. Includes visualization, training, noise injection examples, and model comparison. **Start here for a guided walkthrough!**

- **`notebook_subset.json`** - Lists the specific image paths used in the notebook for reproducibility.

- **`setup_notebook_data.py`** - Script to copy the notebook subset images to a committable folder structure.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it to the `chest_xray/` directory.

Alternatively, for the notebook demo, the subset images are included in `chest_xray_subset/`.

## Usage

### Interactive Demo (Recommended Starting Point)

Open and run **`project.ipynb`** in Jupyter:

```bash
jupyter notebook project.ipynb
```

The notebook walks through:
- Data loading and visualization
- Training ResNet18 and a basic CNN
- Noise injection demonstration
- Robustness evaluation across noise levels
- Performance comparison plots

### Training Models

Train a model using `main.py`:

```bash
# Train ResNet18 (default)
python main.py train --model resnet18 --batch_size 32 --max_epochs 10

# Train DenseNet121
python main.py train --model densenet121 --lr 0.001 --max_epochs 15

# Train with custom settings
python main.py train --model resnet34 --batch_size 64 --lr 0.0005 --img_size 224
```

**Training arguments:**
- `--model`: Model architecture (resnet18, resnet34, densenet121, mobilenet_v2, efficientnet_b0, basic)
- `--batch_size`: Batch size for training (default: 32)
- `--max_epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--img_size`: Input image size (default: 224)
- `--data_root`: Path to dataset directory (default: chest_xray)

### Testing Models

Evaluate a trained model on clean or noisy test data:

```bash
# Test on clean images
python main.py test --model_ckpt checkpoints/resnet18-epoch=05-val_auroc=0.950.ckpt --poiss 0 --gauss 0

# Test with Poisson noise (intensity 4)
python main.py test --model_ckpt checkpoints/resnet18-epoch=05-val_auroc=0.950.ckpt --poiss 4 --gauss 0

# Test with combined Gaussian + Poisson noise
python main.py test --model_ckpt checkpoints/resnet34-epoch=02-val_auroc=1.000.ckpt --poiss 4 --gauss 4

# Test without checkpoint (random initialization - for debugging)
python main.py test --model resnet18 --poiss 2 --gauss 2
```

**Testing arguments:**
- `--model_ckpt`: Path to checkpoint file (required unless using `--model`)
- `--model`: Model architecture (if testing without checkpoint)
- `--poiss`: Poisson noise intensity (0 = no noise, higher = more noise)
- `--gauss`: Gaussian noise intensity (0 = no noise, higher = more noise)
- `--batch_size`: Batch size for evaluation (default: 32)

### Running Systematic Experiments

Run comprehensive robustness analysis across multiple models and noise levels:

```bash
bash experiments.sh
```

This will train and evaluate all model architectures on various noise configurations.

### Generating Noise Examples

Visualize noise effects on X-ray images:

```bash
python noise.py <poisson noise intensity> <gaussian noise intensity>
```

This script loads sample images, applies Poisson and Gaussian noise at the provided intensities, and displays the results. (Setting a parameter to 0 means the noise will not be applied to this image)

## Results

Evaluation results are saved as CSV files in `checkpoints/eval/` with metrics including:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Accuracy, Precision, Recall, F1-Score

Use `summary.py` to aggregate results and `plot.py` to generate visualization of model robustness.

## Project Workflow

1. **Data Preparation**: Run `split.py` to create train/val/test splits
2. **Model Training**: Use `main.py train` or `experiments.sh` to train models
3. **Noise Analysis**: Use `main.py test` with various `--poiss` and `--gauss` values
4. **Result Aggregation**: Run `summary.py` to compile results
5. **Visualization**: Use `plot.py` to generate performance charts
6. **Interactive Demo**: Explore `project.ipynb` for a complete walkthrough

## Notes

- The notebook uses a small subset (~100 images) for fast demonstration
- Full training uses the complete dataset (~5,800 images)
- Models are trained on clean images and evaluated on noisy test sets
- Noise intensities typically range from 0 (clean) to 10 (very noisy). The default experiment structure for this projetc is (0, 2, 4, 6, 8) for each type of noise.

## Requirements

`pip install -r requirements.txt`

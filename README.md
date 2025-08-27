# Multimodal Medical Prediction Model

A clean, modular implementation of a multimodal machine learning system for medical mortality prediction. This system combines clinical notes, medical codes, laboratory results, and medical images using Graph Neural Networks (GNNs) and Large Language Models (LLMs).

## Overview

This refactored codebase provides a well-organized implementation of a sophisticated medical prediction system that:

- Processes multimodal medical data (notes, codes, labs, images)
- Uses Graph Neural Networks to capture relationships between patients and medical entities
- Employs Large Language Models for text understanding and prediction
- Implements multimodal alignment techniques inspired by ImageBind
- Provides comprehensive training and evaluation pipelines

## Architecture

The system consists of several key components:

1. **Graph Neural Networks (GNNs)**: Process structured medical data with temporal and similarity-based connections
2. **ImageBind Alignment**: Aligns different modalities (text, codes, labs, images) into a common embedding space
3. **LLM Integration**: Uses LLaMA-3 for natural language understanding and mortality prediction
4. **Context Injection**: Injects multimodal context into LLM hidden states during inference

## Project Structure

```
├── config.py              # Configuration management
├── data_loader.py         # Data loading and preprocessing
├── models.py              # Neural network architectures
├── llm_utils.py           # LLM management and utilities
├── training.py            # Training and evaluation logic
├── main.py               # Main entry point
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── model.py              # Original code (for reference)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd iclr_code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up HuggingFace token:
   - Get a token from https://huggingface.co/settings/tokens
   - Update the `hf_token` in `config.py` or set as environment variable

## Configuration

The system uses a hierarchical configuration system in `config.py`:

- **ModelConfig**: Architecture parameters (dimensions, dropout, etc.)
- **TrainingConfig**: Training hyperparameters (learning rates, batch size, etc.)
- **DataConfig**: Data paths and preprocessing parameters

Key configurations to adjust:

```python
# File paths
train_data_path = "/path/to/train_data.pt"
test_data_path = "/path/to/test_data.pt"

# Model parameters
num_epochs = 10
gnn_lr = 1e-3
imagebind_lr = 1e-4

# HuggingFace token
hf_token = "your_token_here"
```

## Usage

### Basic Training

Run the main training script:

```bash
python main.py
```

### Custom Configuration

Modify `config.py` or create a custom configuration:

```python
from config import Config

# Create custom config
config = Config()
config.training.num_epochs = 20
config.training.gnn_lr = 5e-4

# Use with training pipeline
```

### Loading Pre-trained Models

```python
from training import Trainer

trainer = Trainer(config, models, graphs, llm_manager)
trainer.load_model("/path/to/saved_model.pth")
```

## Data Format

The system expects PyTorch data files with the following structure:

```python
data_entry = {
    'patient_id': str,           # Unique patient identifier
    'code_embeddings': np.array, # Medical code embeddings
    'labs': np.array,           # Laboratory values
    'image_embeddings': list,    # List of image embeddings
    'note_embeddings': np.array, # Clinical note embeddings
    'text': str,                # Raw clinical notes
    'one_year_mortality': int   # Target label (0 or 1)
}
```


## Training Process

1. **Data Loading**: Load and preprocess multimodal data
2. **Graph Construction**: Build similarity and temporal graphs
3. **Model Initialization**: Set up all neural network components
4. **Training Loop**: 
   - Process embeddings through GNNs
   - Align modalities using ImageBind
   - Generate predictions with LLM
   - Compute combined loss and update parameters
5. **Evaluation**: Test on held-out data with comprehensive metrics

## Evaluation Metrics

The system reports:
- **Accuracy**: Binary classification accuracy
- **AUC**: Area under the ROC curve
- **F1 Score**: Macro-averaged F1 score

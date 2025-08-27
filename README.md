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

## Key Features

### 1. Modular Design
- Clear separation of concerns
- Easy to extend and modify
- Reusable components

### 2. Comprehensive Logging
- Structured logging throughout the pipeline
- Training progress tracking
- Error handling and debugging

### 3. Flexible Configuration
- Centralized parameter management
- Easy hyperparameter tuning
- Environment-specific configurations

### 4. Memory Optimization
- GPU memory management
- Gradient accumulation support
- Efficient data loading

### 5. Robust Training
- Gradient clipping
- Best model saving
- Comprehensive evaluation metrics

## Model Components

### Graph Neural Networks
- **Code GNN**: Processes medical code relationships
- **Lab GNN**: Handles laboratory result patterns
- **Image GNN**: Processes medical image features

### Multimodal Alignment
- Projects different modalities to common space
- Uses InfoNCE loss for alignment
- Supports flexible fusion strategies

### Language Model Integration
- LLaMA-3 integration with context injection
- Custom hooks for multimodal conditioning
- Structured prompt engineering

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

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Enable gradient accumulation
   - Use mixed precision training

2. **Model Loading Errors**:
   - Check HuggingFace token configuration
   - Verify model name and access permissions
   - Ensure sufficient disk space

3. **Data Format Issues**:
   - Verify data file paths in config
   - Check data structure matches expected format
   - Handle missing modalities appropriately

### Performance Optimization

- Use GPU when available
- Enable gradient checkpointing for large models
- Optimize graph construction parameters
- Tune learning rates and batch sizes

## Contributing

When contributing to this codebase:

1. Follow the existing modular structure
2. Add comprehensive documentation
3. Include proper error handling
4. Update configuration as needed
5. Add tests for new functionality

## License

[Add appropriate license information]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information if applicable]
```

## Acknowledgments

This implementation builds upon several open-source libraries and research contributions in multimodal learning and medical AI. 
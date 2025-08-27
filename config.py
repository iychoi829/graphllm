"""
Configuration file for multimodal medical prediction model.
"""
import torch
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # GNN dimensions
    code_input_dim: int = 768
    code_hidden_dim: int = 768
    code_output_dim: int = 768
    
    lab_input_dim: int = 2227
    lab_hidden_dim: int = 2227
    lab_output_dim: int = 2227
    
    image_input_dim: int = 2048
    image_hidden_dim: int = 2048
    image_output_dim: int = 2048
    
    # ImageBind dimensions
    note_dim: int = 768
    common_dim: int = 4096
    temperature: float = 0.1
    
    # LLM configuration
    llm_model_name: str = "meta-llama/Meta-Llama-3-8B"
    max_sequence_length: int = 4096
    max_new_tokens: int = 10
    generation_temperature: float = 0.3
    top_p: float = 0.9
    
    # Dropout and regularization
    dropout_rate: float = 0.5
    weight_decay: float = 1e-5

@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 10
    batch_size: int = 1
    accumulation_steps: int = 1
    
    # Learning rates
    gnn_lr: float = 1e-3
    imagebind_lr: float = 1e-4
    
    # Loss weights
    reward_weight: float = 1.0
    align_weight: float = 0.01
    ratio_weight: float = 0.01
    validity_weight: float = 0.2
    accuracy_weight: float = 1.0
    class_0_weight: float = 1.0
    class_1_weight: float = 3.0
    
    # Regularization
    gradient_clip_value: float = 1.0
    context_loss_beta: float = 0.0001
    
    # Context mixing ratios
    ratio_a: float = 0.2
    ratio_b: float = 1.0

@dataclass
class DataConfig:
    """Data configuration."""
    # File paths (to be set via environment variables or config file)
    train_data_path: str = "/cbica/home/NAME/project/downsampled_data/train_data_20.pt"
    test_data_path: str = "/cbica/home/NAME/project/downsampled_data/test_data_20.pt"
    model_save_path: str = "/cbica/home/NAME/project/downsampled_data/combined_models_llama3_3layers_k1000_final_10epochs_bceLoss_lrrate_5_nnloss_allgraph.pth"
    
    # Graph construction parameters
    similarity_k: int = 1000
    similarity_threshold: float = 0.7
    
    # Data preprocessing
    prediction_threshold: float = 0.5

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    # Device configuration
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # HuggingFace token (should be set via environment variable)
    hf_token: str = "TOKEN"  # Replace with actual token or load from env
    
    def __post_init__(self):
        """Post-initialization setup."""
        if torch.cuda.is_available():
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU is not available, using CPU") 
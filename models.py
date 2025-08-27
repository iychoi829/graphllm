"""
Neural network models for multimodal medical prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional
from torch_geometric.nn import GCNConv

from config import Config

logger = logging.getLogger(__name__)


class GNN(nn.Module):
    """Graph Neural Network for processing graph-structured data."""
    
    def __init__(self, in_channels: int, hidden_channels: int, 
                 out_channels: int, dropout_rate: float = 0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.prediction_head = nn.Linear(out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            
        Returns:
            Tuple of (node_embeddings, predictions)
        """
        x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        embeddings = F.relu(self.conv3(x, edge_index))
        predictions = self.prediction_head(embeddings).squeeze(-1)
        
        return torch.nan_to_num(embeddings), torch.nan_to_num(predictions)


class SelfAttentionLayer(nn.Module):
    """Self-attention layer for fusing multimodal embeddings."""
    
    def __init__(self, embed_dim: int):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to multimodal embeddings.
        
        Args:
            x: Input tensor of shape (n_datapoints, n_modalities, embed_dim)
            
        Returns:
            Fused embeddings of shape (n_datapoints, embed_dim)
        """
        # Linear projections
        q = self.query(x)  # (n_datapoints, n_modalities, embed_dim)
        k = self.key(x)
        v = self.value(x)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        
        # Aggregate across modalities
        fused_proj = attn_output.sum(dim=1)
        
        return fused_proj


def min_max_normalize(embeds: torch.Tensor) -> torch.Tensor:
    """Min-max normalize embeddings."""
    min_val = embeds.min(dim=1, keepdim=True)[0]
    max_val = embeds.max(dim=1, keepdim=True)[0]
    normalized_embeds = (embeds - min_val) / (max_val - min_val + 1e-8)
    return normalized_embeds


class ImageBindAlignment(nn.Module):
    """Multimodal alignment module inspired by ImageBind."""
    
    def __init__(self, note_dim: int, code_dim: int, lab_dim: int, 
                 image_dim: int, common_dim: int, temperature: float):
        super(ImageBindAlignment, self).__init__()
        self.note_proj = nn.Linear(note_dim, common_dim)
        self.code_proj = nn.Linear(code_dim, common_dim)
        self.lab_proj = nn.Linear(lab_dim, common_dim)
        self.image_proj = nn.Linear(image_dim, common_dim)
        self.temperature = temperature

    def forward(self, note_embeddings: torch.Tensor, code_embeddings: torch.Tensor,
                lab_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align multimodal embeddings to a common space.
        
        Args:
            note_embeddings: Clinical note embeddings
            code_embeddings: Medical code embeddings
            lab_embeddings: Lab result embeddings
            image_embeddings: Medical image embeddings
            
        Returns:
            Tuple of (fused_embeddings, alignment_loss)
        """
        # Project to common dimension
        note_proj = self.note_proj(note_embeddings)
        code_proj = self.code_proj(code_embeddings)
        lab_proj = self.lab_proj(lab_embeddings)
        image_proj = self.image_proj(image_embeddings)

        # Normalize projections
        note_proj = min_max_normalize(note_proj)
        code_proj = min_max_normalize(code_proj)
        lab_proj = min_max_normalize(lab_proj)
        image_proj = min_max_normalize(image_proj)

        # Calculate cross-modal similarities
        similarity_losses = self._calculate_similarity_losses(
            note_proj, code_proj, lab_proj, image_proj
        )
        
        total_loss = sum(similarity_losses)

        # Fuse modalities
        fused_proj = torch.stack([note_proj, code_proj, lab_proj, image_proj], dim=1)
        fused_proj = fused_proj.sum(dim=1)  # Simple fusion by summation
        
        return fused_proj, total_loss

    def _calculate_similarity_losses(self, note_proj: torch.Tensor, code_proj: torch.Tensor,
                                   lab_proj: torch.Tensor, image_proj: torch.Tensor) -> list:
        """Calculate InfoNCE losses for cross-modal alignment."""
        batch_size = note_proj.size(0)
        labels = torch.arange(batch_size, device=note_proj.device)
        
        # Calculate similarities
        sim_note_code = torch.matmul(note_proj, code_proj.T) / self.temperature
        sim_code_note = torch.matmul(code_proj, note_proj.T) / self.temperature
        
        sim_note_lab = torch.matmul(note_proj, lab_proj.T) / self.temperature
        sim_lab_note = torch.matmul(lab_proj, note_proj.T) / self.temperature
        
        sim_note_image = torch.matmul(note_proj, image_proj.T) / self.temperature
        sim_image_note = torch.matmul(image_proj, note_proj.T) / self.temperature

        # InfoNCE losses
        losses = [
            self.info_nce_loss(sim_note_code, labels) + self.info_nce_loss(sim_code_note, labels),
            self.info_nce_loss(sim_note_lab, labels) + self.info_nce_loss(sim_lab_note, labels),
            self.info_nce_loss(sim_note_image, labels) + self.info_nce_loss(sim_image_note, labels)
        ]
        
        return losses

    def info_nce_loss(self, similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss."""
        log_prob = F.log_softmax(similarity, dim=-1)
        loss = F.nll_loss(log_prob, labels)
        return loss


class CustomLoss(nn.Module):
    """Custom loss function for medical prediction with validity constraints."""
    
    def __init__(self, weight_valid: float = 1.0, weight_accuracy: float = 1.0, 
                 weight_0: float = 1.0, weight_1: float = 3.0):
        super(CustomLoss, self).__init__()
        self.weight_valid = weight_valid
        self.weight_accuracy = weight_accuracy
        self.weight_0 = weight_0
        self.weight_1 = weight_1

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute custom loss combining validity and accuracy terms.
        
        Args:
            prediction: Model predictions (0-100 scale)
            target: Ground truth labels (0 or 1)
            
        Returns:
            Combined loss value
        """
        # Ensure tensors have batch dimension
        if prediction.dim() == 0:
            prediction = prediction.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        # Validity loss: penalize predictions outside [0, 100]
        lower_penalty = torch.relu(-prediction)
        upper_penalty = torch.relu(prediction - 100)
        validity_loss = torch.mean(lower_penalty + upper_penalty)

        # Convert to probability space
        predicted_prob = torch.clamp(prediction / 100, min=1e-7, max=1-1e-7)

        # Weighted BCE loss
        weights = torch.where(target == 1, 
                            torch.tensor(self.weight_1, device=target.device), 
                            torch.tensor(self.weight_0, device=target.device))
        
        bce_loss_fn = nn.BCELoss(weight=weights, reduction='mean')
        bce_loss = bce_loss_fn(predicted_prob, target.float())

        # Combine losses
        total_loss = self.weight_valid * validity_loss + self.weight_accuracy * bce_loss
        
        return total_loss


class ContextMixer(nn.Module):
    """Context mixing module for combining context vectors with model outputs."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.ratio_a = nn.Parameter(torch.tensor(0.2, device=device))
        self.ratio_b = nn.Parameter(torch.tensor(1.0, device=device))

    def forward(self, context_vector: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Mix context vector with model output."""
        return self.ratio_a * context_vector + self.ratio_b * output


def compute_policy_loss(prediction: torch.Tensor, ground_truth: torch.Tensor, 
                       align_loss: torch.Tensor, context_vector: torch.Tensor,
                       config: Config) -> torch.Tensor:
    """
    Compute the combined policy loss for training.
    
    Args:
        prediction: Model prediction
        ground_truth: Ground truth label
        align_loss: Alignment loss from multimodal fusion
        context_vector: Context vector from embeddings
        config: Configuration object
        
    Returns:
        Combined loss value
    """
    # Main prediction loss
    custom_loss = CustomLoss(
        weight_valid=config.training.validity_weight,
        weight_accuracy=config.training.accuracy_weight,
        weight_0=config.training.class_0_weight,
        weight_1=config.training.class_1_weight
    )
    
    prediction = prediction.float()
    ground_truth = ground_truth.float()
    
    reward_loss = custom_loss(prediction, ground_truth)
    
    # Scale auxiliary losses
    align_loss = align_loss / 100000  # Scale down alignment loss
    context_loss = torch.log(1 + context_vector.norm() + 1e-7)  # Regularize context vector

    # Combine losses with weights
    combined_loss = (config.training.reward_weight * reward_loss +
                    config.training.align_weight * align_loss +
                    config.training.context_loss_beta * context_loss)

    return combined_loss


def create_models(config: Config) -> Tuple[GNN, GNN, GNN, ImageBindAlignment]:
    """
    Create and initialize all model components.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (code_gnn, lab_gnn, image_gnn, imagebind)
    """
    logger.info("Initializing models...")
    
    code_gnn = GNN(
        config.model.code_input_dim,
        config.model.code_hidden_dim,
        config.model.code_output_dim,
        config.model.dropout_rate
    ).to(config.device)
    
    lab_gnn = GNN(
        config.model.lab_input_dim,
        config.model.lab_hidden_dim,
        config.model.lab_output_dim,
        config.model.dropout_rate
    ).to(config.device)
    
    image_gnn = GNN(
        config.model.image_input_dim,
        config.model.image_hidden_dim,
        config.model.image_output_dim,
        config.model.dropout_rate
    ).to(config.device)
    
    imagebind = ImageBindAlignment(
        config.model.note_dim,
        config.model.code_output_dim,
        config.model.lab_output_dim,
        config.model.image_output_dim,
        config.model.common_dim,
        config.model.temperature
    ).to(config.device)
    
    logger.info("Models initialized successfully")
    
    return code_gnn, lab_gnn, image_gnn, imagebind 
"""
Training and evaluation utilities for multimodal medical prediction.
"""
import torch
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict

from config import Config
from data_loader import create_graph_data, get_subject_last_labels
from models import compute_policy_loss
from llm_utils import LLMManager

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: Config, models: Tuple, graphs: Dict, llm_manager: LLMManager):
        self.config = config
        self.code_gnn, self.lab_gnn, self.image_gnn, self.imagebind = models
        self.graphs = graphs
        self.llm_manager = llm_manager
        self.scaler = GradScaler()
        self._setup_optimizer()
        
    def _setup_optimizer(self):
        """Initialize the optimizer with different learning rates for different components."""
        self.optimizer = optim.Adam([
            {'params': self.code_gnn.parameters(), 'lr': self.config.training.gnn_lr},
            {'params': self.lab_gnn.parameters(), 'lr': self.config.training.gnn_lr},
            {'params': self.image_gnn.parameters(), 'lr': self.config.training.gnn_lr},
            {'params': self.imagebind.parameters(), 'lr': self.config.training.imagebind_lr},
        ], weight_decay=self.config.model.weight_decay)
        
    def process_embeddings(self, data: Dict, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process embeddings through GNNs for a given data split.
        
        Args:
            data: Data dictionary
            split: 'train', 'test', or 'all'
            
        Returns:
            Tuple of (code_embeddings, lab_embeddings, image_embeddings)
        """
        if split == 'train':
            code_features, code_edge_index = create_graph_data(
                self.graphs['train_code'], self.config.device, include_labels=False
            )
            lab_features, lab_edge_index = create_graph_data(
                self.graphs['train_lab'], self.config.device, include_labels=False
            )
            image_features, image_edge_index = create_graph_data(
                self.graphs['train_image'], self.config.device, include_labels=False
            )
        elif split == 'test':
            code_features, code_edge_index = create_graph_data(
                self.graphs['test_code'], self.config.device, include_labels=False
            )
            lab_features, lab_edge_index = create_graph_data(
                self.graphs['test_lab'], self.config.device, include_labels=False
            )
            image_features, image_edge_index = create_graph_data(
                self.graphs['test_image'], self.config.device, include_labels=False
            )
        else:  # 'all' - concatenate train and test
            # For now, use train graphs as placeholder
            # In practice, you'd want to build combined graphs
            code_features, code_edge_index = create_graph_data(
                self.graphs['train_code'], self.config.device, include_labels=False
            )
            lab_features, lab_edge_index = create_graph_data(
                self.graphs['train_lab'], self.config.device, include_labels=False
            )
            image_features, image_edge_index = create_graph_data(
                self.graphs['train_image'], self.config.device, include_labels=False
            )
        
        # Process through GNNs
        code_embeds, _ = self.code_gnn(code_features, code_edge_index)
        lab_embeds, _ = self.lab_gnn(lab_features, lab_edge_index)
        image_embeds, _ = self.image_gnn(image_features, image_edge_index)
        
        return code_embeds, lab_embeds, image_embeds
    
    def train_epoch(self, train_data: Dict, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_data: Training data dictionary
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")
        
        # Set models to training mode
        self.code_gnn.train()
        self.lab_gnn.train()
        self.image_gnn.train()
        self.imagebind.train()
        
        # Get subject labels
        subject_last_labels = get_subject_last_labels(
            train_data['subject_id'], train_data['labels']
        )
        
        # Get unique subject IDs in order
        subject_id_ordered = list(OrderedDict.fromkeys(train_data['subject_id']))
        shuffled_indices = torch.randperm(len(subject_id_ordered))
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_texts = []
        
        for i in range(len(subject_id_ordered)):
            index = shuffled_indices[i].item()
            current_subject_id = subject_id_ordered[index]
            
            # Get all indices for this subject
            subject_indices = [
                idx for idx, subject_id in enumerate(train_data['subject_id']) 
                if subject_id == current_subject_id
            ]
            
            self.optimizer.zero_grad()
            
            # Process embeddings
            code_embeds, lab_embeds, image_embeds = self.process_embeddings(train_data, 'train')
            code_embeds = code_embeds.to(self.config.device)
            lab_embeds = lab_embeds.to(self.config.device) 
            image_embeds = image_embeds.to(self.config.device)
            
            # Align embeddings
            fused_proj, align_loss = self.imagebind(
                torch.tensor(train_data['note_embeddings'].squeeze(1)).to(self.config.device),
                code_embeds, lab_embeds, image_embeds
            )
            
            # Get context vector for this subject
            context_vector = fused_proj[subject_indices].mean(dim=0)
            
            # Concatenate notes for this subject
            subject_notes = [train_data['notes'][idx] for idx in subject_indices]
            concatenated_notes = " ".join(subject_notes)
            
            # Make prediction
            prediction, text = self.llm_manager.make_prediction(concatenated_notes, context_vector)
            ground_truth = subject_last_labels[current_subject_id]
            
            # Compute loss
            loss = compute_policy_loss(
                torch.tensor(prediction), torch.tensor(ground_truth),
                align_loss, context_vector, self.config
            )
            
            # Backpropagation
            loss = loss / self.config.training.accumulation_steps
            loss.backward()
            total_loss += loss.item() * self.config.training.accumulation_steps
            
            # Gradient clipping and optimizer step
            if ((i + 1) % self.config.training.accumulation_steps == 0 or 
                (i + 1) == len(subject_id_ordered)):
                clip_grad_norm_(
                    list(self.code_gnn.parameters()) + 
                    list(self.lab_gnn.parameters()) +
                    list(self.image_gnn.parameters()) +
                    list(self.imagebind.parameters()),
                    max_norm=self.config.training.gradient_clip_value
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Store results
            all_predictions.append(prediction)
            all_labels.append(ground_truth)
            all_texts.append(text)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss
        
        logger.info(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}, "
                   f"Accuracy: {metrics['accuracy']:.4f}, "
                   f"AUC: {metrics['auc']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def evaluate(self, test_data: Dict) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data dictionary
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        # Set models to evaluation mode
        self.code_gnn.eval()
        self.lab_gnn.eval()
        self.image_gnn.eval()
        self.imagebind.eval()
        
        # Get subject labels
        subject_last_labels = get_subject_last_labels(
            test_data['subject_id'], test_data['labels']
        )
        
        # Get unique subject IDs
        subject_id_ordered = list(OrderedDict.fromkeys(test_data['subject_id']))
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(len(subject_id_ordered)):
                current_subject_id = subject_id_ordered[i]
                
                # Get all indices for this subject
                subject_indices = [
                    idx for idx, subject_id in enumerate(test_data['subject_id']) 
                    if subject_id == current_subject_id
                ]
                
                # Process embeddings (using combined graphs for better performance)
                code_embeds, lab_embeds, image_embeds = self.process_embeddings(test_data, 'all')
                code_embeds = code_embeds.to(self.config.device)
                lab_embeds = lab_embeds.to(self.config.device)
                image_embeds = image_embeds.to(self.config.device)
                
                # For test evaluation, we need to slice appropriately
                # This is a simplified version - in practice you'd want proper test graph handling
                train_size = len(train_data['subject_id']) if 'train_data' in globals() else 0
                if train_size > 0:
                    code_embeds = code_embeds[train_size:]
                    lab_embeds = lab_embeds[train_size:]
                    image_embeds = image_embeds[train_size:]
                
                # Align embeddings
                fused_proj, _ = self.imagebind(
                    torch.tensor(test_data['note_embeddings'].squeeze(1)).to(self.config.device),
                    code_embeds, lab_embeds, image_embeds
                )
                
                # Get context vector for this subject
                context_vector = fused_proj[subject_indices].mean(dim=0)
                
                # Concatenate notes for this subject
                subject_notes = [test_data['notes'][idx] for idx in subject_indices]
                concatenated_notes = " ".join(subject_notes)
                
                # Make prediction
                prediction, _ = self.llm_manager.make_prediction(concatenated_notes, context_vector)
                ground_truth = subject_last_labels[current_subject_id]
                
                all_predictions.append(prediction)
                all_labels.append(ground_truth)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        logger.info(f"Evaluation - Accuracy: {metrics['accuracy']:.4f}, "
                   f"AUC: {metrics['auc']:.4f}, "
                   f"F1: {metrics['f1']:.4f}")
        
        return metrics, all_predictions, all_labels
    
    def _calculate_metrics(self, predictions: List[float], labels: List[int]) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: List of prediction values (0-100 scale)
            labels: List of true labels (0 or 1)
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        y_pred = np.array(predictions) / 100.0  # Normalize to 0-1
        y_true = np.array(labels)
        
        # Convert to binary predictions
        y_pred_binary = (y_pred >= self.config.data.prediction_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        auc = roc_auc_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred_binary, average='macro')
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1
        }
    
    def save_model(self, path: str):
        """Save model state dictionaries."""
        combined_state_dict = {
            'code_gnn': self.code_gnn.state_dict(),
            'lab_gnn': self.lab_gnn.state_dict(),
            'image_gnn': self.image_gnn.state_dict(),
            'imagebind': self.imagebind.state_dict()
        }
        torch.save(combined_state_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state dictionaries."""
        state_dict = torch.load(path, map_location=self.config.device)
        self.code_gnn.load_state_dict(state_dict['code_gnn'])
        self.lab_gnn.load_state_dict(state_dict['lab_gnn'])
        self.image_gnn.load_state_dict(state_dict['image_gnn'])
        self.imagebind.load_state_dict(state_dict['imagebind'])
        logger.info(f"Model loaded from {path}")


def train_model(config: Config, models: Tuple, graphs: Dict, 
                train_data: Dict, test_data: Dict, llm_manager: LLMManager) -> Trainer:
    """
    Main training function.
    
    Args:
        config: Configuration object
        models: Tuple of model components
        graphs: Graph data structures
        train_data: Training data
        test_data: Test data
        llm_manager: LLM manager instance
        
    Returns:
        Trained Trainer instance
    """
    trainer = Trainer(config, models, graphs, llm_manager)
    
    best_accuracy = 0.0
    
    for epoch in range(config.training.num_epochs):
        # Training
        train_metrics = trainer.train_epoch(train_data, epoch)
        
        # Save best model
        if train_metrics['accuracy'] > best_accuracy:
            best_accuracy = train_metrics['accuracy']
            trainer.save_model(config.data.model_save_path)
            logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    # Final evaluation
    eval_metrics, predictions, labels = trainer.evaluate(test_data)
    
    return trainer 
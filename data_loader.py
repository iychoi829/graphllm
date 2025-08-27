"""
Data loading and preprocessing utilities for multimodal medical prediction.
"""
import torch
import numpy as np
import networkx as nx
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

from config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and preprocessing of medical data."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load training and test data from files.
        
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        logger.info("Loading training and test data...")
        
        try:
            train_data_raw = torch.load(self.config.data.train_data_path)
            test_data_raw = torch.load(self.config.data.test_data_path)
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
        train_data = self._process_raw_data(train_data_raw)
        test_data = self._process_raw_data(test_data_raw)
        
        logger.info(f"Loaded {len(train_data['subject_id'])} training samples")
        logger.info(f"Loaded {len(test_data['subject_id'])} test samples")
        
        return train_data, test_data
    
    def _process_raw_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Process raw data into structured format."""
        subject_ids = np.array([entry['patient_id'] for entry in raw_data])
        code_embeddings = np.array([entry['code_embeddings'] for entry in raw_data])
        note_embeddings = np.array([entry['note_embeddings'] for entry in raw_data])
        notes = [entry.get('text', '') for entry in raw_data]
        labels = [entry.get('one_year_mortality') for entry in raw_data]
        
        # Process lab embeddings with missing value handling
        lab_embedding_shape = self._get_lab_embedding_shape(raw_data)
        lab_embeddings = self._process_lab_embeddings(raw_data, lab_embedding_shape)
        
        # Process image embeddings
        image_embeddings = self._process_image_embeddings(raw_data)
        
        return {
            'subject_id': subject_ids,
            'code_embeddings': code_embeddings,
            'lab_embeddings': lab_embeddings,
            'image_embeddings': image_embeddings,
            'note_embeddings': note_embeddings,
            'notes': notes,
            'labels': labels
        }
    
    def _get_lab_embedding_shape(self, raw_data: List[Dict]) -> Tuple:
        """Get the shape of lab embeddings from the first valid entry."""
        for entry in raw_data:
            if entry['labs'] is not None:
                return np.shape(entry['labs'])
        raise ValueError("No valid lab embeddings found in data")
    
    def _process_lab_embeddings(self, raw_data: List[Dict], shape: Tuple) -> np.ndarray:
        """Process lab embeddings, filling missing values with zeros."""
        lab_embeddings = []
        for entry in raw_data:
            if entry['labs'] is not None:
                labs = np.expand_dims(np.nan_to_num(entry['labs']).astype(np.float32), axis=0)
            else:
                labs = np.expand_dims(np.zeros(shape, dtype=np.float32), axis=0)
            lab_embeddings.append(labs)
        return np.array(lab_embeddings)
    
    def _process_image_embeddings(self, raw_data: List[Dict]) -> np.ndarray:
        """Process image embeddings, averaging multiple embeddings per entry."""
        image_embeddings = []
        for entry in raw_data:
            if entry['image_embeddings']:
                # Average multiple image embeddings
                avg_embedding = torch.mean(torch.stack(entry['image_embeddings']), dim=0).numpy()
                image_embeddings.append(np.expand_dims(avg_embedding, axis=0))
            else:
                image_embeddings.append(np.expand_dims(np.zeros((1,)), axis=0))
        return np.array(image_embeddings)


class GraphBuilder:
    """Builds graphs for different modalities."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_graphs(self) -> Dict[str, nx.Graph]:
        """Load or build graphs for training and testing."""
        logger.info("Building graphs for different modalities...")
        
        # Load data for graph construction
        train_data_raw = torch.load(self.config.data.train_data_path)
        test_data_raw = torch.load(self.config.data.test_data_path)
        
        # Process data for graph building
        train_subject_ids, train_embeddings = self._extract_graph_data(train_data_raw)
        test_subject_ids, test_embeddings = self._extract_graph_data(test_data_raw)
        
        # Build graphs
        graphs = {}
        
        # Training graphs
        train_code_graph, train_lab_graph, train_image_graph = self._build_modality_graphs(
            train_subject_ids, train_embeddings
        )
        
        # Test graphs
        test_code_graph, test_lab_graph, test_image_graph = self._build_modality_graphs(
            test_subject_ids, test_embeddings
        )
        
        graphs.update({
            'train_code': train_code_graph,
            'train_lab': train_lab_graph,
            'train_image': train_image_graph,
            'test_code': test_code_graph,
            'test_lab': test_lab_graph,
            'test_image': test_image_graph
        })
        
        return graphs
    
    def _extract_graph_data(self, raw_data: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Extract data needed for graph construction."""
        subject_ids = np.array([entry['patient_id'] for entry in raw_data])
        code_embeddings = np.array([entry['code_embeddings'] for entry in raw_data])
        
        # Process lab embeddings
        lab_embedding_shape = None
        for entry in raw_data:
            if entry['labs'] is not None:
                lab_embedding_shape = np.shape(entry['labs'])
                break
        
        lab_embeddings = np.array([
            np.expand_dims(np.nan_to_num(entry['labs']), axis=0) 
            if entry['labs'] is not None 
            else np.expand_dims(np.zeros(lab_embedding_shape), axis=0)
            for entry in raw_data
        ])
        
        # Process image embeddings
        image_embeddings = np.array([
            np.expand_dims(torch.mean(torch.stack(entry['image_embeddings']), dim=0).numpy(), axis=0) 
            if entry['image_embeddings'] 
            else np.expand_dims(np.zeros((1,)), axis=0)
            for entry in raw_data
        ])
        
        embeddings = {
            'code': code_embeddings,
            'lab': lab_embeddings,
            'image': image_embeddings
        }
        
        return subject_ids, embeddings
    
    def _build_modality_graphs(self, subject_ids: np.ndarray, embeddings: Dict[str, np.ndarray]) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
        """Build graphs for all modalities."""
        code_graph = nx.Graph()
        lab_graph = nx.Graph()
        image_graph = nx.Graph()
        
        self._build_graphs(
            subject_ids, code_graph, lab_graph, image_graph,
            embeddings['code'], embeddings['lab'], embeddings['image']
        )
        
        return code_graph, lab_graph, image_graph
    
    def _build_graphs(self, subject_ids: np.ndarray, code_graph: nx.Graph, 
                     lab_graph: nx.Graph, image_graph: nx.Graph,
                     code_embeddings: np.ndarray, lab_embeddings: np.ndarray, 
                     image_embeddings: np.ndarray) -> None:
        """Build graphs with temporal and similarity edges."""
        
        # Add nodes
        for i, subject_id in enumerate(subject_ids):
            code_graph.add_node(i, subject_id=subject_id, embedding=code_embeddings[i][0])
            lab_graph.add_node(i, subject_id=subject_id, embedding=lab_embeddings[i][0])
            image_graph.add_node(i, subject_id=subject_id, embedding=image_embeddings[i][0])
        
        # Add temporal edges
        temporal_edges = self._add_temporal_edges(subject_ids, code_graph, lab_graph, image_graph)
        
        # Add similarity edges
        similarity_edges = self._add_similarity_edges(
            subject_ids, code_graph, lab_graph, image_graph,
            code_embeddings, lab_embeddings, image_embeddings
        )
        
        logger.info(f"Built graphs with {len(subject_ids)} nodes, "
                   f"{temporal_edges} temporal edges, {sum(similarity_edges.values())} similarity edges")
    
    def _add_temporal_edges(self, subject_ids: np.ndarray, *graphs: nx.Graph) -> int:
        """Add temporal edges for the same subject across time."""
        temporal_edge_count = 0
        
        for subject_id in np.unique(subject_ids):
            indices = np.where(subject_ids == subject_id)[0]
            for i in range(len(indices) - 1):
                for graph in graphs:
                    graph.add_edge(indices[i], indices[i + 1], edge_type='temporal')
                temporal_edge_count += 1
        
        return temporal_edge_count
    
    def _add_similarity_edges(self, subject_ids: np.ndarray, 
                            code_graph: nx.Graph, lab_graph: nx.Graph, image_graph: nx.Graph,
                            code_embeddings: np.ndarray, lab_embeddings: np.ndarray, 
                            image_embeddings: np.ndarray) -> Dict[str, int]:
        """Add similarity-based edges."""
        
        # Flatten embeddings for similarity computation
        code_flat = np.squeeze(code_embeddings, axis=1)
        lab_flat = np.squeeze(lab_embeddings, axis=1)
        image_flat = np.squeeze(image_embeddings, axis=1)
        
        # Compute cosine similarities
        code_sim = cosine_similarity(code_flat)
        lab_sim = cosine_similarity(lab_flat)
        image_sim = cosine_similarity(image_flat)
        
        similarity_counts = {'code': 0, 'lab': 0, 'image': 0}
        k = self.config.data.similarity_k
        threshold = self.config.data.similarity_threshold
        
        for i in range(len(subject_ids)):
            # Find top-k similar nodes for each modality
            top_k_code = np.argsort(-code_sim[i, :])[:k+1]
            top_k_lab = np.argsort(-lab_sim[i, :])[:k+1]
            top_k_image = np.argsort(-image_sim[i, :])[:k+1]
            
            # Add edges based on similarity threshold
            for j in top_k_code:
                if (i != j and subject_ids[i] != subject_ids[j] and 
                    code_sim[i, j] > threshold):
                    code_graph.add_edge(i, j, edge_type='similarity', weight=code_sim[i, j])
                    similarity_counts['code'] += 1
            
            for j in top_k_lab:
                if (i != j and subject_ids[i] != subject_ids[j] and 
                    lab_sim[i, j] > threshold):
                    lab_graph.add_edge(i, j, edge_type='similarity', weight=lab_sim[i, j])
                    similarity_counts['lab'] += 1
            
            for j in top_k_image:
                if (i != j and subject_ids[i] != subject_ids[j] and 
                    image_sim[i, j] > threshold):
                    image_graph.add_edge(i, j, edge_type='similarity', weight=image_sim[i, j])
                    similarity_counts['image'] += 1
        
        return similarity_counts


def create_graph_data(graph: nx.Graph, device: torch.device, 
                     include_labels: bool = True) -> Tuple[torch.Tensor, ...]:
    """Convert NetworkX graph to PyTorch tensors."""
    node_features = torch.stack([
        torch.tensor(graph.nodes[n]['embedding'], requires_grad=True) 
        for n in graph.nodes()
    ]).to(device)
    
    edge_index = torch.tensor(
        list(graph.edges()), requires_grad=False
    ).t().contiguous().to(device)
    
    if include_labels:
        labels = torch.tensor([
            graph.nodes[n]['label'] for n in graph.nodes()
        ], requires_grad=False).to(device)
        return node_features, edge_index, labels
    
    return node_features, edge_index


def get_subject_last_labels(subject_ids: np.ndarray, labels: List) -> Dict[str, Any]:
    """Get the last label for each unique subject ID."""
    subject_id_ordered = list(OrderedDict.fromkeys(subject_ids))
    subject_last_label = {}
    
    for subject_id in subject_id_ordered:
        indices = [i for i, sid in enumerate(subject_ids) if sid == subject_id]
        last_label = labels[indices[-1]]
        subject_last_label[subject_id] = last_label
    
    return subject_last_label 
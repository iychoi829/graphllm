import pandas as pd
import torch
import numpy as np
import networkx as nx
import torch.optim as optim
import pickle
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from collections import OrderedDict
from scipy.special import xlogy

# Setup logging
logger = logging.getLogger(__name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    logger.info("GPU is available!")
    logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    logger.info("GPU is not available.")

ratio_a = nn.Parameter(torch.tensor(0.25, device=device))  # Coefficient for  context vector
ratio_b = nn.Parameter(torch.tensor(1.0, device=device)) 
from huggingface_hub import login
access_token = 'TOKEN'
login(token=access_token)

# Use the token to load the model
tokenizer = AutoTokenizer.from_pretrained( "meta-llama/Meta-Llama-3-8B", use_auth_token=access_token)
llm = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=access_token).to(device)

logger.info('Tokenizer and LLM loaded successfully')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if llm.config.pad_token_id is None:
    llm.config.pad_token_id = llm.config.eos_token_id

import re
def extract_numeric_value(prediction):
    # Remove any leading/trailing whitespace
    prediction = prediction.strip()

    # Check if 'ANSWER:' is in the prediction
    if 'ANSWER:' in prediction:
        # Extract everything after 'ANSWER:'
        answer_part = prediction.split('ANSWER:')[-1].strip()
    else:
        # If 'ANSWER:' is not found, use the entire prediction
        answer_part = prediction

    # 1. Look for a number followed by a percentage sign
    match = re.search(r'(\d+)%', answer_part)
    if match:
        return float(match.group(1))  # Return the number before '%'

    # 2. Try to find a numeric value at the end of the answer part
    match = re.search(r'\d+(?:\.\d+)?$', answer_part)
    if match:
        return float(match.group())

    # 3. If no numeric value is found at the end, search for any numeric value
    match = re.search(r'\d+(?:\.\d+)?', answer_part)
    if match:
        return float(match.group())

    # 4. If no numeric value is found, return a default value
    logger.warning('No numeric value found in prediction')
    return -1.0




class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.prediction_head = nn.Linear(out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        embeddings = F.relu(self.conv3(x, edge_index))
        predictions = self.prediction_head(embeddings).squeeze(-1)
        return torch.nan_to_num(embeddings), torch.nan_to_num(predictions)





def load_data():

    train_data = torch.load('/cbica/home/NAME/project/downsampled_data/train_data_20.pt')
    test_data = torch.load('/cbica/home/NAME/project/downsampled_data/test_data_20.pt')
    train_subject_id = np.array([entry['patient_id'] for entry in train_data])
    train_code_embeddings = np.array([entry['code_embeddings'] for entry in train_data])
    for entry in train_data:
        if entry['labs'] is not None:
            lab_embedding_shape = np.shape(entry['labs'])  # Get the shape of a valid 'labs' entry
            break

    # Create lab_embeddings array, replacing missing values with a zero vector of the same shape
    train_lab_embeddings = np.array([
        np.expand_dims(np.nan_to_num(entry['labs']).astype(np.float32), axis=0) if entry['labs'] is not None else np.expand_dims(np.zeros(lab_embedding_shape, dtype=np.float32), axis=0)
        for entry in train_data
    ])

    # Create image_embeddings array, averaging the list of tensors in each entry
    train_image_embeddings = np.array([
        np.expand_dims(torch.mean(torch.stack(entry['image_embeddings']), dim=0).numpy(), axis=0) if entry['image_embeddings'] else np.expand_dims(np.zeros((1,)), axis=0)
        for entry in train_data
    ])
    train_note_embeddings = np.array([entry['note_embeddings'] for entry in train_data])
    train_notes = [entry.get('text', '') for entry in train_data]
    train_labels = [entry.get('one_year_mortality') for entry in train_data]

    

    test_subject_id = np.array([entry['patient_id'] for entry in test_data])
    test_code_embeddings = np.array([entry['code_embeddings'] for entry in test_data])
    # train_code_embeddings = np.array([entry['note_embeddings'] for entry in train_data])
    # lab_embeddings = np.array([entry['labs'] for entry in train_data])
    for entry in test_data:
        if entry['labs'] is not None:
            lab_embedding_shape = np.shape(entry['labs'])  # Get the shape of a valid 'labs' entry
            break

    # Create lab_embeddings array, replacing missing values with a zero vector of the same shape
    test_lab_embeddings = np.array([
        np.expand_dims(np.nan_to_num(entry['labs']).astype(np.float32), axis=0) if entry['labs'] is not None else np.expand_dims(np.zeros(lab_embedding_shape, dtype=np.float32), axis=0)
        for entry in test_data
    ])
    # Create image_embeddings array, averaging the list of tensors in each entry
    test_image_embeddings = np.array([
        np.expand_dims(torch.mean(torch.stack(entry['image_embeddings']), dim=0).numpy(), axis=0) if entry['image_embeddings'] else np.expand_dims(np.zeros((1,)), axis=0)
        for entry in test_data
    ])
    test_note_embeddings = np.array([entry['note_embeddings'] for entry in test_data])
    test_notes = [entry.get('text', '') for entry in test_data]
    test_labels = [entry.get('one_year_mortality') for entry in test_data]

    train_data = {
        'subject_id': train_subject_id,
        'code_embeddings': train_code_embeddings,
        'lab_embeddings': train_lab_embeddings,
        'image_embeddings': train_image_embeddings,
        'note_embeddings': train_note_embeddings,
        'notes': train_notes,
        'labels': train_labels
    }

    test_data = {
        'subject_id': test_subject_id,
        'code_embeddings': test_code_embeddings,
        'lab_embeddings': test_lab_embeddings,
        'image_embeddings':test_image_embeddings,
        'note_embeddings': test_note_embeddings,
        'notes': test_notes,
        'labels':test_labels
    }

    return train_data, test_data


# def load_graphs():
#     graphs = {}
#     for data_type in ['code', 'lab', 'image']:
#    #     for split in ['train', 'test']:
#         for split in ['train', 'test', 'all']:
#             with open(f'./{split}_{data_type}_graph.pickle', 'rb') as f:
#                 graphs[f'{split}_{data_type}'] = pickle.load(f)
#     return graphs
def build_graphs(subject_ids, code_graph, lab_graph, image_graph, code_embeddings, lab_embeddings, image_embeddings, k=100, similarity_threshold=0.99):
    temporal_edge_count = 0
    similarity_edge_count_code = 0
    similarity_edge_count_lab = 0
    similarity_edge_count_image = 0

    for i, subject_id in enumerate(subject_ids):
        code_graph.add_node(i, subject_id=subject_id, embedding=code_embeddings[i][0])
        lab_graph.add_node(i, subject_id=subject_id, embedding=lab_embeddings[i][0])
        image_graph.add_node(i, subject_id=subject_id, embedding=image_embeddings[i][0])

    # Adding temporal edges for the same subject_id
    for subject_id in np.unique(subject_ids):
        indices = np.where(subject_ids == subject_id)[0]
        for i in range(len(indices) - 1):
            code_graph.add_edge(indices[i], indices[i + 1], edge_type='temporal')
            lab_graph.add_edge(indices[i], indices[i + 1], edge_type='temporal')
            image_graph.add_edge(indices[i], indices[i + 1], edge_type='temporal')
            temporal_edge_count += 1

    logger.info('Temporal edges construction completed')

    code_embeddings_flat = np.squeeze(code_embeddings, axis=1)
    lab_embeddings_flat = np.squeeze(lab_embeddings, axis=1)
    image_embeddings_flat = np.squeeze(image_embeddings, axis=1)

    code_cos_sim = cosine_similarity(code_embeddings_flat)
    lab_cos_sim = cosine_similarity(lab_embeddings_flat)
    image_cos_sim = cosine_similarity(image_embeddings_flat)

    for i in range(len(subject_ids)):
        top_k_code_indices = np.argsort(-code_cos_sim[i, :])[:k+1]  # +1 because the node itself is the most similar
        top_k_lab_indices = np.argsort(-lab_cos_sim[i, :])[:k+1]
        top_k_image_indices = np.argsort(-image_cos_sim[i, :])[:k+1]

        for j in top_k_code_indices:
            if i != j and subject_ids[i] != subject_ids[j] and code_cos_sim[i, j] > similarity_threshold:
                code_graph.add_edge(i, j, edge_type='similarity', weight=code_cos_sim[i, j])
                similarity_edge_count_code += 1

        for j in top_k_lab_indices:
            if i != j and subject_ids[i] != subject_ids[j] and lab_cos_sim[i, j] > similarity_threshold:
                lab_graph.add_edge(i, j, edge_type='similarity', weight=lab_cos_sim[i, j])
                similarity_edge_count_lab += 1

        for j in top_k_image_indices:
            if i != j and subject_ids[i] != subject_ids[j] and image_cos_sim[i, j] > similarity_threshold:
                image_graph.add_edge(i, j, edge_type='similarity', weight=image_cos_sim[i, j])
                similarity_edge_count_image += 1

    # Log summary of the graph
    logger.info(f"Code Graph: Nodes={code_graph.number_of_nodes()}, Temporal Edges={temporal_edge_count}, Similarity Edges={similarity_edge_count_code}")
    logger.info(f"Lab Graph: Nodes={lab_graph.number_of_nodes()}, Temporal Edges={temporal_edge_count}, Similarity Edges={similarity_edge_count_lab}")
    logger.info(f"Image Graph: Nodes={image_graph.number_of_nodes()}, Temporal Edges={temporal_edge_count}, Similarity Edges={similarity_edge_count_image}")


def load_graphs():
    graph = {}
    train_data =  torch.load('/cbica/home/NAME/project/downsampled_data/train_data_20.pt')
    train_subject_id = np.array([entry['patient_id'] for entry in train_data])
    train_code_embeddings = np.array([entry['code_embeddings'] for entry in train_data])
    # train_code_embeddings = np.array([entry['note_embeddings'] for entry in train_data])
    # lab_embeddings = np.array([entry['labs'] for entry in train_data])
    for entry in train_data:
        if entry['labs'] is not None:
            lab_embedding_shape = np.shape(entry['labs'])  # Get the shape of a valid 'labs' entry
            break

    # Create lab_embeddings array, replacing missing values with a zero vector of the same shape
    train_lab_embeddings = np.array([
        np.expand_dims(np.nan_to_num(entry['labs']), axis=0) if entry['labs'] is not None else np.expand_dims(np.zeros(lab_embedding_shape), axis=0)
        for entry in train_data
    ])

    # Create image_embeddings array, averaging the list of tensors in each entry
    train_image_embeddings = np.array([
        np.expand_dims(torch.mean(torch.stack(entry['image_embeddings']), dim=0).numpy(), axis=0) if entry['image_embeddings'] else np.expand_dims(np.zeros((1,)), axis=0)
        for entry in train_data
    ])


    test_data =  torch.load('/cbica/home/NAME/project/downsampled_data/test_data_20.pt')
    test_subject_id = np.array([entry['patient_id'] for entry in test_data])
    test_code_embeddings = np.array([entry['code_embeddings'] for entry in test_data])
    # train_code_embeddings = np.array([entry['note_embeddings'] for entry in train_data])
    # lab_embeddings = np.array([entry['labs'] for entry in train_data])
    for entry in test_data:
        if entry['labs'] is not None:
            lab_embedding_shape = np.shape(entry['labs'])  # Get the shape of a valid 'labs' entry
            break

    # Create lab_embeddings array, replacing missing values with a zero vector of the same shape
    test_lab_embeddings = np.array([
        np.expand_dims(np.nan_to_num(entry['labs']), axis=0) if entry['labs'] is not None else np.expand_dims(np.zeros(lab_embedding_shape), axis=0)
        for entry in test_data
    ])
    # Create image_embeddings array, averaging the list of tensors in each entry
    test_image_embeddings = np.array([
        np.expand_dims(torch.mean(torch.stack(entry['image_embeddings']), dim=0).numpy(), axis=0) if entry['image_embeddings'] else np.expand_dims(np.zeros((1,)), axis=0)
        for entry in test_data
    ])
    train_code_graph = nx.Graph()
    train_lab_graph = nx.Graph()
    train_image_graph = nx.Graph()
    build_graphs(train_subject_id, train_code_graph, train_lab_graph, train_image_graph, train_code_embeddings, train_lab_embeddings, train_image_embeddings, k=1000, similarity_threshold=0.7)

    test_code_graph = nx.Graph()
    test_lab_graph = nx.Graph()
    test_image_graph =nx.Graph()
    build_graphs(test_subject_id, test_code_graph, test_lab_graph, test_image_graph, test_code_embeddings, test_lab_embeddings, test_image_embeddings, k=1000, similarity_threshold=0.7)

    graph['train_code'] = train_code_graph
    graph['train_lab'] = train_lab_graph
    graph['train_image'] = train_image_graph
    graph['test_code'] = test_code_graph
    graph['test_lab'] = test_lab_graph
    graph['test_image'] = test_image_graph

    return graph




def create_graph_data(graph, device, include_labels=True):
    node_features = torch.stack([torch.tensor(graph.nodes[n]['embedding'], requires_grad=True) for n in graph.nodes()]).to(device)
    edge_index = torch.tensor(list(graph.edges()), requires_grad=False).t().contiguous().to(device)
    if include_labels:
        labels = torch.tensor([graph.nodes[n]['label'] for n in graph.nodes()], requires_grad=False).to(device)
        return node_features, edge_index, labels
    return node_features, edge_index

def process_embeddings(models, graphs, data, train_test):
    code_gnn, lab_gnn, image_gnn, _ = models
    if train_test == 'train':
        code_features, code_edge_index = create_graph_data(graphs['train_code'], device, include_labels=False)
        lab_features, lab_edge_index = create_graph_data(graphs['train_lab'], device, include_labels=False)
        image_features, image_edge_index = create_graph_data(graphs['train_image'], device, include_labels=False)
    elif train_test == 'test':
        code_features, code_edge_index = create_graph_data(graphs['test_code'], device, include_labels=False)
        lab_features, lab_edge_index = create_graph_data(graphs['test_lab'], device, include_labels=False)
        image_features, image_edge_index = create_graph_data(graphs['test_image'], device, include_labels=False)
    else: 
        code_features, code_edge_index = create_graph_data(graphs['all_code'], device, include_labels=False)
        lab_features, lab_edge_index = create_graph_data(graphs['all_lab'], device, include_labels=False)
        image_features, image_edge_index = create_graph_data(graphs['all_image'], device, include_labels=False)
    code_embeds, _ = code_gnn(code_features, code_edge_index)
    lab_embeds, _ = lab_gnn(lab_features, lab_edge_index)
    image_embeds, _ = image_gnn(image_features, image_edge_index)
    # Return processed embeddings
    return code_embeds, lab_embeds, image_embeds


def align_embeddings(imagebind, data, code_embeds, lab_embeds, image_embeds):

    aligned_code, code_loss = imagebind(data['note_embeddings'].unsqueeze(0), code_embeds, modality='code')
    aligned_lab, lab_loss = imagebind(data['note_embeddings'].unsqueeze(0), lab_embeds, modality='lab')
    aligned_image, image_loss = imagebind(data['note_embeddings'].unsqueeze(0), image_embeds, modality='image')

    aligned_embeddings = torch.stack([aligned_code, aligned_lab, aligned_image], dim=0)

    total_loss = code_loss + lab_loss + image_loss


    return aligned_embeddings, total_loss

def prepare_prompt(notes, context_vector):
    # Get the clinical notes for the target_subject_id
    notes = notes.replace('\n', '')


    instruction = (
        "You are a medical expert. Here are the clinical notes of a patient across multiple visits: "
    )

    question = (
        "\n\nBased on these notes, think step-by-step and assess this patient's probability of experiencing 1-year mortality. "
        "Start by analyzing the patient's medical history, current condition, and relevant lab results. "
        "Then, provide a final mortality risk prediction as a percentage on a scale from 0 to 100, where 0 means no chance of mortality and 100 means certain death.\n\nANSWER: "
    )
    # Tokenize the instruction and question separately (fixed parts)
    instruction_tokens = tokenizer(instruction, return_tensors="pt", truncation=False).to(device)
    question_tokens = tokenizer(question, return_tensors="pt", truncation=False).to(device)
    notes_tokens = tokenizer(notes, return_tensors="pt", truncation=True, max_length = 4096 - instruction_tokens.input_ids.size(1) - question_tokens.input_ids.size(1)).to(device)

    # Concatenate the instruction, notes, and question
    input_ids = torch.cat([instruction_tokens.input_ids, notes_tokens.input_ids, question_tokens.input_ids], dim=-1)
    attention_mask = torch.cat([instruction_tokens.attention_mask, notes_tokens.attention_mask, question_tokens.attention_mask], dim=-1)

    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    # Return inputs and the context vector
    return inputs, context_vector



def make_prediction(notes, context_vector):
    # Prepare the prompt and context vector
    notes = notes.replace('\n', '')


    instruction = (
        "You are a medical expert. Here are the clinical notes of a patient across multiple visits: "
    )

    question = (
        "\n\nBased on these notes, think step-by-step and assess this patient's probability of experiencing 1-year mortality. "
        "Start by analyzing the patient's medical history, current condition, and relevant lab results. "
        "Then, provide a final mortality risk prediction as a percentage on a scale from 0 to 100, where 0 means no chance of mortality and 100 means certain death.\n\nANSWER: "
    )

    # instruction = (
    # "You are a medical expert. Based solely on the information provided to you, assess this patient's probability of experiencing 1-year mortality. Avoid making assumptions or providing additional information beyond the context given."
    #  "Provide a final mortality risk prediction as a percentage on a scale from 0 to 100, where 0 means no chance of mortality and 100 means certain death."

    # )

    # question = (
    #     "\n\n Answer: "
    # )
    # Tokenize the instruction and question separately (fixed parts)
    instruction_tokens = tokenizer(instruction, return_tensors="pt", truncation=False).to(device)
    question_tokens = tokenizer(question, return_tensors="pt", truncation=False).to(device)
    notes_tokens = tokenizer(notes, return_tensors="pt", truncation=True, max_length = 4096 - instruction_tokens.input_ids.size(1) - question_tokens.input_ids.size(1)).to(device)

    # Concatenate the instruction, notes, and question
    input_ids = torch.cat([instruction_tokens.input_ids, notes_tokens.input_ids, question_tokens.input_ids], dim=-1)
    attention_mask = torch.cat([instruction_tokens.attention_mask, notes_tokens.attention_mask, question_tokens.attention_mask], dim=-1)

    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    # inputs, context_vector = prepare_prompt(notes, context_vector)
    # context_vector = injected_llm(context_vector)

    if torch.isnan(context_vector).any() or torch.isinf(context_vector).any():
        logger.warning("Context vector contains NaN or Inf values!")

    input_ids = inputs['input_ids']

    # Ensure input is valid
    if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
        logger.warning("Input IDs contain NaN or Inf values!")
    if torch.isnan(inputs['attention_mask']).any() or torch.isinf(inputs['attention_mask']).any():
        logger.warning("Attention mask contains NaN or Inf values!")

    def hook(module, input, output):
        # Check if the module is MLP by looking for its characteristic submodules

        # ratio_a = nn.Parameter(torch.tensor(0.5, device=device))  # Coefficient for  context vector
        # ratio_b = nn.Parameter(torch.tensor(1.0, device=device))
        if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
            context_vector_local = context_vector.to(output.device)
            
            # Modify the output (which is a single tensor, not a tuple)
            modified_output = ratio_a * context_vector_local +ratio_b * output
            return modified_output  # Return modified tensor
        
        else:
            # For other layers, you may just return the original output
            return output

    # Register hooks for LLM layers
    handles = []
    for layer in llm.model.layers:
        handles.append(layer.mlp.register_forward_hook(hook))
        handles.append(layer.self_attn.register_forward_hook(hook))

    try:
        # Instead of using generate(), call the model's forward method
        input_ids = input_ids.to(llm.device)
        output = llm.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=inputs['attention_mask'].to(llm.device),
                max_new_tokens=10,  # Number of tokens to generate
                temperature=0.3,     # Control creativity
                top_p=0.9,           # Control diversity
                do_sample=True       # Enable sampling
            )

        
        # prediction=(classification_llm(input_ids, inputs['attention_mask'].to(llm.device)))
    finally:
        # Remove hooks after the forward pass
        for handle in handles:
            handle.remove()
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)

    # # Extract the numeric value from the prediction
    prediction_value = extract_numeric_value(prediction)
    return prediction_value, prediction

from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttentionLayer, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5  # Scaling factor for attention scores

    def forward(self, x):
        # x: (n_datapoints, 4, embed_dim)
        
        # Linear projections for query, key, and value
        q = self.query(x)  # (n_datapoints, 4, embed_dim)
        k = self.key(x)    # (n_datapoints, 4, embed_dim)
        v = self.value(x)  # (n_datapoints, 4, embed_dim)
        
        # Calculate attention scores: (n_datapoints, 4, 4)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights: (n_datapoints, 4, 4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Calculate weighted sum of values: (n_datapoints, 4, embed_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Aggregate the attention outputs from all modalities
        # Summing over the modality dimension (4) to get the final fused representation
        fused_proj = attn_output.sum(dim=1)  # (n_datapoints, embed_dim)
        
        return fused_proj

def min_max_normalize(embeds):
    min_val = embeds.min(dim=1, keepdim=True)[0]  # Minimum value along rows
    max_val = embeds.max(dim=1, keepdim=True)[0]  # Maximum value along rows
    normalized_embeds = (embeds - min_val) / (max_val - min_val + 1e-8)  # Add a small epsilon to prevent division by zero
    return normalized_embeds

class ImageBindAlignment(nn.Module):
    def __init__(self, note_dim, code_dim, lab_dim, image_dim, common_dim, temperature):
        super(ImageBindAlignment, self).__init__()
        self.note_proj = nn.Linear(note_dim, common_dim)
        self.code_proj = nn.Linear(code_dim, common_dim)
        self.lab_proj = nn.Linear(lab_dim, common_dim)
        self.image_proj = nn.Linear(image_dim, common_dim)
        self.temperature = temperature
        # self.attention_layer = SelfAttentionLayer(common_dim)
        # self.reward_weight = nn.Parameter(torch.tensor(0.9), requires_grad = True)  # Initialize to 0.8
        # # self.focal_weight = nn.Parameter(torch.tensor(0.0001), requires_grad = True)    # Initialize to 0.1
        # self.align_weight = nn.Parameter(torch.tensor(0.1), requires_grad = True)    # Initialize to 0.2
        # self.ratio_weight = nn.Parameter(torch.tensor(0.001), requires_grad = True)
    def forward(self, note_embeddings, code_embeddings, lab_embeddings, image_embeddings):
        # Project all embeddings to the common dimension
        note_proj = self.note_proj(note_embeddings)
        code_proj = self.code_proj(code_embeddings)
        lab_proj = self.lab_proj(lab_embeddings)
        image_proj = self.image_proj(image_embeddings)

        # # # Normalize all projections
        # note_proj = F.normalize(note_proj, p=2, dim=-1, eps=1e-8)
        note_proj = min_max_normalize(note_proj)
        code_proj = min_max_normalize(code_proj)
        lab_proj = min_max_normalize(lab_proj)
        image_proj = min_max_normalize(image_proj)
        # code_proj = F.normalize(code_proj, p=2, dim=-1, eps=1e-8)
        # lab_proj = F.normalize(lab_proj, p=2, dim=-1, eps=1e-8)
        # image_proj = F.normalize(image_proj, p=2, dim=-1, eps=1e-8)

        # Calculate similarities between modalities (both directions for symmetry)
        sim_note_code = torch.matmul(note_proj, code_proj.T) / self.temperature
        sim_code_note = torch.matmul(code_proj, note_proj.T) / self.temperature

        sim_note_lab = torch.matmul(note_proj, lab_proj.T) / self.temperature
        sim_lab_note = torch.matmul(lab_proj, note_proj.T) / self.temperature

        sim_note_image = torch.matmul(note_proj, image_proj.T) / self.temperature
        sim_image_note = torch.matmul(image_proj, note_proj.T) / self.temperature

        batch_size = note_proj.size(0)
        labels = torch.arange(batch_size, device=note_proj.device)
        # InfoNCE loss computation (both directions)
        loss_note_code = self.info_nce_loss(sim_note_code, labels) + self.info_nce_loss(sim_code_note, labels)
        loss_note_lab = self.info_nce_loss(sim_note_lab, labels) + self.info_nce_loss(sim_lab_note, labels)
        loss_note_image = self.info_nce_loss(sim_note_image, labels) + self.info_nce_loss(sim_image_note, labels)


        total_loss = loss_note_code + loss_note_lab + loss_note_image

        # return note_proj, code_proj, lab_proj, image_proj, total_loss
        fused_proj = torch.stack([note_proj, code_proj, lab_proj, image_proj], dim=1)  # Shape: (n_datapoints, 4, 4096)
        # fused_proj = self.attention_layer(fused_proj) 
        fused_proj = fused_proj.sum(dim=1)
        return fused_proj, total_loss

    def info_nce_loss(self, similarity, labels):
        log_prob = F.log_softmax(similarity, dim=-1)
        loss = F.nll_loss(log_prob, labels)
        return loss




def evaluate_model(models, test_data, graphs):
    code_gnn, lab_gnn, image_gnn, imagebind = models
    test_predictions = {}
    test_true_labels = {}
    
    with torch.no_grad():
        for index in range(len(test_data['notes'])):
            code_embeds, lab_embeds, image_embeds = process_embeddings(models, graphs, test_data, index)
            aligned_embeddings, _ = align_embeddings(imagebind, test_data, code_embeds, lab_embeds, image_embeds, index)
            prediction = make_prediction(test_data['notes'][index], aligned_embeddings)
            
            subject_id = test_data['subject_id'][index]
            test_predictions[subject_id] = prediction
            test_true_labels[subject_id] = test_data['labels'][index]
    
    return test_predictions, test_true_labels


def calculate_metrics(predictions, true_labels):
    """
    Calculate accuracy, AUC, and F1 score.
    """
    # Extract just the predictions (first element in the [prediction, text] list)
    y_pred = np.array([pred[0] for pred in predictions.values()])
    y_pred = y_pred / 100.0
    y_true = np.array([label for label in true_labels.values()])
    
    # Convert predictions to binary labels (thresholding for classification)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Calculate accuracy, AUC, and F1 score
    accuracy = accuracy_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary)
    
    return accuracy, auc, f1

class CustomLoss(nn.Module):
    def __init__(self, weight_valid=1.0, weight_accuracy=1.0, weight_0=1, weight_1=3):
        super(CustomLoss, self).__init__()
        self.weight_valid = weight_valid
        self.weight_accuracy = weight_accuracy
        self.weight_0 = weight_0  # Weight for class 0
        self.weight_1 = weight_1  # Weight for class 1

    def forward(self, prediction, target):
        # Ensure prediction and target are tensors with batch dimension
        if prediction.dim() == 0:  # Check if prediction is scalar
            prediction = prediction.unsqueeze(0)  # Add batch dimension [1]
        if target.dim() == 0:  # Check if target is scalar
            target = target.unsqueeze(0)  # Add batch dimension [1]

        # Validity Loss: Penalize predictions outside [0, 100]
        lower_penalty = torch.relu(-prediction)  # Penalize predictions < 0
        upper_penalty = torch.relu(prediction - 100)  # Penalize predictions > 100
        validity_loss = torch.mean(lower_penalty + upper_penalty)  # Sum penalties

        # Scale prediction to logits in [0, 1] range
        predicted_prob = prediction / 100
        predicted_prob = torch.clamp(predicted_prob, min=1e-7, max=1-1e-7)  # Avoid log(0)

        # Create a weight tensor based on target values
        weights = torch.where(target == 1, torch.tensor(self.weight_1), torch.tensor(self.weight_0))
        
        # Calculate BCE loss with per-sample weights
        bce_loss_fn = nn.BCELoss(weight=weights, reduction='mean')
        bce_loss = bce_loss_fn(predicted_prob, target)

        # Combine the two loss terms
        total_loss = self.weight_valid * validity_loss + self.weight_accuracy * bce_loss

        return total_loss

def compute_policy_loss(prediction, ground_truth, align_loss, context_vector, model, alpha=0.9, beta=0.0001, eps=1e-7):
    # reward_module = DifferentiableReward()

    # # Calculate main reward
    # reward = reward_module(prediction, ground_truth) if prediction != -1.0 else torch.tensor(0.0, device=prediction.device)
    # reward_loss = (1 - reward).mean()
    custom_loss = CustomLoss(weight_valid=0.2, weight_accuracy=1.0)
    prediction = prediction.float()
    ground_truth = ground_truth.float()
  # Compute the loss
    reward_loss = custom_loss(prediction, ground_truth)
    # Scale losses
    align_loss = align_loss / 100000
    context_loss = torch.log(1 + context_vector.norm() + eps)

    reward_weight = 1 # Primary objective
    align_weight = 0.01  # Secondary objective
    ratio_weight = 0.01  # Secondary objective

    # Combine the losses
    combined_loss = (reward_weight * reward_loss +
                     align_weight * align_loss +
                    #  ratio_weight * ratio_loss +
                     beta * context_loss)

    return combined_loss


class ContextMixer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ratio_a = nn.Parameter(torch.tensor(0.2, device=device))
        self.ratio_b = nn.Parameter(torch.tensor(1.0, device=device))

    def forward(self, context_vector, output):
        return self.ratio_a * context_vector + self.ratio_b * output


def main():
    train_data, test_data= load_data()
    DEVICE = device
    graphs = load_graphs()
    subject_id_array = train_data['subject_id']
    labels_array = train_data['labels']

    # Get unique ordered subject IDs
    subject_id_ordered = list(OrderedDict.fromkeys(subject_id_array))

    # Dictionary to store the last label for each subject_id
    subject_last_label = {}

    # Iterate over unique ordered subject_ids
    for subject_id in subject_id_ordered:
        # Get indices where this subject_id occurs
        indices = [i for i, sid in enumerate(subject_id_array) if sid == subject_id]

        # Get the last label for this subject_id
        last_label = labels_array[indices[-1]]  # Take the label at the last index

        # Store the last label in the dictionary
        subject_last_label[subject_id] = last_label


    # Initialize models
    code_gnn = GNN(768, 768, 768).to(DEVICE)
    lab_gnn = GNN(2227, 2227, 2227).to(DEVICE)
    image_gnn = GNN(2048, 2048, 2048).to(DEVICE)
    imagebind = ImageBindAlignment(768, 768, 2227, 2048, 4096, temperature=0.1).to(DEVICE)
    models = (code_gnn, lab_gnn, image_gnn, imagebind)


            # code_gnn, lab_gnn, image_gnn, imagebind = models
    context_vector_dim = 4096  # Dimension of each modality (notes, labs, codes)
    llm_hidden_dim = 4096  # Hidden size of the LLM

    # Initialize the InjectedLLM class
    # injected_llm = InjectedLLM(context_vector_dim, llm_hidden_dim).to(device)
    # injected_llm.train()

    code_gnn.train()
    lab_gnn.train()
    image_gnn.train()
    imagebind.train()
    # context_mixer.train()
    # optimizer = optim.Adam(list(code_gnn.parameters()) + list(lab_gnn.parameters()) +
    #                         list(image_gnn.parameters()) + list(imagebind.parameters()) + list(injected_llm.parameters()), lr=1e-4)

    criterion = nn.BCELoss()
    scaler = GradScaler()
    prediction_lst = []
    prediction_text = []

    ratio_a = nn.Parameter(torch.tensor(0.2, device=device))  # Coefficient for  context vector
    ratio_b = nn.Parameter(torch.tensor(1.0, device=device))  # Coefficient for MHA query vector

    # optimizer = optim.Adam(list(code_gnn.parameters()) + list(lab_gnn.parameters()) +
    #                     list(image_gnn.parameters()) + list(imagebind.parameters()), lr=1e-4, weight_decay=1e-5)
    optimizer = optim.Adam([
        {'params': code_gnn.parameters(), 'lr': 1e-3},
        {'params': lab_gnn.parameters(), 'lr': 1e-3},
        {'params': image_gnn.parameters(), 'lr': 1e-3},
        {'params': imagebind.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-5)
    # optimizer = optim.Adam(list(imagebind.parameters()), lr = 1e-3)
    num_epochs = 10
    accumulation_steps = 1
    subject_id_ordered = list(OrderedDict.fromkeys(train_data['subject_id']))
    clip_value = 1.0

    optimizer.zero_grad()
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        total_loss = 0
        shuffled_indices = torch.randperm(len(subject_id_ordered))



        code_gnn.train()
        lab_gnn.train()
        image_gnn.train()
        imagebind.train()

        all_preds = []
        all_labels = []
        all_text = []
        count = 0
        # for i in range(len(train_data['notes'])):
        for i in range(len(set(train_data['subject_id']))):
        # for i in range(10):
            index = shuffled_indices[i].item()
            # index = i
            current_subject_id = subject_id_ordered[index]
            indices_for_current_subject_id = [i for i, subject_id in enumerate(train_data['subject_id']) if subject_id == current_subject_id]
            optimizer.zero_grad()
            count += 1


            code_embeds, lab_embeds, image_embeds = process_embeddings(models, graphs, train_data, 'train')

            code_embeds, lab_embeds, image_embeds = code_embeds.to(device), lab_embeds.to(device), image_embeds.to(device)
            # note_proj, code_proj, lab_proj, image_proj, align_loss = imagebind(torch.tensor(train_data['note_embeddings'].squeeze(1)).to(device), code_embeds, lab_embeds, image_embeds)
            # print(f"Iteration {i}, note_proj mean: {note_proj.mean().item()}, std: {note_proj.std().item()}")
            # print(f"Iteration {i}, code_proj mean: {code_proj.mean().item()}, std: {code_proj.std().item()}")
            fused_proj, align_loss = imagebind(torch.tensor(train_data['note_embeddings'].squeeze(1)).to(device), code_embeds, lab_embeds, image_embeds)

            # aligned_embeddings = torch.stack([code_proj, lab_proj, image_proj], dim=0).reshape(-1, 3, 4096)
            # context_vector = aligned_embeddings[[indices_for_current_subject_id]].mean(dim = (0,1))

            notes_at_indices = [train_data['notes'][i] for i in indices_for_current_subject_id]

            # context_vector = note_proj[indices_for_current_subject_id].mean(dim = 0)
            context_vector = fused_proj[indices_for_current_subject_id].mean(dim = 0)



            # Concatenate all notes into a single string
            concatenated_notes = " ".join(notes_at_indices)
            # context_vector = modality_rnn(aligned_embeddings[[indices_for_current_subject_id]].unsqueeze(0)).squeeze(0)
            # prediction, text = make_prediction(concatenated_notes,context_vector, context_mixer)
            prediction, text = make_prediction(concatenated_notes, context_vector)


            ground_truth = subject_last_label[current_subject_id]

            loss = compute_policy_loss(torch.tensor(prediction), torch.tensor(ground_truth), align_loss, context_vector, imagebind)
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item() * accumulation_steps




            if (i + 1) % accumulation_steps == 0 or (i + 1) == 100:  # If we've accumulated enough steps or it's the last iteration
                torch.nn.utils.clip_grad_norm_(
                    list(code_gnn.parameters()) + 
                    list(lab_gnn.parameters()) +
                    list(image_gnn.parameters()) +
                    list(imagebind.parameters()),
                    max_norm=clip_value
                )

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad() 
            torch.cuda.empty_cache()
            all_preds.append(prediction)
            all_labels.append(ground_truth)
            all_text.append(text)

            torch.cuda.empty_cache()
        normalized_predictions = np.array(all_preds) / 100.0
        binary_predictions = (normalized_predictions > 0.5).astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, binary_predictions)

        # Calculate AUC (Area Under the ROC Curve)
        auc = roc_auc_score(all_labels, normalized_predictions)

        # Calculate F1 Score
        f1 = f1_score(all_labels, binary_predictions, average='macro')

        logger.info(f"Training predictions: {all_preds}")
        # Log the results
        logger.info(f"Training - Accuracy: {accuracy:.4f}")
        logger.info(f"Training - AUC: {auc:.4f}")
        logger.info(f"Training - F1 Score: {f1:.4f}")

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
        if accuracy > best_accuracy:
            best_accuracy  = accuracy
            combined_state_dict = {
                'code_gnn': code_gnn.state_dict(),
                'lab_gnn': lab_gnn.state_dict(),
                'image_gnn': image_gnn.state_dict(),
                'imagebind': imagebind.state_dict()
            }
            logger.info('Saving best model to /cbica/home/NAME/project/downsampled_data/combined_models_llama3_3layers_k1000_final_10epochs_bceLoss_lrrate_5_nnloss_allgraph.pth')
            torch.save(combined_state_dict, '/cbica/home/NAME/project/downsampled_data/combined_models_llama3_3layers_k1000_final_10epochs_bceLoss_lrrate_5_nnloss_allgraph.pth')


    test_predictions = {}
    test_true_labels = {}
    test_preds_lst = []
    test_labels_lst = []

    subject_id_array = test_data['subject_id']
    labels_array = test_data['labels']

    # Get unique ordered subject IDs
    subject_id_ordered = list(OrderedDict.fromkeys(subject_id_array))

    # Dictionary to store the last label for each subject_id
    subject_last_label = {}

    # Iterate over unique ordered subject_ids
    for subject_id in subject_id_ordered:
        # Get indices where this subject_id occurs
        indices = [i for i, sid in enumerate(subject_id_array) if sid == subject_id]
        
        # Get the last label for this subject_id
        last_label = labels_array[indices[-1]]  # Take the label at the last index
        
        # Store the last label in the dictionary
        subject_last_label[subject_id] = last_label

    code_gnn.eval()
    lab_gnn.eval()
    image_gnn.eval()
    imagebind.eval()
    # context_mixer.eval()
    # with torch.no_grad():
    all_preds = []
    all_labels = []
    for i in range(len(set(test_data['subject_id']))):

        # index = shuffled_indices[i].item()
        index = i
        current_subject_id = subject_id_ordered[index]
        indices_for_current_subject_id = [i for i, subject_id in enumerate(test_data['subject_id']) if subject_id == current_subject_id]


        # Process embeddings for evaluation

        code_embeds, lab_embeds, image_embeds = process_embeddings(models, graphs, test_data, 'all')
        code_embeds, lab_embeds, image_embeds = code_embeds.to(device), lab_embeds.to(device), image_embeds.to(device)

        code_embeds = code_embeds[len(train_data['subject_id']):]
        lab_embeds = lab_embeds[len(train_data['subject_id']):]
        image_embeds = image_embeds[len(train_data['subject_id']):]
        fused_proj, align_loss= imagebind(torch.cat((torch.tensor(train_data['note_embeddings'].squeeze(1)), torch.tensor(test_data['note_embeddings'].squeeze(1)))).to(device), code_embeds, lab_embeds, image_embeds)
        fused_proj = fused_proj[len(train_data['subject_id']):]
        context_vector = fused_proj[indices_for_current_subject_id].mean(dim = 0)

        notes_at_indices = [test_data['notes'][i] for i in indices_for_current_subject_id]

        # Concatenate all notes into a single string
        concatenated_notes = " ".join(notes_at_indices)
        prediction, text = make_prediction(concatenated_notes, context_vector)
        ground_truth = subject_last_label[current_subject_id]

        all_preds.append(prediction)
        all_labels.append(ground_truth)
        
    logger.info(f'Test Predictions: {all_preds}')
    normalized_predictions = np.array(all_preds) / 100.0
    logger.info(f'Normalized predictions: {normalized_predictions}')
    logger.info(f'Test labels: {all_labels}')
    binary_predictions = (normalized_predictions > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, binary_predictions)

    # Calculate AUC (Area Under the ROC Curve)
    auc = roc_auc_score(all_labels, normalized_predictions)

    # Calculate F1 Score
    f1 = f1_score(all_labels, binary_predictions, average='macro')


    # Log the final results
    logger.info(f"Final Test Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    
    # Optionally save results as JSON
    # with open("test_predictions_last3_all_lr4.json", "w") as f:
    #     json.dump(all_preds, f, indent=4)
    # with open('./test_ground_truth_last3_all_lr4.json', 'w') as f:
    #     json.dump(all_labels, f, indent=4)

            
if __name__ == "__main__":
    main()
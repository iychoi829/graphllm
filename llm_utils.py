"""
Large Language Model utilities for medical text processing and prediction.
"""
import torch
import re
import logging
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from config import Config

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM initialization and inference for medical predictions."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the language model and tokenizer."""
        logger.info("Initializing LLM...")
        
        # Login to HuggingFace
        if self.config.hf_token and self.config.hf_token != "TOKEN":
            login(token=self.config.hf_token)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.llm_model_name,
            use_auth_token=self.config.hf_token if self.config.hf_token != "TOKEN" else None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.llm_model_name,
            use_auth_token=self.config.hf_token if self.config.hf_token != "TOKEN" else None
        ).to(self.config.device)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id
            
        logger.info("LLM initialized successfully")
    
    def prepare_prompt(self, notes: str, context_vector: torch.Tensor) -> dict:
        """
        Prepare the input prompt for the LLM.
        
        Args:
            notes: Clinical notes text
            context_vector: Multimodal context vector
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Clean notes
        notes = notes.replace('\n', '')
        
        # Define prompt components
        instruction = (
            "You are a medical expert. Here are the clinical notes of a patient across multiple visits: "
        )
        
        question = (
            "\n\nBased on these notes, think step-by-step and assess this patient's probability of experiencing 1-year mortality. "
            "Start by analyzing the patient's medical history, current condition, and relevant lab results. "
            "Then, provide a final mortality risk prediction as a percentage on a scale from 0 to 100, where 0 means no chance of mortality and 100 means certain death.\n\nANSWER: "
        )
        
        # Tokenize components
        instruction_tokens = self.tokenizer(instruction, return_tensors="pt", truncation=False).to(self.config.device)
        question_tokens = self.tokenizer(question, return_tensors="pt", truncation=False).to(self.config.device)
        
        # Calculate available space for notes
        max_notes_length = (self.config.model.max_sequence_length - 
                           instruction_tokens.input_ids.size(1) - 
                           question_tokens.input_ids.size(1))
        
        notes_tokens = self.tokenizer(
            notes, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_notes_length
        ).to(self.config.device)
        
        # Concatenate all components
        input_ids = torch.cat([
            instruction_tokens.input_ids, 
            notes_tokens.input_ids, 
            question_tokens.input_ids
        ], dim=-1)
        
        attention_mask = torch.cat([
            instruction_tokens.attention_mask, 
            notes_tokens.attention_mask, 
            question_tokens.attention_mask
        ], dim=-1)
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    def make_prediction(self, notes: str, context_vector: torch.Tensor) -> Tuple[float, str]:
        """
        Generate a prediction using the LLM with context injection.
        
        Args:
            notes: Clinical notes text
            context_vector: Multimodal context vector
            
        Returns:
            Tuple of (numeric_prediction, full_text_response)
        """
        # Validate inputs
        if torch.isnan(context_vector).any() or torch.isinf(context_vector).any():
            logger.warning("Context vector contains NaN or Inf values!")
        
        # Prepare inputs
        inputs = self.prepare_prompt(notes, context_vector)
        
        # Validate inputs
        if torch.isnan(inputs['input_ids']).any() or torch.isinf(inputs['input_ids']).any():
            logger.warning("Input IDs contain NaN or Inf values!")
        if torch.isnan(inputs['attention_mask']).any() or torch.isinf(inputs['attention_mask']).any():
            logger.warning("Attention mask contains NaN or Inf values!")
        
        # Define hook for context injection
        def context_injection_hook(module, input, output):
            """Hook to inject context into MLP layers."""
            if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
                context_vector_local = context_vector.to(output.device)
                modified_output = (self.config.training.ratio_a * context_vector_local + 
                                 self.config.training.ratio_b * output)
                return modified_output
            return output
        
        # Register hooks
        handles = []
        for layer in self.model.model.layers:
            handles.append(layer.mlp.register_forward_hook(context_injection_hook))
            handles.append(layer.self_attn.register_forward_hook(context_injection_hook))
        
        try:
            # Generate response
            input_ids = inputs['input_ids'].to(self.model.device)
            attention_mask = inputs['attention_mask'].to(self.model.device)
            
            output = self.model.generate(
                input_ids=input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=attention_mask,
                max_new_tokens=self.config.model.max_new_tokens,
                temperature=self.config.model.generation_temperature,
                top_p=self.config.model.top_p,
                do_sample=True
            )
            
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        # Decode response
        prediction_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract numeric value
        prediction_value = extract_numeric_value(prediction_text)
        
        return prediction_value, prediction_text


def extract_numeric_value(prediction: str) -> float:
    """
    Extract numeric mortality prediction from LLM response.
    
    Args:
        prediction: Raw text prediction from LLM
        
    Returns:
        Numeric prediction value (0-100) or -1.0 if not found
    """
    # Clean input
    prediction = prediction.strip()
    
    # Look for answer section
    if 'ANSWER:' in prediction:
        answer_part = prediction.split('ANSWER:')[-1].strip()
    else:
        answer_part = prediction
    
    # Pattern 1: Number followed by percentage sign
    match = re.search(r'(\d+)%', answer_part)
    if match:
        return float(match.group(1))
    
    # Pattern 2: Numeric value at the end
    match = re.search(r'\d+(?:\.\d+)?$', answer_part)
    if match:
        return float(match.group())
    
    # Pattern 3: Any numeric value
    match = re.search(r'\d+(?:\.\d+)?', answer_part)
    if match:
        return float(match.group())
    
    # No numeric value found
    logger.warning("No numeric value found in prediction")
    return -1.0 
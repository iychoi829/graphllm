"""
Main entry point for multimodal medical prediction model.

This script orchestrates the training and evaluation of a multimodal model
that combines clinical notes, medical codes, lab results, and medical images
for mortality prediction using graph neural networks and large language models.
"""
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from config import Config
from data_loader import DataLoader, GraphBuilder
from models import create_models
from llm_utils import LLMManager
from training import train_model


def setup_environment():
    """Set up the environment and check requirements."""
    logger.info("Setting up environment...")
    
    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA not available, using CPU")
    
    # Check critical dependencies
    try:
        import transformers
        import torch_geometric
        import sklearn
        logger.info("All dependencies verified")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        sys.exit(1)


def validate_config(config: Config):
    """Validate configuration and check file paths."""
    logger.info("Validating configuration...")
    
    # Check if data files exist
    if not Path(config.data.train_data_path).exists():
        logger.error(f"Training data file not found: {config.data.train_data_path}")
        # Don't exit - might be running in different environment
        
    if not Path(config.data.test_data_path).exists():
        logger.error(f"Test data file not found: {config.data.test_data_path}")
        # Don't exit - might be running in different environment
    
    # Create output directory if it doesn't exist
    output_dir = Path(config.data.model_save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Configuration validated")


def main():
    """Main execution function."""
    logger.info("Starting multimodal medical prediction training")
    
    try:
        # Setup
        setup_environment()
        
        # Load configuration
        config = Config()
        validate_config(config)
        
        logger.info("Initializing components...")
        
        # Initialize data loader and load data
        data_loader = DataLoader(config)
        train_data, test_data = data_loader.load_data()
        
        # Build graphs
        graph_builder = GraphBuilder(config)
        graphs = graph_builder.load_graphs()
        
        # Initialize models
        models = create_models(config)
        
        # Initialize LLM manager
        llm_manager = LLMManager(config)
        
        logger.info("Starting training...")
        
        # Train the model
        trainer = train_model(
            config=config,
            models=models,
            graphs=graphs,
            train_data=train_data,
            test_data=test_data,
            llm_manager=llm_manager
        )
        
        logger.info("Training completed successfully!")
        
        # Optional: Run additional evaluation or save results
        logger.info("Running final evaluation...")
        eval_metrics, predictions, labels = trainer.evaluate(test_data)
        
        logger.info(f"Final Results:")
        logger.info(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        logger.info(f"  AUC: {eval_metrics['auc']:.4f}")
        logger.info(f"  F1 Score: {eval_metrics['f1']:.4f}")
        
        # Save predictions if needed
        # import json
        # with open('final_predictions.json', 'w') as f:
        #     json.dump({
        #         'predictions': predictions,
        #         'labels': labels,
        #         'metrics': eval_metrics
        #     }, f, indent=2)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 
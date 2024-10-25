import argparse
import torch
import numpy as np
from tqdm import tqdm
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from model import TextClassifierPyTorch
from dataset import IMDBDataset, SST2Dataset, CustomTextDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model(model_path, device, args):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        args: Arguments for model initialization
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    # Create model with same architecture
    model = TextClassifierPyTorch(
        num_layers=args.n_transformer_blocks,
        embed_dim=args.embed_dim,
        num_heads=args.n_heads,
        dff=args.ffn_dim,
        vocab_size=args.vocab_size,
        num_classes=args.n_classes,
        maximum_position_encoding=1024,
        dropout_rate=args.dropout_rate,
        n_qubits_transformer=args.n_qubits_transformer,
        n_qubits_ffn=args.n_qubits_ffn,
        n_qlayers=args.n_qlayers,
        q_device=args.q_device
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    model = model.to(device)
    
    return model


def evaluate(model, dataloader, device, n_classes):
    """
    Evaluate model on dataset
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to evaluate on
        n_classes: Number of classes
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    # No gradient computation for evaluation
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs, training=False)
            
            # Get predictions
            if outputs.size(-1) == 1 or len(outputs.size()) == 1:  # Binary classification
                probabilities = outputs.view(-1)
                predicted = (outputs > 0.5).float()
            else:  # Multi-class classification
                probabilities = outputs
                _, predicted = torch.max(outputs.data, 1)
            
            # Store all targets and predictions for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    # Generate classification report
    cls_report = classification_report(all_targets, all_predictions, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'targets': all_targets,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'classification_report': cls_report,
        'confusion_matrix': conf_matrix
    }


def plot_confusion_matrix(conf_matrix, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate a quantum-enhanced transformer model")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'sst2', 'custom'],
                       help='Dataset to use for evaluation (default: imdb)')
    parser.add_argument('--data_path', type=str, default='',
                       help='Path to custom dataset (required if dataset=custom)')
    parser.add_argument('--text_col', type=str, default='text',
                       help='Column name for text in custom CSV dataset (default: text)')
    parser.add_argument('--label_col', type=str, default='label',
                       help='Column name for label in custom CSV dataset (default: label)')
    parser.add_argument('--file_type', type=str, default='csv', choices=['csv', 'txt'],
                       help='File type for custom dataset (default: csv)')
    
    # Model parameters (need to match training configuration)
    parser.add_argument('--embed_dim', type=int, default=8,
                       help='Embedding dimension (default: 8)')
    parser.add_argument('--n_transformer_blocks', type=int, default=1,
                       help='Number of transformer blocks (default: 1)')
    parser.add_argument('--n_heads', type=int, default=2,
                       help='Number of attention heads (default: 2)')
    parser.add_argument('--ffn_dim', type=int, default=8,
                       help='Feed-forward network dimension (default: 8)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    # Quantum parameters
    parser.add_argument('--n_qubits_transformer', type=int, default=0,
                       help='Number of qubits for transformer (default: 0, use classical transformer)')
    parser.add_argument('--n_qubits_ffn', type=int, default=0,
                       help='Number of qubits for FFN (default: 0, use classical FFN)')
    parser.add_argument('--n_qlayers', type=int, default=1,
                       help='Number of quantum layers (default: 1)')
    parser.add_argument('--q_device', type=str, default='default.qubit',
                       help='Quantum device to use (default: default.qubit)')
    
    # Evaluation parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--max_seq_len', type=int, default=64,
                       help='Maximum sequence length (default: 64)')
    parser.add_argument('--vocab_size', type=int, default=20000,
                       help='Vocabulary size (default: 20000)')
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of classes (default: 2)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == 'imdb':
        dataset = IMDBDataset(max_seq_len=args.max_seq_len, vocab_size=args.vocab_size)
        if dataset.test_data is not None:
            eval_dataset = torch.utils.data.TensorDataset(
                torch.tensor(dataset.test_data, dtype=torch.long),
                torch.tensor(dataset.test_labels, dtype=torch.float)
            )
        else:
            eval_dataset = dataset
        class_names = ['Negative', 'Positive']
    elif args.dataset == 'sst2':
        dataset = SST2Dataset(max_seq_len=args.max_seq_len, vocab_size=args.vocab_size)
        if dataset.val_data is not None:
            eval_dataset = torch.utils.data.TensorDataset(
                torch.tensor(dataset.val_data, dtype=torch.long),
                torch.tensor(dataset.val_labels, dtype=torch.float)
            )
        else:
            eval_dataset = dataset
        class_names = ['Negative', 'Positive']
    elif args.dataset == 'custom':
        if not args.data_path:
            parser.error("--data_path is required when dataset=custom")
        dataset = CustomTextDataset(
            data_path=args.data_path,
            text_col=args.text_col,
            label_col=args.label_col,
            max_seq_len=args.max_seq_len,
            vocab_size=args.vocab_size,
            file_type=args.file_type
        )
        if dataset.test_data is not None:
            eval_dataset = torch.utils.data.TensorDataset(
                torch.tensor(dataset.test_data, dtype=torch.long),
                torch.tensor(dataset.test_labels, dtype=torch.float)
            )
        else:
            eval_dataset = dataset
        # Generate class names based on number of classes
        class_names = [f'Class {i}' for i in range(args.n_classes)]
    else:
        parser.error(f"Unknown dataset: {args.dataset}")
    
    # Create dataloader
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Load model
    model = load_model(args.model_path, device, args)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    metrics = evaluate(model, eval_loader, device, args.n_classes)
    
    # Print metrics
    logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    # Print classification report
    logger.info("Classification Report:")
    for class_idx, metrics_dict in metrics['classification_report'].items():
        if class_idx not in ['accuracy', 'macro avg', 'weighted avg']:
            class_name = class_names[int(class_idx)] if class_idx.isdigit() else class_idx
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics_dict['precision']:.4f}")
            logger.info(f"    Recall: {metrics_dict['recall']:.4f}")
            logger.info(f"    F1 Score: {metrics_dict['f1-score']:.4f}")
            logger.info(f"    Support: {metrics_dict['support']}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Save metrics to file
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classification_report': metrics['classification_report']
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    logger.info(f"Metrics saved to {os.path.join(args.output_dir, 'metrics.json')}")
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
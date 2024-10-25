import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import project modules
from model import TextClassifierPyTorch
from dataset import IMDBDataset, SST2Dataset, CustomTextDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for training data
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to run on (cuda or cpu)
        
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for progress bar
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_idx, (inputs, targets) in enumerate(tepoch):
            tepoch.set_description(f"Training")
            
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, training=True)
            
            # Calculate loss
            if outputs.size(-1) == 1 or len(outputs.size()) == 1:  # Binary classification
                loss = criterion(outputs.view(-1), targets)
                predicted = (outputs > 0.5).float()
            else:  # Multi-class classification
                loss = criterion(outputs, targets.long())
                _, predicted = torch.max(outputs.data, 1)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            total_loss += loss.item()
            
            # Update progress bar
            tepoch.set_postfix(loss=total_loss/(batch_idx+1), acc=100.*correct/total)
    
    # Calculate epoch statistics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation or test data
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run on (cuda or cpu)
        
    Returns:
        Average loss, accuracy, and metrics
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    
    # No gradient computation for evaluation
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs, training=False)
            
            # Calculate loss
            if outputs.size(-1) == 1 or len(outputs.size()) == 1:  # Binary classification
                loss = criterion(outputs.view(-1), targets)
                predicted = (outputs > 0.5).float()
            else:  # Multi-class classification
                loss = criterion(outputs, targets.long())
                _, predicted = torch.max(outputs.data, 1)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store all targets and predictions for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, precision, recall, f1


def save_checkpoint(model, optimizer, epoch, loss, accuracy, file_path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        file_path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, file_path)
    logger.info(f"Checkpoint saved to {file_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train a quantum-enhanced transformer model")
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'sst2', 'custom'],
                       help='Dataset to use for training (default: imdb)')
    parser.add_argument('--data_path', type=str, default='',
                       help='Path to custom dataset (required if dataset=custom)')
    parser.add_argument('--text_col', type=str, default='text',
                       help='Column name for text in custom CSV dataset (default: text)')
    parser.add_argument('--label_col', type=str, default='label',
                       help='Column name for label in custom CSV dataset (default: label)')
    parser.add_argument('--file_type', type=str, default='csv', choices=['csv', 'txt'],
                       help='File type for custom dataset (default: csv)')
    
    # Model parameters
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
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=5,
                       help='Number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--max_seq_len', type=int, default=64,
                       help='Maximum sequence length (default: 64)')
    parser.add_argument('--vocab_size', type=int, default=20000,
                       help='Vocabulary size (default: 20000)')
    parser.add_argument('--n_classes', type=int, default=2,
                       help='Number of classes (default: 2)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs (default: 1)')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    if args.dataset == 'imdb':
        dataset = IMDBDataset(max_seq_len=args.max_seq_len, vocab_size=args.vocab_size)
    elif args.dataset == 'sst2':
        dataset = SST2Dataset(max_seq_len=args.max_seq_len, vocab_size=args.vocab_size)
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
    else:
        parser.error(f"Unknown dataset: {args.dataset}")
    
    # Create train and validation dataloaders
    if dataset.val_data is None and args.dataset != 'custom':
        # Create validation split from training data
        train_size = int((1 - args.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # If we have a test set, create a test loader
        if dataset.test_data is not None:
            test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(dataset.test_data, dtype=torch.long),
                torch.tensor(dataset.test_labels, dtype=torch.float)
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False
            )
        else:
            test_loader = None
    else:
        # Dataset already has validation set
        train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=args.batch_size)
    
    # Create model
    logger.info("Creating model...")
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
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model architecture:\n{model}")
    
    # Define loss function
    if args.n_classes == 2:
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Track best validation accuracy
    best_val_acc = 0.0
    
    # Train the model
    logger.info("Starting training...")
    for epoch in range(args.n_epochs):
        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - start_time
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.n_epochs} - "
                    f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%, "
                    f"Val loss: {val_loss:.4f}, Val acc: {val_acc*100:.2f}%, "
                    f"Val precision: {val_precision:.4f}, Val recall: {val_recall:.4f}, Val f1: {val_f1:.4f}, "
                    f"Time: {train_time:.2f}s")
        
        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(args.save_dir, f"best_model.pt")
            )
        
        # Save regular checkpoints
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            )
    
    # Evaluate on test set if available
    if test_loader is not None:
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test - "
                    f"Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%, "
                    f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.n_epochs, val_loss, val_acc,
        os.path.join(args.save_dir, "final_model.pt")
    )
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
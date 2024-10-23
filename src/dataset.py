import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import pandas as pd
from abc import ABC, abstractmethod
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class TextClassificationDataset(Dataset, ABC):
    """
    Abstract base class for text classification datasets
    
    This class can be subclassed to implement different text classification datasets.
    """
    
    def __init__(self, 
                 max_seq_len: int = 256, 
                 vocab_size: int = 20000,
                 pad_token: str = "<pad>",
                 unk_token: str = "<unk>",
                 tokenizer_name: str = "basic_english"):
        """
        Initialize the dataset with basic parameters
        
        Args:
            max_seq_len: Maximum sequence length for padding/truncation
            vocab_size: Maximum vocabulary size
            pad_token: Token used for padding
            unk_token: Token used for unknown words
            tokenizer_name: Name of the tokenizer to use (for torchtext)
        """
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Initialize tokenizer
        self.tokenizer = get_tokenizer(tokenizer_name)
        
        # Will be initialized in subclasses
        self.vocab = None
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None
        self.test_data = None
        self.test_labels = None
        
    @abstractmethod
    def load_data(self) -> None:
        """
        Load the dataset, to be implemented by subclasses
        """
        pass
    
    def build_vocabulary(self, text_iterator) -> None:
        """
        Build vocabulary from text iterator
        
        Args:
            text_iterator: Iterator that yields tokenized texts
        """
        # Special tokens
        special_tokens = [self.pad_token, self.unk_token]
        
        # Build vocabulary from data
        self.vocab = build_vocab_from_iterator(
            text_iterator,
            max_tokens=self.vocab_size,
            specials=special_tokens
        )
        
        # Set default index for unknown tokens
        self.vocab.set_default_index(self.vocab[self.unk_token])
    
    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert text to token indices
        
        Args:
            text: Input text to convert
            
        Returns:
            List of token indices
        """
        # Tokenize text
        tokens = self.tokenizer(text)
        
        # Convert tokens to indices
        indices = [self.vocab[token] for token in tokens]
        
        # Pad or truncate to max_seq_len
        if len(indices) < self.max_seq_len:
            # Pad with pad_token
            indices = indices + [self.vocab[self.pad_token]] * (self.max_seq_len - len(indices))
        else:
            # Truncate to max_seq_len
            indices = indices[:self.max_seq_len]
        
        return indices
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.train_data) if self.train_data is not None else 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item at index"""
        return (
            torch.tensor(self.train_data[idx], dtype=torch.long),
            torch.tensor(self.train_labels[idx], dtype=torch.float)
        )
    
    def get_dataloaders(self, batch_size: int = 32, shuffle: bool = True
                       ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Get dataloaders for train, validation, and test sets
        
        Args:
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        # Create validation dataloader if validation data exists
        val_loader = None
        if self.val_data is not None and self.val_labels is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.val_data, dtype=torch.long),
                torch.tensor(self.val_labels, dtype=torch.float)
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Create test dataloader if test data exists
        test_loader = None
        if self.test_data is not None and self.test_labels is not None:
            test_dataset = torch.utils.data.TensorDataset(
                torch.tensor(self.test_data, dtype=torch.long),
                torch.tensor(self.test_labels, dtype=torch.float)
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        return train_loader, val_loader, test_loader


class IMDBDataset(TextClassificationDataset):
    """
    IMDB movie review dataset for sentiment analysis
    """
    
    def __init__(self, max_seq_len: int = 256, vocab_size: int = 20000):
        """
        Initialize IMDB dataset
        
        Args:
            max_seq_len: Maximum sequence length for padding/truncation
            vocab_size: Maximum vocabulary size
        """
        super().__init__(max_seq_len=max_seq_len, vocab_size=vocab_size)
        
        # Load the dataset
        self.load_data()
    
    def load_data(self) -> None:
        """
        Load IMDB dataset using torchtext
        """
        from torchtext.datasets import IMDB
        
        # Load raw data
        train_iter = IMDB(split='train')
        test_iter = IMDB(split='test')
        
        # Convert to list for processing
        train_data = list(train_iter)
        test_data = list(test_iter)
        
        # Extract texts and labels
        train_texts = [text for (label, text) in train_data]
        train_labels = [int(label) for (label, text) in train_data]
        test_texts = [text for (label, text) in test_data]
        test_labels = [int(label) for (label, text) in test_data]
        
        # Tokenize texts
        train_tokens = [self.tokenizer(text) for text in train_texts]
        
        # Build vocabulary from tokens
        self.build_vocabulary(train_tokens)
        
        # Convert texts to token indices
        self.train_data = [self.text_to_indices(text) for text in train_texts]
        self.test_data = [self.text_to_indices(text) for text in test_texts]
        
        # Convert labels to numpy arrays
        self.train_labels = np.array(train_labels)
        self.test_labels = np.array(test_labels)
        
        # No separate validation set, so we'll split train set in training
        self.val_data = None
        self.val_labels = None


class SST2Dataset(TextClassificationDataset):
    """
    Stanford Sentiment Treebank dataset for sentiment analysis
    """
    
    def __init__(self, max_seq_len: int = 256, vocab_size: int = 20000):
        """
        Initialize SST2 dataset
        
        Args:
            max_seq_len: Maximum sequence length for padding/truncation
            vocab_size: Maximum vocabulary size
        """
        super().__init__(max_seq_len=max_seq_len, vocab_size=vocab_size)
        
        # Load the dataset
        self.load_data()
    
    def load_data(self) -> None:
        """
        Load SST2 dataset
        """
        try:
            # Try to import datasets (HuggingFace)
            from datasets import load_dataset
            
            # Load SST2 dataset
            dataset = load_dataset("glue", "sst2")
            
            # Extract train, validation, and test sets
            train_data = dataset["train"]
            val_data = dataset["validation"]
            
            # Extract texts and labels
            train_texts = train_data["sentence"]
            train_labels = train_data["label"]
            val_texts = val_data["sentence"]
            val_labels = val_data["label"]
            
            # Tokenize texts
            train_tokens = [self.tokenizer(text) for text in train_texts]
            
            # Build vocabulary from tokens
            self.build_vocabulary(train_tokens)
            
            # Convert texts to token indices
            self.train_data = [self.text_to_indices(text) for text in train_texts]
            self.val_data = [self.text_to_indices(text) for text in val_texts]
            
            # Convert labels to numpy arrays
            self.train_labels = np.array(train_labels)
            self.val_labels = np.array(val_labels)
            
            # No separate test set in GLUE benchmark
            self.test_data = None
            self.test_labels = None
            
        except ImportError:
            print("HuggingFace datasets library not found. Cannot load SST2 dataset.")
            raise


class CustomTextDataset(TextClassificationDataset):
    """
    Custom text classification dataset from CSV or text files
    """
    
    def __init__(self, 
                 data_path: str,
                 text_col: str = "text",
                 label_col: str = "label",
                 max_seq_len: int = 256, 
                 vocab_size: int = 20000,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 test_split: float = 0.1,
                 file_type: str = "csv"):
        """
        Initialize custom dataset
        
        Args:
            data_path: Path to data file
            text_col: Column name for text (if CSV)
            label_col: Column name for label (if CSV)
            max_seq_len: Maximum sequence length for padding/truncation
            vocab_size: Maximum vocabulary size
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            file_type: Type of data file ("csv" or "txt")
        """
        super().__init__(max_seq_len=max_seq_len, vocab_size=vocab_size)
        
        self.data_path = data_path
        self.text_col = text_col
        self.label_col = label_col
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.file_type = file_type
        
        # Load the dataset
        self.load_data()
    
    def load_data(self) -> None:
        """
        Load custom dataset from files
        """
        if self.file_type == "csv":
            # Load from CSV
            df = pd.read_csv(self.data_path)
            
            # Extract texts and labels
            texts = df[self.text_col].tolist()
            labels = df[self.label_col].tolist()
        elif self.file_type == "txt":
            # Load from text files organized in folders by class
            texts = []
            labels = []
            class_folders = [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
            
            for i, class_folder in enumerate(class_folders):
                folder_path = os.path.join(self.data_path, class_folder)
                files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
                
                for file in files:
                    with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(i)  # Use folder index as class label
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        
        # Tokenize texts
        tokens = [self.tokenizer(text) for text in texts]
        
        # Build vocabulary from tokens
        self.build_vocabulary(tokens)
        
        # Convert texts to token indices
        indices = [self.text_to_indices(text) for text in texts]
        
        # Split data into train, validation, and test sets
        n_samples = len(indices)
        indices = np.array(indices)
        labels = np.array(labels)
        
        # Shuffle data
        shuffle_idx = np.random.permutation(n_samples)
        indices = indices[shuffle_idx]
        labels = labels[shuffle_idx]
        
        # Split data
        n_train = int(n_samples * self.train_split)
        n_val = int(n_samples * self.val_split)
        
        self.train_data = indices[:n_train]
        self.train_labels = labels[:n_train]
        
        self.val_data = indices[n_train:n_train+n_val]
        self.val_labels = labels[n_train:n_train+n_val]
        
        self.test_data = indices[n_train+n_val:]
        self.test_labels = labels[n_train+n_val:]
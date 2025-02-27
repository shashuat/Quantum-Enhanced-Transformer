# Quantum Enhanced Transformer

A PyTorch implementation of a quantum-enhanced transformer model for text classification tasks. This project combines classical transformer architectures with quantum computing elements to potentially improve performance on NLP tasks.

![Quantum Transformer Architecture]()

## Features

- **Hybrid Classical-Quantum Architecture**: Integrate quantum computing into transformer models
- **PennyLane Integration**: Leverage PennyLane for quantum machine learning
- **Flexible Dataset Support**: Works with IMDB, SST2, and custom datasets
- **Modular Design**: Separate model, dataset, training, and evaluation components
- **Quantum Circuit Customization**: Configure qubits, quantum layers, and quantum devices
- **Comprehensive Evaluation**: Detailed metrics and visualizations for model performance

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- PennyLane 0.25+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/shashuat/quantum-enhanced-transformer.git
   cd quantum-enhanced-transformer
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
quantum-enhanced-transformer/
|-- src/
    ├── model.py           # Transformer model implementation
    ├── dataset.py         # Dataset handling and preprocessing
    ├── train.py           # Training script
    ├── evaluate.py        # Evaluation script
├── requirements.txt   # Dependencies
└── README.md          # This readme file
```

## Usage

### Training a Model

To train a model using the IMDB dataset with classical transformer:

```bash
python train.py --dataset imdb --n_epochs 5 --batch_size 32
```

To train a model with quantum enhancement:

```bash
python train.py --dataset imdb --n_qubits_transformer 8 --n_qubits_ffn 8 --n_qlayers 2 --q_device default.qubit
```

### Using a Custom Dataset

For CSV files:

```bash
python train.py --dataset custom --data_path your_data.csv --text_col text_column --label_col label_column
```

For text files organized in class folders:

```bash
python train.py --dataset custom --data_path your_data_folder --file_type txt
```

### Evaluating a Model

```bash
python evaluate.py --model_path checkpoints/best_model.pt --dataset imdb --n_qubits_transformer 8 --n_qubits_ffn 8
```

## Quantum Components Explained

The model supports two different types of implementation:

1. **Classical Transformer**: Standard transformer architecture with multi-head attention and feed-forward networks
2. **Quantum-Enhanced Transformer**: 
   - **Quantum Attention**: Replaces the linear projections in the attention mechanism with quantum circuits
   - **Quantum Feed-Forward**: Uses quantum circuits in the feed-forward network

### Quantum Circuit Configuration

- `n_qubits_transformer`: Number of qubits for the quantum attention mechanism (should match embed_dim)
- `n_qubits_ffn`: Number of qubits for the quantum feed-forward network
- `n_qlayers`: Number of quantum layers (entangling layers) in the circuit
- `q_device`: Quantum device to use ('default.qubit', 'qulacs.simulator', 'braket.local.qubit', etc.)

## Key Parameters

### Model Parameters
- `embed_dim`: Dimension of token embeddings
- `n_transformer_blocks`: Number of transformer blocks
- `n_heads`: Number of attention heads
- `ffn_dim`: Dimension of the feed-forward network
- `dropout_rate`: Dropout rate for regularization

### Training Parameters
- `batch_size`: Batch size for training
- `n_epochs`: Number of training epochs
- `lr`: Learning rate
- `max_seq_len`: Maximum sequence length
- `vocab_size`: Size of the vocabulary

## Evaluation Metrics

The evaluation script provides:

- Accuracy, Precision, Recall, F1 Score
- Per-class performance metrics
- Confusion matrix visualization
- Detailed classification report

## Acknowledgements

- The classical transformer implementation is based on the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.
- Quantum computing components use [PennyLane](https://pennylane.ai/) for quantum circuit simulation.
- A large part of the code is based of from the implementation of [qtransformer](https://github.com/rdisipio/qtransformer) from the paper [The Dawn of Quantum Natural Language Processing](https://arxiv.org/pdf/2110.06510)

## Citations

If you use this code in your research, please cite:

```
@misc{quantum-enhanced-transformer,
  author = {Shashwat Sharma},
  title = {Quantum Enhanced Transformer},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shashuat/quantum-enhanced-transformer}}
}
```


## License

This project is licensed under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0) - see the LICENSE file for details.

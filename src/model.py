import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml
import os
import math

USE_GPU = bool(os.environ.get('USE_GPU', False))

def get_positional_encoding(seq_len, embed_dim):
    """
    Creates positional encoding for transformer model
    """
    # Create position indices tensor
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    
    # Initialize positional encoding
    pe = torch.zeros(seq_len, embed_dim)
    
    # Apply sin to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cos to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def create_padding_mask(seq):
    """
    Creates padding mask for sequences with padding tokens (0)
    """
    # Create mask where 1's are at padded positions
    mask = (seq == 0).float()
    # Add extra dimensions for attention
    return mask.unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size):
    """
    Creates causal/look-ahead mask for self-attention
    """
    # Create upper triangular mask
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Computes scaled dot product attention
    """
    # Get dimensions
    batch_size, num_heads, seq_len_q, depth = q.size()
    _, _, seq_len_k, _ = k.size()
    
    # Matmul and scale
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(depth)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 1, -1e9)
    
    # Apply softmax for attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi-head attention"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBase, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        assert embed_dim % self.num_heads == 0, f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
        
        self.depth = embed_dim // self.num_heads
        
        self.wq = None
        self.wk = None
        self.wv = None
        self.dense = None
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth)
        Transpose result to shape (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def apply_dense_layers(self, v, k, q):
        """Apply dense layers to the query, key, value inputs"""
        raise NotImplementedError("Base class does not implement apply_dense_layers() function")
    
    def apply_combine_heads(self, x):
        """Combine heads back to original dimensions"""
        raise NotImplementedError("Base class does not implement apply_combine_heads() function")
    
    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        
        v, k, q = self.apply_dense_layers(v, k, q)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # Transpose to (batch_size, seq_len, num_heads, depth)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        
        concat_attention = scaled_attention.reshape(batch_size, -1, self.embed_dim)
        
        output = self.apply_combine_heads(concat_attention)
        return output, attention_weights


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Classical implementation of multi-head attention"""
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionClassical, self).__init__(embed_dim, num_heads)
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)
    
    def apply_dense_layers(self, v, k, q):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        return v, k, q
    
    def apply_combine_heads(self, x):
        return self.dense(x)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum implementation of multi-head attention"""
    def __init__(self, embed_dim, num_heads, n_qubits, n_qlayers=1, q_device='default.qubit'):
        super(MultiHeadAttentionQuantum, self).__init__(embed_dim, num_heads)
        
        assert n_qubits == embed_dim, f"Number of qubits ({n_qubits}) does not match embedding dim ({embed_dim})"
        
        # Setup quantum device
        if 'qulacs' in q_device:
            print(f"Quantum device: Qulacs: {q_device}")
            if USE_GPU is True:
                print("Qulacs will use the GPU")
            self.dev = qml.device(q_device, wires=n_qubits, gpu=USE_GPU)
        elif 'braket' in q_device:
            print(f"Quantum device: Amazon Braket: {q_device}")
            self.dev = qml.device(q_device, wires=n_qubits, parallel=True)
        else:
            print(f"Quantum device: {q_device}")
            self.dev = qml.device(q_device, wires=n_qubits)
        
        # Define quantum circuit
        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        
        self.qlayer = qml.QNode(_circuit, self.dev, interface="torch")
        
        # Initialize quantum layers
        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")
        
        self.wq = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
        self.wk = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
        self.wv = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
        self.dense = qml.qnn.TorchLayer(self.qlayer, weight_shapes)
    
    def apply_dense_layers(self, v, k, q):
        batch_size, seq_len, _ = q.size()
        
        # Process each sequence position separately
        q_list = [self.wq(q[:, t, :]) for t in range(seq_len)]
        k_list = [self.wk(k[:, t, :]) for t in range(seq_len)]
        v_list = [self.wv(v[:, t, :]) for t in range(seq_len)]
        
        # Stack and reshape to original dimensions
        q = torch.stack(q_list, dim=1)
        k = torch.stack(k_list, dim=1)
        v = torch.stack(v_list, dim=1)
        
        return v, k, q
    
    def apply_combine_heads(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Process each sequence position separately
        output_list = [self.dense(x[:, t, :]) for t in range(seq_len)]
        
        # Stack and reshape to original dimensions
        output = torch.stack(output_list, dim=1)
        
        return output


class FeedForwardClassical(nn.Module):
    """Classical feed-forward network for transformer"""
    def __init__(self, embed_dim, dff):
        super(FeedForwardClassical, self).__init__()
        self.linear1 = nn.Linear(embed_dim, dff)
        self.linear2 = nn.Linear(dff, embed_dim)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class FeedForwardQuantum(nn.Module):
    """Quantum feed-forward network for transformer"""
    def __init__(self, embed_dim, dff, n_qubits_ffn, n_qlayers=1, q_device='default.qubit'):
        super(FeedForwardQuantum, self).__init__()
        # For simplicity, we'll use classical implementation but in a real quantum model
        # this would use a quantum circuit like in MultiHeadAttentionQuantum
        self.linear1 = nn.Linear(embed_dim, dff)
        self.linear2 = nn.Linear(dff, embed_dim)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlockBase(nn.Module):
    """Base class for transformer block"""
    def __init__(self, embed_dim, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlockBase, self).__init__()
        self.mha = None
        self.ffn = None
        
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None, training=True):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output) if training else attn_output
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output) if training else ffn_output
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block"""
    def __init__(self, embed_dim, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlockClassical, self).__init__(embed_dim, num_heads, dff, dropout_rate)
        self.mha = MultiHeadAttentionClassical(embed_dim, num_heads)
        self.ffn = FeedForwardClassical(embed_dim, dff)


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum transformer block"""
    def __init__(self, embed_dim, num_heads, dff, dropout_rate=0.1,
                n_qubits_transformer=0, n_qubits_ffn=0, n_qlayers=1, q_device='default.qubit'):
        super(TransformerBlockQuantum, self).__init__(embed_dim, num_heads, dff, dropout_rate)
        self.mha = MultiHeadAttentionQuantum(embed_dim, num_heads, n_qubits_transformer, n_qlayers, q_device)
        self.ffn = FeedForwardQuantum(embed_dim, dff, n_qubits_ffn, n_qlayers, q_device)


class EncoderLayerBase(nn.Module):
    """Base encoder layer"""
    def __init__(self, num_layers, embed_dim, num_heads, dff, vocab_size,
                maximum_position_encoding, dropout_rate=0.1):
        super(EncoderLayerBase, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = get_positional_encoding(maximum_position_encoding, embed_dim)
        self.enc_layers = None
        self.dropout = nn.Dropout(dropout_rate)
        
        # Register positional encoding as buffer (non-trainable tensor)
        self.register_buffer('positional_encoding', self.pos_encoding)
    
    def forward(self, x, mask=None, training=True):
        seq_len = x.size(1)
        
        # Convert input to embeddings
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Apply dropout
        x = self.dropout(x) if training else x
        
        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)
        
        return x


class EncoderLayerClassical(EncoderLayerBase):
    """Classical encoder layer"""
    def __init__(self, num_layers, embed_dim, num_heads, dff, vocab_size,
                maximum_position_encoding, dropout_rate=0.1):
        super(EncoderLayerClassical, self).__init__(num_layers, embed_dim, num_heads, dff, 
                                                  vocab_size, maximum_position_encoding, dropout_rate)
        
        self.enc_layers = nn.ModuleList([
            TransformerBlockClassical(embed_dim, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])


class EncoderLayerQuantum(EncoderLayerBase):
    """Quantum encoder layer"""
    def __init__(self, num_layers, embed_dim, num_heads, dff, vocab_size,
                maximum_position_encoding, dropout_rate=0.1,
                n_qubits_transformer=0, n_qubits_ffn=0, n_qlayers=1, q_device="default.qubit"):
        super(EncoderLayerQuantum, self).__init__(num_layers, embed_dim, num_heads, dff, 
                                                vocab_size, maximum_position_encoding, dropout_rate)
        
        self.enc_layers = nn.ModuleList([
            TransformerBlockQuantum(embed_dim, num_heads, dff, dropout_rate,
                                   n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device)
            for _ in range(num_layers)
        ])


class TextClassifierPyTorch(nn.Module):
    """Text classifier using transformer architecture"""
    def __init__(self, num_layers, embed_dim, num_heads, dff, vocab_size, num_classes,
                maximum_position_encoding=10000, dropout_rate=0.1,
                n_qubits_transformer=0, n_qubits_ffn=0, n_qlayers=1, q_device="default.qubit"):
        super(TextClassifierPyTorch, self).__init__()
        
        # Choose between classical and quantum encoder
        if n_qubits_transformer == 0 and n_qubits_ffn == 0:
            self.encoder = EncoderLayerClassical(num_layers, embed_dim, num_heads, dff,
                                               vocab_size, maximum_position_encoding, dropout_rate)
        else:
            self.encoder = EncoderLayerQuantum(num_layers, embed_dim, num_heads, dff,
                                             vocab_size, maximum_position_encoding, dropout_rate,
                                             n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device)
        
        # Final classifier layer
        if num_classes < 2:
            raise RuntimeError("Number of classes must be at least 2")
        elif num_classes == 2:
            self.final_layer = nn.Linear(embed_dim, 1)
            self.activation = nn.Sigmoid()
        else:
            self.final_layer = nn.Linear(embed_dim, num_classes)
            self.activation = nn.Softmax(dim=-1)
    
    def forward(self, x, training=True):
        # Get encoded output from transformer encoder
        encoded_output = self.encoder(x, training=training)
        
        # Use first token ([CLS]) for classification
        pooled_output = encoded_output[:, 0, :]
        
        # Apply final classification layer
        logits = self.final_layer(pooled_output)
        
        # Apply activation (sigmoid for binary, softmax for multi-class)
        if logits.size(-1) == 1:  # Binary classification
            output = self.activation(logits).squeeze(-1)
        else:  # Multi-class classification
            output = self.activation(logits)
        
        return output
"""
LEMUR Training Pipeline
"""

import json
import os
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import save_file

@dataclass
class LemurConfig:
    """Configuration for Lemur model."""

    hidden_dim: int = 2048
    n_train_tokens: int = 100000
    n_train_docs: int = 8192
    n_ols_tokens: int = 16384
    epochs: int = 100
    lr: float = 0.003
    batch_size: int = 512
    grad_clip: float = 0.5
    seed: int = 42
    device: object = None

class FeatureEncoder(nn.Module):
    """One-hidden-layer feature encoder."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """Forward pass."""
        return self.norm(self.act(self.linear(x)))

class LemurMLP(nn.Module):
    """Training MLP (encoder + linear head)."""

    def __init__(self, input_dim, hidden_dim, n_target_docs):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_target_docs, bias=False)

    def forward(self, x):
        """Forward pass."""
        psi = self.encoder(x)
        return self.head(psi)
    


class LemurTrainer:
    """Trains the LEMUR Feature Encoder and serializes it for the HuggingFace Hub."""

    def __init__(self, config=None):
        self.config = config or LemurConfig() 
        self.device = self.config.device or self._device()
        self.encoder = None

    def __call__(self, documents, output_dir):
        """Main training pipeline."""
        self.train(documents)
        self.save(output_dir)

    def train(self, documents):
        """Samples the dataset, builds targets, and trains the feature encoder."""
        torch.manual_seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)

        n_docs = len(documents)
        d = documents[0].shape[1]

        m_prime = min(self.config.n_train_docs, n_docs)
        target_idx = rng.choice(n_docs, size=m_prime, replace=False)
        target_docs = [documents[i] for i in target_idx]

        all_tokens = np.vstack(documents).astype(np.float32)
        n_total = all_tokens.shape[0]

        n_sample = min(self.config.n_train_tokens, n_total)
        train_tokens = all_tokens[rng.choice(n_total, size=n_sample, replace=False)]

        targets = self._targets(train_tokens, target_docs)

        out_mean = targets.mean(axis=0)
        out_std = targets.std(axis=0) + 1e-8
        targets_norm = (targets - out_mean) / out_std

        model = LemurMLP(d, self.config.hidden_dim, m_prime).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion = nn.MSELoss()

        x_tensor = torch.from_numpy(train_tokens).to(self.device)
        y_tensor = torch.from_numpy(targets_norm).to(self.device)

        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        model.train()
        for _ in range(self.config.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

        self.encoder = model.encoder
        self.encoder.eval()

    def _targets(self, tokens, target_docs):
        """Computes targets."""
        targets = np.zeros((tokens.shape[0], len(target_docs)), dtype=np.float32)

        for j, doc in enumerate(target_docs):
            targets[:, j] = self._maxsim_per_token(tokens, doc.astype(np.float32))

        return targets

    def _maxsim_per_token(self, tokens, doc):
        """Per-token MaxSim."""
        return (tokens @ doc.T).max(axis=1)
    
    def _device(self):
        """Selects best device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def save(self, output_dir):
        """Serializes the model using Safetensors and config.json."""
        if self.encoder is None:
            raise ValueError("Model must be trained before saving.")

        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights safely
        model_path = os.path.join(output_dir, "model.safetensors")
        save_file(self.encoder.state_dict(), model_path)
        
        # Save configuration for HuggingFace Hub compatibility
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            config_dict = {k: v for k, v in self.config.__dict__.items()}
            # Drop the non-serializable device object
            config_dict.pop('device', None) 
            json.dump(config_dict, f, indent=2)

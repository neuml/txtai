"""
LEMUR module
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


class Lemur:
    """LEMUR (Learned Multi-Vector Retrieval)."""

    def __init__(self, config=None):
        """Initializes Lemur with configuration."""
        config = config or LemurConfig()

        self.hidden_dim = config.hidden_dim
        self.n_train_tokens = config.n_train_tokens
        self.n_train_docs = config.n_train_docs
        self.n_ols_tokens = config.n_ols_tokens
        self.epochs = config.epochs
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.grad_clip = config.grad_clip
        self.seed = config.seed
        self.device = config.device or self._device()

        self.encoder = None
        self.vectors = None
        self._out_mean = None
        self._out_std = None

    def __call__(self, documents, category):
        """Main entry point."""
        if category == "data":
            return self._index(documents)
        if category == "query":
            return self._encode_query(documents)
        raise ValueError(f"Invalid category: {category}")

    def maxsim(self, query, doc):
        """Computes MaxSim similarity."""
        scores = query @ doc.T
        return float(scores.max(axis=1).sum())

    def search(self, query, documents, k=10):
        """Search top-k documents."""
        psi_query = self._encode_query([query])[0]
        approx_scores = self.vectors @ psi_query

        k_prime = min(5 * k, len(documents))
        candidate_idx = np.argpartition(approx_scores, -k_prime)[-k_prime:]

        reranked = [(int(idx), self.maxsim(query, documents[idx].astype(np.float32))) for idx in candidate_idx]

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]

    def _index(self, documents):
        """Indexes documents."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        n_docs = len(documents)
        d = documents[0].shape[1]

        m_prime = min(self.n_train_docs, n_docs)
        target_idx = rng.choice(n_docs, size=m_prime, replace=False)
        target_docs = [documents[i] for i in target_idx]

        all_tokens = np.vstack(documents).astype(np.float32)
        n_total = all_tokens.shape[0]

        n_sample = min(self.n_train_tokens, n_total)
        train_tokens = all_tokens[rng.choice(n_total, size=n_sample, replace=False)]

        targets = self._targets(train_tokens, target_docs)

        self._out_mean = targets.mean(axis=0)
        self._out_std = targets.std(axis=0) + 1e-8
        targets_norm = (targets - self._out_mean) / self._out_std

        self.encoder = self._train(train_tokens, targets_norm, d, m_prime)
        self.encoder.eval()

        n_ols = min(self.n_ols_tokens, n_total)
        ols_tokens = all_tokens[rng.choice(n_total, size=n_ols, replace=False)]

        phi = self._encode(ols_tokens)

        self.vectors = np.zeros((n_docs, self.hidden_dim), dtype=np.float32)

        for j, doc in enumerate(documents):
            doc = doc.astype(np.float32)
            g_j = self._maxsim_per_token(ols_tokens, doc)
            self.vectors[j] = self._ols(phi, g_j)

        return self.vectors

    def _encode_query(self, documents):
        """Encodes queries."""
        vectors = np.zeros((len(documents), self.hidden_dim), dtype=np.float32)

        for i, query in enumerate(documents):
            tokens = query.astype(np.float32)
            psi = self._encode(tokens)
            vectors[i] = psi.sum(axis=0)

        return vectors

    def _train(self, tokens, targets, d, m_prime):
        """Trains model."""
        model = LemurMLP(d, self.hidden_dim, m_prime).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        x_tensor = torch.from_numpy(tokens).to(self.device)
        y_tensor = torch.from_numpy(targets).to(self.device)

        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model.train()
        for _ in range(self.epochs):  # Fix W0612: renamed unused 'epoch' to '_'
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

        return model.encoder

    def _targets(self, tokens, target_docs):
        """Computes targets."""
        targets = np.zeros((tokens.shape[0], len(target_docs)), dtype=np.float32)

        for j, doc in enumerate(target_docs):
            targets[:, j] = self._maxsim_per_token(tokens, doc.astype(np.float32))

        return targets

    def _maxsim_per_token(self, tokens, doc):
        """Per-token MaxSim."""
        return (tokens @ doc.T).max(axis=1)

    @torch.no_grad()
    def _encode(self, tokens):
        """Encodes tokens."""
        x_tensor = torch.from_numpy(tokens).to(self.device)
        return self.encoder(x_tensor).cpu().numpy()

    def _ols(self, phi, g_j):
        """Solves ridge regression."""
        lam = 1e-4
        a_mat = phi.T @ phi + lam * np.eye(self.hidden_dim)
        b_vec = phi.T @ g_j
        return np.linalg.solve(a_mat, b_vec).astype(np.float32)

    def _device(self):
        """Selects best device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

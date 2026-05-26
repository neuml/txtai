"""
LEMUR module
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from safetensors.torch import load_file


class Lemur:
    """LEMUR (Learned Multi-Vector Retrieval) Inference Engine."""

    def __init__(self, model_path=None):
        """Initializes Lemur by loading pre-trained weights."""
        if not model_path or not os.path.exists(model_path):
            raise ValueError("A valid model_path containing safetensors and config is required.")

        self.device = self._device()

        # 1. Load the Configuration
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.hidden_dim = self.config.get("hidden_dim", 2048)

        # 2. Load the Safetensors Weights into the Feature Encoder
        weights_file = os.path.join(model_path, "model.safetensors")
        
        state_dict = load_file(weights_file)
        input_dim = state_dict["linear.weight"].shape[1]
        
        # LOCAL IMPORT TO BREAK CIRCULAR DEPENDENCY
        from txtai.pipeline.train.lemur import FeatureEncoder
        
        self.encoder = FeatureEncoder(input_dim, self.hidden_dim).to(self.device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

        self.vectors = None

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
        """Builds the ridge regression vectors using the pre-trained encoder."""
        n_docs = len(documents)
        all_tokens = np.vstack(documents).astype(np.float32)
        n_total = all_tokens.shape[0]

        # Use the ols_tokens parameter from the loaded config
        n_ols = min(self.config.get("n_ols_tokens", 16384), n_total)
        
        rng = np.random.default_rng(self.config.get("seed", 42))
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

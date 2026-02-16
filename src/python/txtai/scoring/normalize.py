"""
Normalize module
"""

import numpy as np


class Normalize:
    """
    Applies score normalization methods.

    Bayesian mode supports BB25-style score calibration aliases ("bayes", "bb25", "bayesian-bm25").
    Reference implementations:
      - https://github.com/instructkr/bb25
      - https://github.com/cognica-io/bayesian-bm25
    """

    def __init__(self, config):
        """
        Creates a new Normalize instance.

        Args:
            config: normalize configuration
        """

        # Normalize settings
        self.config = config if isinstance(config, dict) else {}
        method = self.config.get("method", config if isinstance(config, str) else "default")
        self.method = str(method).lower()

        # Bayesian settings
        self.alpha = float(self.config.get("alpha", 1.0))
        self.beta = self.config.get("beta")

        if self.beta is not None:
            self.beta = float(self.beta)

    def __call__(self, scores, avgscore):
        """
        Normalizes scores.

        Args:
            scores: list of (id, score)
            avgscore: average score across index

        Returns:
            normalized scores
        """

        # BB25-compatible aliases for Bayesian normalization mode.
        bayesian = ("bayes", "bayesian", "bayesian-bm25", "bb25")
        return self.bayes(scores) if self.method in bayesian else self.default(scores, avgscore)

    def default(self, scores, avgscore):
        """
        Default normalization implementation.

        Args:
            scores: list of (id, score)
            avgscore: average score across index

        Returns:
            normalized scores
        """

        # Use average index score in max score calculation
        maxscore = min(scores[0][1] + avgscore, 6 * avgscore)

        # Normalize scores between 0 - 1 using maxscore
        return [(uid, min(score / maxscore, 1.0)) for uid, score in scores]

    def bayes(self, scores):
        """
        Bayesian normalization implementation.

        Args:
            scores: list of (id, score)

        Returns:
            normalized scores
        """

        # Extract candidate scores
        values = np.array([score for _, score in scores], dtype=np.float32)

        # Dynamically derive beta using candidate score distribution, if not configured
        beta = self.beta if self.beta is not None else float(np.median(values))

        # Scale alpha by standard deviation for score-range invariance
        std = float(np.std(values))
        alpha = abs(self.alpha / std if std > 0 else self.alpha)

        # Transform to posterior probabilities in [0, 1]
        logits = np.clip(alpha * (values - beta), -60, 60)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        probabilities = np.clip(probabilities, 0.0, 1.0)

        # Convert back to score tuples
        return [(uid, float(probabilities[x])) for x, (uid, _) in enumerate(scores)]

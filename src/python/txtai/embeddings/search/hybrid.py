"""
Hybrid module
"""

import math


# Numerical clamp for log-odds computation
_EPSILON = 1e-10


class Hybrid:
    """
    Hybrid score fusion strategies for combining dense and sparse search results.

    Selects a fusion method based on the sparse scoring configuration:
      - Log-odds conjunction for Bayesian (BB25) normalized scores
      - Convex combination for default normalized scores
      - Reciprocal Rank Fusion (RRF) for unnormalized scores
    """

    def __init__(self, scoring):
        """
        Creates a new Hybrid instance.

        Args:
            scoring: sparse scoring instance
        """

        if scoring.isbayes():
            self.method = self.logodds
        elif scoring.isnormalized():
            self.method = self.convex
        else:
            self.method = self.rrf

    def __call__(self, vectors, weights, limit):
        """
        Fuses dense and sparse result vectors into a single ranked list.

        Args:
            vectors: tuple of (dense_results, sparse_results)
            weights: [dense_weight, sparse_weight]
            limit: maximum results

        Returns:
            sorted list of (uid, score)
        """

        return self.method(vectors, weights, limit)

    def calibrate(self, dense_raw):
        """
        Computes per-query calibration parameters for dense cosine scores.

        Uses the same approach as BB25: beta=median, alpha_eff=1/std so the
        logit for a dense score is alpha * (score - median), centering the
        median candidate at logit 0.

        Args:
            dense_raw: list of raw dense cosine scores

        Returns:
            (median, alpha) calibration parameters
        """

        d_median, d_alpha = 0.0, 1.0

        dense_arr = [s for s in dense_raw if s > 0]
        if dense_arr:
            d_median = sorted(dense_arr)[len(dense_arr) // 2]
            d_std = (sum((x - sum(dense_arr) / len(dense_arr)) ** 2 for x in dense_arr) / len(dense_arr)) ** 0.5
            d_alpha = 1.0 / d_std if d_std > 0 else 1.0

        return d_median, d_alpha

    def logodds(self, vectors, weights, limit):
        """
        Log-odds conjunction fusion for Bayesian (BB25) normalized scores.

        Implements the framework from "From Bayesian Inference to Neural Computation"
        (Jeong, 2026) with asymmetric dynamic calibration:

          1. Calibrate dense cosine scores via per-query dynamic sigmoid
             (beta=median, alpha_eff=1/std) to produce logits centered at 0.
          2. Convert sparse BB25 probabilities to logits.
          3. Fuse via weighted mean log-odds with confidence scaling.

        Scores are returned as raw logits (not mapped back through sigmoid) to
        preserve ranking resolution among top candidates.

        Args:
            vectors: tuple of (dense_results, sparse_results)
            weights: [dense_weight, sparse_weight]
            limit: maximum results

        Returns:
            sorted list of (uid, score) where score is a fused logit
        """

        # Phase 1: Collect raw scores per document
        uids = {}
        dense_raw = []
        for v, scores in enumerate(vectors):
            for uid, score in scores if weights[v] > 0 else []:
                if uid not in uids:
                    uids[uid] = [None, None]

                if v == 0:
                    uids[uid][0] = score
                    dense_raw.append(score)
                else:
                    # Sparse BB25 score: already a calibrated probability
                    uids[uid][1] = score

        # Phase 2: Compute per-query calibration parameters for dense cosine scores.
        # Same approach as BB25: beta=median, alpha_eff=1/std. The logit for a dense
        # score is alpha * (score - median), centering the median candidate at logit 0.
        d_median, d_alpha = self.calibrate(dense_raw)

        # Phase 3: Fuse via weighted mean log-odds with confidence scaling.
        # Raw logit scores are used for ranking instead of sigmoid(logit) to
        # preserve fine-grained ordering among top candidates.
        fused = {}
        n = 2
        alpha = 0.5
        scale = n**alpha

        for uid, pair in uids.items():
            raw_dense = pair[0]
            p_sparse = pair[1]

            if raw_dense is not None and p_sparse is not None:
                # Calibrate dense score via dynamic sigmoid
                logit_d = d_alpha * (raw_dense - d_median)
                logit_d = max(min(logit_d, 500), -500)

                # Sparse BB25 score -> logit
                p_s = min(max(p_sparse, _EPSILON), 1.0 - _EPSILON)
                logit_s = math.log(p_s / (1.0 - p_s))

                # Weighted mean log-odds with confidence scaling (Paper 2, Def 4.2.1)
                l_bar = weights[0] * logit_d + weights[1] * logit_s
                fused[uid] = l_bar * scale
            elif raw_dense is not None:
                # Only dense signal: calibrated logit scaled by weight
                logit_d = d_alpha * (raw_dense - d_median)
                logit_d = max(min(logit_d, 500), -500)
                fused[uid] = logit_d * weights[0]
            else:
                # Only sparse signal: logit scaled by weight
                p_s = min(max(p_sparse, _EPSILON), 1.0 - _EPSILON)
                fused[uid] = math.log(p_s / (1.0 - p_s)) * weights[1]

        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:limit]

    def convex(self, vectors, weights, limit):
        """
        Convex combination fusion for default normalized scores.

        Args:
            vectors: tuple of (dense_results, sparse_results)
            weights: [dense_weight, sparse_weight]
            limit: maximum results

        Returns:
            sorted list of (uid, score)
        """

        uids = {}
        for v, scores in enumerate(vectors):
            for uid, score in scores if weights[v] > 0 else []:
                if uid not in uids:
                    uids[uid] = 0.0
                uids[uid] += score * weights[v]

        return sorted(uids.items(), key=lambda x: x[1], reverse=True)[:limit]

    def rrf(self, vectors, weights, limit):
        """
        Reciprocal Rank Fusion for unnormalized scores.

        Args:
            vectors: tuple of (dense_results, sparse_results)
            weights: [dense_weight, sparse_weight]
            limit: maximum results

        Returns:
            sorted list of (uid, score)
        """

        uids = {}
        for v, scores in enumerate(vectors):
            for r, (uid, _) in enumerate(scores if weights[v] > 0 else []):
                if uid not in uids:
                    uids[uid] = 0.0
                uids[uid] += (1.0 / (r + 1)) * weights[v]

        return sorted(uids.items(), key=lambda x: x[1], reverse=True)[:limit]

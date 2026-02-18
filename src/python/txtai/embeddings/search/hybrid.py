"""
Hybrid module
"""

import math


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

    def logodds(self, vectors, weights, limit):
        """
        Log-odds conjunction fusion for Bayesian (BB25) normalized scores.

        Args:
            vectors: tuple of (dense results, sparse results)
            weights: [dense weight, sparse weight]
            limit: maximum results

        Returns:
            sorted list (uid, score)
        """

        return LogOdds()(vectors, weights, limit)

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


class LogOdds:
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
    """

    # Numerical clamp for log-odds computation
    EPSILON = 1e-10

    def __call__(self, vectors, weights, limit):
        """
        Log-odds fusion.

        Args:
            vectors: tuple of (dense vectors, sparse vectors)
            weights: [dense weights, sparse weights]
            limit: maximum results

        Returns:
            sorted list of (uid, score) where score is a fused logit
        """

        # Phase 1: Collect raw scores per document
        uids, denseraw = self.rawscores(vectors, weights)

        # Phase 2: Compute per-query calibration parameters for dense cosine scores.
        # Same approach as BB25: beta=median, alpha_eff=1/std. The logit for a dense
        # score is alpha * (score - median), centering the median candidate at logit 0.
        densemedian, densealpha = self.calibrate(denseraw)

        # Phase 3: Fuse via weighted mean log-odds with confidence scaling.
        # Raw logit scores are used for ranking instead of sigmoid(logit) to
        # preserve fine-grained ordering among top candidates.
        fused = self.fuse(uids, weights, densemedian, densealpha)

        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:limit]

    def rawscores(self, vectors, weights):
        """
        Collects raw scores.

        Args:
            vectors: tuple of (dense vectors, sparse vectors)
            weights: [dense weights, sparse weights]

        Returns:
            (uids, raw dense scores)
        """

        uids, denseraw = {}, []
        for v, scores in enumerate(vectors):
            for uid, score in scores if weights[v] > 0 else []:
                if uid not in uids:
                    uids[uid] = [None, None]

                if v == 0:
                    uids[uid][0] = score
                    denseraw.append(score)
                else:
                    # Sparse BB25 score: already a calibrated probability
                    uids[uid][1] = score

        return uids, denseraw

    def calibrate(self, raw):
        """
        Computes per-query calibration parameters for dense cosine scores.

        Uses the same approach as BB25: beta=median, alpha_eff=1/std so the
        logit for a dense score is alpha * (score - median), centering the
        median candidate at logit 0.

        Args:
            raw: list of raw dense cosine scores

        Returns:
            (median, alpha) calibration parameters
        """

        median, alpha = 0.0, 1.0

        array = [s for s in raw if s > 0]
        if array:
            median = sorted(array)[len(array) // 2]
            std = (sum((x - sum(array) / len(array)) ** 2 for x in array) / len(array)) ** 0.5
            alpha = 1.0 / std if std > 0 else 1.0

        return median, alpha

    def fuse(self, uids, weights, densemedian, densealpha):
        """
        Fuses scores together.

        Args:
            uids: result ids
            weights: [dense weights, sparse weights]
            densemedian: dense median
            densealpha: dense alpha

        Returns:
            fused scores
        """

        fused, n, alpha = {}, 2, 0.5
        scale = n**alpha

        for uid, pair in uids.items():
            # Unpair scores
            rawdense, psparse = pair

            if rawdense is not None and psparse is not None:
                # Calibrate dense score via dynamic sigmoid
                logitdense = densealpha * (rawdense - densemedian)
                logitdense = max(min(logitdense, 500), -500)

                # Sparse BB25 score -> logit
                psparse = min(max(psparse, LogOdds.EPSILON), 1.0 - LogOdds.EPSILON)
                logitsparse = math.log(psparse / (1.0 - psparse))

                # Weighted mean log-odds with confidence scaling (Paper 2, Def 4.2.1)
                lbar = weights[0] * logitdense + weights[1] * logitsparse
                fused[uid] = lbar * scale

            elif rawdense is not None:
                # Only dense signal: calibrated logit scaled by weight
                logitdense = densealpha * (rawdense - densemedian)
                logitdense = max(min(logitdense, 500), -500)
                fused[uid] = logitdense * weights[0]

            else:
                # Only sparse signal: logit scaled by weight
                psparse = min(max(psparse, LogOdds.EPSILON), 1.0 - LogOdds.EPSILON)
                fused[uid] = math.log(psparse / (1.0 - psparse)) * weights[1]

        return fused

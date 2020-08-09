"""
Scoring module
"""

import math
import pickle

from collections import Counter

from .tokenizer import Tokenizer

class Scoring(object):
    """
    Base scoring object. Default method scores documents using TF-IDF.
    """

    @staticmethod
    def create(method):
        """
        Factory method to construct a Scoring object.

        Args:
            method: scoring method (bm25, sif, tfidf)

        Returns:
            Scoring object
        """

        if method == "bm25":
            return BM25()
        elif method == "sif":
            return SIF()
        elif method == "tfidf":
            # Default scoring object implements tf-idf
            return Scoring()

        return None

    def __init__(self):
        """
        Initializes backing statistic objects.
        """

        # Document stats
        self.total = 0
        self.tokens = 0
        self.avgdl = 0

        # Word frequency
        self.docfreq = Counter()
        self.wordfreq = Counter()
        self.avgfreq = 0

        # IDF index
        self.idf = {}
        self.avgidf = 0

        # Tag boosting
        self.tags = Counter()

    def index(self, documents):
        """
        Indexes a collection of documents using a scoring method. Documents are tuples of (id, text|tokens, tags).

        Args:
            documents: input documents
        """

        # Calculate word frequency, total tokens and total documents
        for _, tokens, tags in documents:
            # Convert to tokens if necessary
            if isinstance(tokens, str):
                tokens = Tokenizer.tokenize(tokens)

            # Total number of times token appears, count all tokens
            self.wordfreq.update(tokens)

            # Total number of documents a token is in, count unique tokens
            self.docfreq.update(set(tokens))

            # Get list of unique tags
            if tags:
                self.tags.update(tags.split())

            # Total document count
            self.total += 1

        # Calculate total token frequency
        self.tokens = sum(self.wordfreq.values())

        # Calculate average frequency per token
        self.avgfreq = self.tokens / len(self.wordfreq.values())

        # Calculate average document length in tokens
        self.avgdl = self.tokens / self.total

        # Compute IDF scores
        for word, freq in self.docfreq.items():
            self.idf[word] = self.computeIDF(freq)

        # Average IDF score per token
        self.avgidf = sum(self.idf.values()) / len(self.idf)

        # Filter for tags that appear in at least 1% of the documents
        self.tags = {tag:number for tag, number in self.tags.items() if number >= self.total * 0.005}

    def weights(self, document):
        """
        Builds weight vector for each token in the input token.

        Args:
            document: (id, tokens, tags)

        Returns:
            list of weights for each token
        """

        # Weights array
        weights = []

        # Unpack document
        _, tokens, _ = document

        # Document length
        length = len(tokens)

        for token in tokens:
            # Lookup frequency and idf score - default to averages if not in repository
            freq = self.wordfreq[token] if token in self.wordfreq else self.avgfreq
            idf = self.idf[token] if token in self.idf else self.avgidf

            # Calculate score for each token, use as weight
            weights.append(self.score(freq, idf, length))

        # Boost weights of tag tokens to match the largest weight in the list
        if self.tags:
            tags = {token: self.tags[token] for token in tokens if token in self.tags}
            if tags:
                maxWeight = max(weights)
                maxTag = max(tags.values())

                weights = [max(maxWeight * (tags[tokens[x]] / maxTag), weight)
                           if tokens[x] in tags else weight for x, weight in enumerate(weights)]

        return weights

    def load(self, path):
        """
        Loads a saved Scoring object from path.

        Args:
            path: directory path to load model
        """

        with open("%s/scoring" % path, "rb") as handle:
            self.__dict__.update(pickle.load(handle))

    def save(self, path):
        """
        Saves a Scoring object to path.

        Args:
            path: directory path to save model
        """

        with open("%s/scoring" % path, "wb") as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def computeIDF(self, freq):
        """
        Computes an idf score for word frequency.

        Args:
            freq: word frequency

        Returns:
            idf score
        """

        return math.log(self.total / (1 + freq))

    # pylint: disable=W0613
    def score(self, freq, idf, length):
        """
        Calculates a score for each token.

        Args:
            freq: token frequency
            idf: token idf score
            length: total number of tokens in source document

        Returns:
            token score
        """

        return idf

class BM25(Scoring):
    """
    BM25 scoring. Scores using Apache Lucene's version of BM25 which adds 1 to prevent
    negative scores.
    """

    def __init__(self, k1=0.1, b=0.75):
        super(BM25, self).__init__()

        # BM25 configurable parameters
        self.k1 = k1
        self.b = b

    def computeIDF(self, freq):
        # Calculate BM25 IDF score
        return math.log(1 + (self.total - freq + 0.5)/(freq + 0.5))

    def score(self, freq, idf, length):
        # Calculate BM25 score
        k = self.k1 * ((1 - self.b) + self.b * length / self.avgdl)
        return idf * (freq * (self.k1 + 1)) / (freq + k)

class SIF(Scoring):
    """
    Smooth Inverse Frequency (SIF) scoring.
    """

    def __init__(self, a=0.001):
        super(SIF, self).__init__()

        # SIF configurable parameters
        self.a = a

    def score(self, freq, idf, length):
        # Calculate SIF score
        return self.a / (self.a + freq/self.tokens)

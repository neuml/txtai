"""
Scoring module
"""

import math
import pickle

from collections import Counter

from .. import __pickle__

from ..pipeline import Tokenizer


class Scoring:
    """
    Base scoring. Uses term frequency-inverse document frequency (TF-IDF).
    """

    def __init__(self, config=None):
        """
        Initializes backing statistic objects.

        Args:
            config: input configuration
        """

        # Scoring configuration
        self.config = config if config else {}

        # Document stats
        self.total = 0
        self.tokens = 0
        self.avgdl = 0

        # Document data
        self.documents = {} if self.config.get("content") else None
        self.docterms = {} if self.config.get("terms") else None

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
        Indexes a collection of documents using a scoring method.

        Args:
            documents: list of (id, dict|text|tokens, tags)
        """

        # Parse documents
        tokenlists = self.parse(documents)

        # Build index if tokens parsed
        if self.wordfreq:
            # Calculate total token frequency
            self.tokens = sum(self.wordfreq.values())

            # Calculate average frequency per token
            self.avgfreq = self.tokens / len(self.wordfreq.values())

            # Calculate average document length in tokens
            self.avgdl = self.tokens / self.total

            # Compute IDF scores
            for word, freq in self.docfreq.items():
                self.idf[word] = self.computeidf(freq)

            # Average IDF score per token
            self.avgidf = sum(self.idf.values()) / len(self.idf)

            # Filter for tags that appear in at least 1% of the documents
            self.tags = {tag: number for tag, number in self.tags.items() if number >= self.total * 0.005}

            # Process document terms, if necessary
            self.terms(tokenlists)

    def parse(self, documents):
        """
        Parses document stats, word frequencies and data from documents.

        Args:
            docments: list of (id, dict|text|tokens, tags)

        Returns:
            tokenlists: list of parsed documents when term parsing is enabled, empty dict otherwise
        """

        # Store tokenlists when term indexing enabled
        tokenlists = {}

        # Calculate word frequency, total tokens and total documents
        for uid, data, tags in documents:
            if self.documents is not None:
                self.documents[uid] = data

            # Extract text, if necessary
            if isinstance(data, dict):
                data = data.get("text")

            # Convert to tokens, if necessary
            tokens = Tokenizer.tokenize(data) if isinstance(data, str) else data

            # Save tokens for term indexing
            if self.docterms is not None:
                tokenlists[uid] = tokens

            # Total number of times token appears, count all tokens
            self.wordfreq.update(tokens)

            # Total number of documents a token is in, count unique tokens
            self.docfreq.update(set(tokens))

            # Get list of unique tags
            if tags:
                self.tags.update(tags.split())

            # Total document count
            self.total += 1

        return tokenlists

    def terms(self, tokenlist):
        """
        Add term weights for each tokenized document in tokenlist.

        Args:
            tokenlist: list of tokenized documents
        """

        for uid in tokenlist:
            tokens = tokenlist[uid]
            weights = self.weights(tokens)

            for x, token in enumerate(tokens):
                if token not in self.docterms:
                    self.docterms[token] = {}

                self.docterms[token][uid] = weights[x]

    def weights(self, tokens):
        """
        Builds weight vector for each token in the input token.

        Args:
            tokens: input tokens

        Returns:
            list of weights for each token
        """

        # Weights array
        weights = []

        # Document length
        length = len(tokens)

        # Calculate token counts
        freq = self.computefreq(tokens)

        for token in tokens:
            # Lookup idf score
            idf = self.idf[token] if token in self.idf else self.avgidf

            # Calculate score for each token, use as weight
            weights.append(self.score(freq[token], idf, length))

        # Boost weights of tag tokens to match the largest weight in the list
        if self.tags:
            tags = {token: self.tags[token] for token in tokens if token in self.tags}
            if tags:
                maxWeight = max(weights)
                maxTag = max(tags.values())

                weights = [max(maxWeight * (tags[tokens[x]] / maxTag), weight) if tokens[x] in tags else weight for x, weight in enumerate(weights)]

        return weights

    def search(self, query, limit=3):
        """
        Search index for documents matching query. Request terms config parameter to be enabled.

        Args:
            query: input query
            limit: maximum results

        Returns:
            list of (id, score) or (data, score) if content is enabled
        """

        # Check if document terms available
        if self.docterms:
            query = Tokenizer.tokenize(query) if isinstance(query, str) else query

            scores = {}
            for token in query:
                if token in self.docterms:
                    for x in self.docterms[token]:
                        if x not in scores:
                            scores[x] = 0.0

                        scores[x] += self.docterms[token][x]

            # Sort and get topn results
            topn = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

            # Format results and add content if available
            return self.results(topn)

        return None

    def results(self, topn):
        """
        Resolves a list of (id, score) with document content, if available. Otherwise, the original input is returned.

        Args:
            topn: list of (id, score)

        Returns:
            resolved results
        """

        if self.documents:
            results = []
            for x, score in topn:
                data = self.documents[x]
                if isinstance(data, dict):
                    results.append({"id": x, "text": data.get("text"), "score": score, "data": data})
                else:
                    results.append({"id": x, "text": data, "score": score})

            return results

        return topn

    def count(self):
        """
        Returns the total number of documents indexed.

        Returns:
            total number of documents indexed
        """

        return self.total

    def load(self, path):
        """
        Loads a saved Scoring object from path.

        Args:
            path: directory path to load scoring index
        """

        with open(path, "rb") as handle:
            self.__dict__.update(pickle.load(handle))

    def save(self, path):
        """
        Saves a Scoring object to path.

        Args:
            path: directory path to save scoring index
        """

        with open(path, "wb") as handle:
            pickle.dump(self.__dict__, handle, protocol=__pickle__)

    def computefreq(self, tokens):
        """
        Computes token frequency.

        Args:
            tokens: input tokens

        Returns:
            {token: count}
        """

        return Counter(tokens)

    def computeidf(self, freq):
        """
        Computes an idf score for word frequency.

        Args:
            freq: word frequency

        Returns:
            idf score
        """

        return math.log((self.total + 1) / (freq + 1)) + 1

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

        return idf * math.sqrt(freq) * (1 / math.sqrt(length))

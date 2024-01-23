"""
Topics module
"""

from ..pipeline import Tokenizer
from ..scoring import ScoringFactory


class Topics:
    """
    Topic modeling using community detection.
    """

    def __init__(self, config):
        """
        Creates a new Topics instance.

        Args:
            config: topic configuration
        """

        self.config = config if config else {}
        self.tokenizer = Tokenizer(stopwords=True)

        # Additional stopwords to ignore when building topic names
        self.stopwords = set()
        if "stopwords" in self.config:
            self.stopwords.update(self.config["stopwords"])

    def __call__(self, graph):
        """
        Runs topic modeling for input graph.

        Args:
            graph: Graph instance

        Returns:
            dictionary of {topic name: [ids]}
        """

        # Detect communities
        communities = graph.communities(self.config)

        # Sort by community size, largest to smallest
        communities = sorted(communities, key=len, reverse=True)

        # Calculate centrality of graph
        centrality = graph.centrality()

        # Score communities and generate topn terms
        topics = [self.score(graph, x, community, centrality) for x, community in enumerate(communities)]

        # Merge duplicate topics and return
        return self.merge(topics)

    def score(self, graph, index, community, centrality):
        """
        Scores a community of nodes and generates the topn terms in the community.

        Args:
            graph: Graph instance
            index: community index
            community: community of nodes
            centrality: node centrality scores

        Returns:
            (topn topic terms, topic ids sorted by score descending)
        """

        # Tokenize input and build scoring index
        scoring = ScoringFactory.create({"method": self.config.get("labels", "bm25"), "terms": True})
        scoring.index(((node, self.tokenize(graph, node), None) for node in community))

        # Check if scoring index has data
        if scoring.idf:
            # Sort by most commonly occurring terms (i.e. lowest score)
            idf = sorted(scoring.idf, key=scoring.idf.get)

            # Term count for generating topic labels
            topn = self.config.get("terms", 4)

            # Get topn terms
            terms = self.topn(idf, topn)

            # Sort community by score descending
            community = [uid for uid, _ in scoring.search(terms, len(community))]
        else:
            # No text found for topic, generate topic name
            terms = ["topic", str(index)]

            # Sort community by centrality scores
            community = sorted(community, key=lambda x: centrality[x], reverse=True)

        return (terms, community)

    def tokenize(self, graph, node):
        """
        Tokenizes node text.

        Args:
            graph: Graph instance
            node: node id

        Returns:
            list of node tokens
        """

        text = graph.attribute(node, "text")
        return self.tokenizer(text) if text else []

    def topn(self, terms, n):
        """
        Gets topn terms.

        Args:
            terms: list of terms
            n: topn

        Returns:
            topn terms
        """

        topn = []

        for term in terms:
            # Add terms that pass tokenization rules
            if self.tokenizer(term) and term not in self.stopwords:
                topn.append(term)

            # Break once topn terms collected
            if len(topn) == n:
                break

        return topn

    def merge(self, topics):
        """
        Merges duplicate topics

        Args:
            topics: list of (topn terms, topic ids)

        Returns:
            dictionary of {topic name:[ids]}
        """

        merge, termslist = {}, {}

        for terms, uids in topics:
            # Use topic terms as key
            key = frozenset(terms)

            # Add key to merged topics, if necessary
            if key not in merge:
                merge[key], termslist[key] = [], terms

            # Merge communities
            merge[key].extend(uids)

        # Sort communities largest to smallest since the order could have changed with merges
        results = {}
        for k, v in sorted(merge.items(), key=lambda x: len(x[1]), reverse=True):
            # Create composite string key using topic terms and store ids
            results["_".join(termslist[k])] = v

        return results

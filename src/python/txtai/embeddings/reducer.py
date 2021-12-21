"""
Reducer module
"""

import pickle

# Conditionally import dimensionality reduction libraries as they aren't installed by default
try:
    from sklearn.decomposition import TruncatedSVD

    REDUCER = True
except ImportError:
    REDUCER = False


class Reducer:
    """
    LSA dimensionality reduction model
    """

    def __init__(self, embeddings=None, components=None):
        """
        Creates a dimensionality reduction model.

        Args:
            embeddings: input embeddings matrix
            components: number of model components
        """

        if not REDUCER:
            raise ImportError('Dimensionality reduction is not available - install "similarity" extra to enable')

        self.model = self.build(embeddings, components) if embeddings is not None and components else None

    def __call__(self, embeddings):
        """
        Applies a dimensionality reduction model to embeddings, removed the top n principal components. Operation applied
        directly on array.

        Args:
            embeddings: input embeddings matrix
        """

        pc = self.model.components_
        factor = embeddings.dot(pc.transpose())

        # Apply LSA model
        # Calculation is different if n_components = 1
        if pc.shape[0] == 1:
            embeddings -= factor * pc
        elif len(embeddings.shape) > 1:
            # Apply model on a row-wise basis to limit memory usage
            for x in range(embeddings.shape[0]):
                embeddings[x] -= factor[x].dot(pc)
        else:
            # Single embedding
            embeddings -= factor.dot(pc)

    def build(self, embeddings, components):
        """
        Builds a LSA model. This model is used to remove the principal component within embeddings. This helps to
        smooth out noisy embeddings (common words with less value).

        Args:
            embeddings: input embeddings matrix
            components: number of model components

        Returns:
            LSA model
        """

        model = TruncatedSVD(n_components=components, random_state=0)
        model.fit(embeddings)

        return model

    def load(self, path):
        """
        Loads a Reducer object from path.

        Args:
            path: directory path to load model
        """

        # Dimensionality reduction
        with open(path, "rb") as handle:
            self.model = pickle.load(handle)

    def save(self, path):
        """
        Saves a Reducer object to path.

        Args:
            path: directory path to save model
        """

        with open(path, "wb") as handle:
            pickle.dump(self.model, handle, protocol=4)

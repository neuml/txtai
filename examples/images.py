"""
Builds a similarity index for a directory of images

Requires streamlit to be installed.
  pip install streamlit
"""

import glob
import os
import sys

import streamlit as st

from PIL import Image

from txtai.embeddings import Embeddings


class Application:
    """
    Main application
    """

    def __init__(self, directory):
        """
        Creates a new application.

        Args:
            directory: directory of images
        """

        self.embeddings = self.build(directory)

    def build(self, directory):
        """
        Builds an image embeddings index.

        Args:
            directory: directory with images

        Returns:
            Embeddings index
        """

        embeddings = Embeddings({"method": "sentence-transformers", "path": "clip-ViT-B-32"})
        embeddings.index(self.images(directory))

        # Update model to support multilingual queries
        embeddings.config["path"] = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
        embeddings.model = embeddings.loadvectors()

        return embeddings

    def images(self, directory):
        """
        Generator that loops over each image in a directory.

        Args:
            directory: directory with images
        """

        for path in glob.glob(directory + "/*jpg") + glob.glob(directory + "/*png"):
            yield (path, Image.open(path), None)

    def run(self):
        """
        Runs a Streamlit application.
        """

        st.title("Image search")

        st.markdown("This application shows how images and text can be embedded into the same space to support similarity search. ")
        st.markdown(
            "[sentence-transformers](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/image-search) "
            + "recently added support for the [OpenAI CLIP model](https://github.com/openai/CLIP). This model embeds text and images into "
            + "the same space, enabling image similarity search. txtai can directly utilize these models."
        )

        query = st.text_input("Search query:")
        if query:
            index, _ = self.embeddings.search(query, 1)[0]
            st.image(Image.open(index))


@st.cache(allow_output_mutation=True)
def create(directory):
    """
    Creates and caches a Streamlit application.

    Args:
        directory: directory of images to index

    Returns:
        Application
    """

    return Application(directory)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create and run application
    app = create(sys.argv[1])
    app.run()

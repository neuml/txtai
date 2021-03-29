"""
Streamlit version of https://colab.research.google.com/github/neuml/txtai/blob/master/examples/13_Similarity_search_with_images.ipynb

Requires streamlit and torchvision to be installed.
  pip install streamlit torchvision
"""

import glob
import sys

import streamlit as st

from PIL import Image

from txtai.embeddings import Embeddings


def images(directory):
    """
    Generator that loops over each image in a directory.

    Args:
        directory: directory with images
    """

    for path in glob.glob(directory + "/*jpg") + glob.glob(directory + "/*png"):
        yield (path, Image.open(path), None)


@st.cache(allow_output_mutation=True)
def build(directory):
    """
    Builds an image embeddings index.

    Args:
        directory: directory with images

    Returns:
        Embeddings index
    """

    embeddings = Embeddings({"method": "transformers", "path": "clip-ViT-B-32", "modelhub": False})
    embeddings.index(images(directory))

    return embeddings


def app(directory):
    """
    Streamlit application that runs searches against an image embeddings index.

    Args:
        directory: directory with images
    """

    # Build embeddings index
    embeddings = build(directory)

    st.title("Image search")

    st.markdown("This application shows how images and text can be embedded into the same space to support similarity search. ")
    st.markdown(
        "[sentence-transformers](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/image-search) "
        + "recently added support for the [OpenAI CLIP model](https://github.com/openai/CLIP). This model embeds text and images into "
        + "the same space, enabling image similarity search. txtai can directly utilize these models."
    )

    query = st.text_input("Search query:")
    if query:
        index, _ = embeddings.search(query, 1)[0]
        st.image(Image.open(index))


if __name__ == "__main__":
    app(sys.argv[1])

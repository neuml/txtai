"""
Runs a RAG application backed by an Embeddings database.

Requires streamlit to be installed.
  pip install streamlit
"""

import os

import streamlit as st

from txtai import Embeddings, LLM

# Build logger
logger = st.logger.get_logger(__name__)


class Application:
    """
    RAG application
    """

    def __init__(self):
        """
        Creates a new application.
        """

        # Load Wikipedia model
        self.embeddings = Embeddings()
        self.embeddings.load(provider="huggingface-hub", container=os.environ.get("EMBEDDINGS", "neuml/txtai-wikipedia"))

        # Load LLM
        self.llm = LLM(os.environ.get("LLM", "TheBloke/Mistral-7B-OpenOrca-AWQ"))

        self.template = """<|im_start|>system
You are a friendly assistant. You answer questions from users.<|im_end|>
<|im_start|>user
Answer the following question using only the context below. Only include information
specifically discussed.

question: {question}
context: {context} <|im_end|>
<|im_start|>assistant
"""

    def run(self):
        """
        Runs a Streamlit application.
        """

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"role": "assistant", "content": 'Ask a question such as  "Who created Linux?"'}]

        if question := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": question})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Researching..."):
                    # Run RAG
                    response = self.rag(question)

                    # Render response
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    def rag(self, question):
        """
        Runs a RAG query.

        Args:
            question: question to ask

        Returns:
            answer
        """

        # Generate context
        context = "\n".join([x["text"] for x in self.embeddings.search(question)])

        # Build prompt
        prompt = self.template.format(question=question, context=context)
        logger.debug(prompt)

        # Run RAG
        return self.llm(prompt, maxlength=2048)


@st.cache_resource(show_spinner="Downloading models...")
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    st.set_page_config(page_title="RAG with txtai", page_icon="🚀", layout="centered", initial_sidebar_state="auto", menu_items=None)

    st.title(os.environ.get("TITLE", "Talk with Wikipedia 💬"))

    # Create and run RAG application
    app = create()
    app.run()

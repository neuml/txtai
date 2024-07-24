"""
Runs a RAG application backed by an Embeddings database.

Requires streamlit to be installed.
  pip install streamlit
"""

import os

from glob import glob
from io import BytesIO
from uuid import UUID

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from txtai import Embeddings, LLM, RAG
from txtai.pipeline import Textractor

# Build logger
logger = st.logger.get_logger(__name__)


class GraphContext:
    """
    Builds graph contexts for GraphRAG
    """

    def __init__(self, embeddings, context):
        """
        Creates a new GraphContext.

        Args:
            embeddings: embeddings instance
            context: number of records to use as context
        """

        self.embeddings = embeddings
        self.context = context

    def __call__(self, question):
        """
        Attempts to create a graph context for the input question. This method checks if:
          - Embeddings has a graph
          - Question is a graph query

        If both of the above are true, the graph is scanned to find the best matching records
        to use as a context.

        Args:
            question: input question

        Returns:
            question, [context]
        """

        query, concepts, context = self.parse(question)
        if self.embeddings.graph and (query or concepts):
            # Generate graph path query
            path = self.path(query, concepts)

            # Build graph network from path query
            graph = self.embeddings.graph.search(path, graph=True)
            if graph.count():
                # Draw and display graph
                response = self.plot(graph)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Build graph context
                context = [graph.attribute(node, "text") for node in list(graph.scan())]
                if context:
                    question = query if query else f"Write a title and text that must relate all of the following concepts: {concepts}"

        return question, context

    def parse(self, question):
        """
        Attempts to parse question as a graph query. This method will return either a query
        or concepts if this is a graph query. Otherwise, both will be None.

        Args:
            question: input question

        Returns:
            query, concepts, context
        """

        # Graph query prefix
        prefix = "gq: "

        # Parse graph query
        query, concepts, context = None, None, None
        if "->" in question or question.strip().lower().startswith(prefix):
            # Split into concepts
            concepts = [x.strip() for x in question.strip().lower().split("->")]

            # Parse query out of concepts, if necessary
            if prefix in concepts[-1]:
                query, concepts = concepts[-1], concepts[:-1]

                # Look for search prefix
                query = [x.strip() for x in query.split(prefix, 1)]

                # Add concept, if necessary
                if query[0]:
                    concepts.append(query[0])

                # Extract query, if present
                if len(query) > 1:
                    query = query[1]

        return query, concepts, context

    def path(self, question, concepts):
        """
        Creates a graph path query with one of two strategies.
          - If an array of concepts is provided, the best matching row is found for each graph node
          - Otherwise, the top 3 nodes running an embeddings search for query are used

        Each node is then joined together in as a Cypher MATCH PATH query and returned.

        Args:
            question: input question
            concepts: input concepts

        Returns:
            MATCH PATH query
        """

        # Find graph nodes
        ids = []
        if concepts:
            for concept in concepts:
                uid = self.embeddings.search(concept, 1)[0]["id"]
                ids.append(f'({{id: "{uid}"}})')
        else:
            for x in self.embeddings.search(question, 3):
                ids.append(f"({{id: \"{x['id']}\"}})")

        # Create graph path query
        ids = "-[*1..4]->".join(ids)
        query = f"MATCH P={ids} RETURN P LIMIT {self.context}"
        logger.debug(query)

        return query

    def plot(self, graph):
        """
        Plot graph as an image.

        Args:
            graph: input graph

        Returns:
            Image
        """

        labels = {}
        for x in graph.scan():
            uid, topic = graph.attribute(x, "id"), graph.attribute(x, "topic")
            labels[x] = topic if self.isautoid(uid) and topic else uid

        options = {
            "node_size": 700,
            "node_color": "#ffbd45",
            "edge_color": "#e9ecef",
            "font_color": "#454545",
            "font_size": 10,
            "alpha": 1.0,
        }

        # Draw graph
        _, ax = plt.subplots(figsize=(9, 5))
        pos = nx.spring_layout(graph.backend, seed=0, k=0.9, iterations=50)
        nx.draw_networkx(graph.backend, pos=pos, labels=labels, **options)

        # Disable axes and draw margins
        ax.axis("off")
        plt.margins(x=0.15)

        # Save and return image
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        return Image.open(buffer)

    def isautoid(self, uid):
        """
        Checks if uid is a UUID or numeric id.

        Args:
            uid: input id

        Returns:
            True if this is an autoid, False otherwise
        """

        # Check if this is a UUID
        try:
            return UUID(str(uid))
        except ValueError:
            pass

        # Return True if this is numeric, False otherwise
        return isinstance(uid, int) or uid.isdigit()


class Application:
    """
    RAG application
    """

    def __init__(self):
        """
        Creates a new application.
        """

        # Load LLM
        self.llm = LLM(os.environ.get("LLM", "TheBloke/Mistral-7B-OpenOrca-AWQ"))

        # Load embeddings
        self.embeddings = self.load()

        # Context size
        self.context = 10

        # Define prompt template
        template = """
Answer the following question using only the context below. Only include information
specifically discussed.

question: {question}
context: {context} """

        # Create RAG pipeline
        self.rag = RAG(
            self.embeddings,
            self.llm,
            system="You are a friendly assistant. You answer questions from users.",
            template=template,
            context=self.context,
        )

    def load(self):
        """
        Loads or creates a new Embeddings instance.

        Returns:
            Embeddings
        """

        embeddings = None

        # Check if a data directory parameter is set, if so build new embeddings
        data = os.environ.get("DATA")
        if data:
            embeddings = Embeddings(
                autoid="uuid5",
                path="intfloat/e5-large",
                instructions={"query": "query: ", "data": "passage: "},
                content=True,
                graph={
                    "approximate": False,
                },
            )
            embeddings.index(self.stream(data))

            # Create LLM-generated topics
            self.infertopics(embeddings)

        else:
            # Read embeddings path parameter
            path = os.environ.get("EMBEDDINGS", "neuml/txtai-wikipedia-slim")

            # Load existing model
            embeddings = Embeddings()
            if os.path.exists(path):
                embeddings.load(path)
            else:
                embeddings.load(provider="huggingface-hub", container=path)

        return embeddings

    def stream(self, data):
        """
        Runs a textractor pipeline and streams extracted content from a data directory.

        Args:
            data: input data directory
        """

        # Stream sections from content
        textractor = Textractor(paragraphs=True, minlength=50)
        for sections in textractor(glob(f"{data}/**/*", recursive=True)):
            yield from sections

    def infertopics(self, embeddings):
        """
        Traverses the graph associated with an embeddings instance and adds
        LLM-generated topics for each entry.

        Args:
            embeddings: embeddings database
        """

        batch = []
        for uid in tqdm(embeddings.graph.scan(), total=embeddings.graph.count()):
            text = embeddings.graph.attribute(uid, "text")

            batch.append((uid, text))
            if len(batch) == 32:
                self.topics(embeddings, batch)
                batch = []

        if batch:
            self.topics(embeddings, batch)

    def topics(self, embeddings, batch):
        """
        Generates a batch of topics with a LLM. Topics are set directly on the embeddings
        instance.

        Args:
            embeddings: embeddings database
            batch: batch of (id, text) elements
        """

        prompt = """
Create a simple, concise topic for the following text. Only return the topic name.

Text:
{text}"""

        # Build batch of prompts
        prompts = []
        for _, text in batch:
            prompts.append([{"role": "user", "content": prompt.format(text=text)}])

        # Check if batch processing is enabled
        topicsbatch = os.environ.get("TOPICSBATCH")
        kwargs = {"batch_size": int(topicsbatch)} if topicsbatch else {}

        # Run prompt batch and set topics
        for x, topic in enumerate(self.llm(prompts, maxlength=2048, **kwargs)):
            embeddings.graph.addattribute(batch[x][0], "topic", topic)

    def instructions(self):
        """
        Generates a welcome message with instructions.

        Returns:
            instructions
        """

        instructions = 'Ask a question such as  "Who created Linux?"'
        if self.embeddings.graph:
            instructions += (
                "\n\nThis index also supports `📈 graph rag`. Examples are shown below.\n"
                "- `gq: Tell me about Linux`\n"
                "  - Graph rag query, the `gq: ` prefix enables graph rag\n"
                "- `linux -> macos -> microsoft windows`\n"
                "  - Graph path query for a list of concepts separated by `->`\n"
                "  - The graph path is analyzed and described by the LLM\n"
                "- `linux -> macos -> microsoft windows gq: Tell me about Linux`\n"
                "  - Graph path with a graph rag query"
            )

        return instructions

    def run(self):
        """
        Runs a Streamlit application.
        """

        if "messages" not in st.session_state.keys():
            # Add instructions
            st.session_state.messages = [{"role": "assistant", "content": self.instructions()}]

        if question := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": question})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                logger.debug(f"QUESTION: {question}")

                # Check for Graph RAG
                graph = GraphContext(self.embeddings, self.context)
                question, context = graph(question)

                # Graph RAG
                if context:
                    logger.debug(f"----------------- GRAPH CONTEXT ({len(context)})----------------")
                    for x in context:
                        logger.debug(x)

                # Vector RAG
                else:
                    logger.debug("-----------------CONTEXT----------------")
                    for x in self.embeddings.search(question, self.context):
                        logger.debug(x)

                # Run RAG
                response = self.rag(question, context, maxlength=4096, stream=True)

                # Render response
                response = st.write_stream(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


@st.cache_resource(show_spinner="Initializing models and database...")
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
    st.title(os.environ.get("TITLE", "🚀 RAG with txtai"))

    # Create and run RAG application
    app = create()
    app.run()

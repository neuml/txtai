"""
Extractor module
"""

from nltk.tokenize import sent_tokenize

from .pipeline import Pipeline
from .tokenizer import Tokenizer

class Extractor(object):
    """
    Class that uses an extractive question-answering model to extract content from a given text context.
    """

    def __init__(self, embeddings, path, quantize=False, tokenizer=None):
        """
        Builds a new extractor.

        Args:
            embeddings: embeddings model
            path: path to qa model
            quantize: True if model should be quantized before inference, False otherwise.
            tokenizer: Tokenizer class
        """

        # Embeddings model and open database cursor
        self.embeddings = embeddings

        # QA Pipeline
        self.pipeline = Pipeline(path, quantize)

        # Tokenizer class use default method if not set
        self.tokenizer = tokenizer if tokenizer else Tokenizer

    def __call__(self, sections, queue):
        """
        Extracts answers to input questions. This method runs queries against a list of text sections, finds the top n best matches
        and uses that as the question context. A question-answering model is then run against the context for the input question,
        with the answer returned.

        Args:
            sections: list of (id, text) sections
            queue: input queue (name, query, question, snippet)

        Returns:
            extracted answers
        """

        # Tokenize text
        segments, tokenlist = [], []
        for sid, text in sections:
            tokens = self.tokenizer.tokenize(text)
            if tokens:
                segments.append((sid, text))
                tokenlist.append(tokens)

        # Build question-context pairs
        names, questions, contexts, snippets = [], [], [], []
        for name, query, question, snippet in queue:
            # Get list of required tokens
            must = [token.strip("+") for token in query.split() if token.startswith("+")]

            # Tokenize search query
            query = self.tokenizer.tokenize(query)

            # List of matches
            matches = []

            scores = self.embeddings.similarity(query, tokenlist)
            for x, score in enumerate(scores):
                # Get segment text
                text = segments[x][1]

                # Add result if all required tokens are present or there are not required tokens
                if not must or all([token.lower() in text.lower() for token in must]):
                    matches.append(segments[x] + (score,))

            # Build context using top n best matching segments
            topn = sorted(matches, key=lambda x: x[2], reverse=True)[:3]
            context = " ".join([text for _, text, _ in sorted(topn, key=lambda x: x[0])])

            names.append(name)
            questions.append(question)
            contexts.append(context)
            snippets.append(snippet)

        # Run qa pipeline and return answers
        return self.answers(names, questions, contexts, snippets)

    def answers(self, names, questions, contexts, snippets):
        """
        Executes QA pipeline and formats extracted answers.

        Args:
            names: column names
            questions: questions
            contexts: question context
            snippets: flags to enable answer snippets per answer
        """

        results = []

        # Run qa pipeline
        answers = self.pipeline(questions, contexts)

        # Extract and format answer
        for x, answer in enumerate(answers):
            # Extract answer
            value = answer["answer"]

            # Resolve snippet if necessary
            if answer and snippets[x]:
                value = self.snippet(contexts[x], value)

            results.append((names[x], value))

        return results

    def snippet(self, context, answer):
        """
        Extracts text surrounding the answer within context.

        Args:
            context: full context
            answer: answer within context

        Returns:
            text surrounding answer as a snippet
        """

        # Searches for first sentence to contain answer
        if answer:
            for x in sent_tokenize(context):
                if answer in x:
                    return x

        return answer

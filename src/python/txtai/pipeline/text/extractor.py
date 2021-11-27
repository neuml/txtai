"""
Extractor module
"""

from ..base import Pipeline
from ..data import Tokenizer

from .questions import Questions
from .similarity import Similarity


class Extractor(Pipeline):
    """
    Class that uses an extractive question-answering model to extract content from a given text context.
    """

    def __init__(self, similarity, path, quantize=False, gpu=True, model=None, tokenizer=None, minscore=None, mintokens=None, context=None):
        """
        Builds a new extractor.

        Args:
            similarity: similarity instance (embeddings or similarity instance)
            path: path to qa model
            quantize: True if model should be quantized before inference, False otherwise.
            gpu: if gpu inference should be used (only works if GPUs are available)
            model: optional existing pipeline model to wrap
            tokenizer: Tokenizer class
            minscore: minimum score to include context match, defaults to None
            mintokens: minimum number of tokens to include context match, defaults to None
            context: topn context matches to include, defaults to 3
        """

        # Similarity instance
        self.similarity = similarity

        # QA Pipeline
        self.pipeline = Questions(path, quantize, gpu, model)

        # Tokenizer class use default method if not set
        self.tokenizer = tokenizer if tokenizer else Tokenizer

        # Minimum score to include context match
        self.minscore = minscore if minscore is not None else 0.0

        # Minimum number of tokens to include context match
        self.mintokens = mintokens if mintokens is not None else 0.0

        # Top N context matches to include for question-answering
        self.context = context if context else 3

    def __call__(self, queue, texts):
        """
        Extracts answers to input questions. This method runs queries against a list of text, finds the top n best matches
        and uses that as the question context. A question-answering model is then run against the context for the input question,
        with the answer returned.

        Args:
            queue: input queue (name, query, question, snippet)
            texts: list of text

        Returns:
            list of (name, answer)
        """

        # Execute embeddings query
        results = self.query([query for _, query, _, _ in queue], texts)

        # Build question-context pairs
        names, questions, contexts, topns, snippets = [], [], [], [], []
        for x, (name, _, question, snippet) in enumerate(queue):
            # Build context using top n best matching segments
            topn = sorted(results[x], key=lambda y: y[2], reverse=True)[: self.context]
            context = " ".join([text for _, text, _ in sorted(topn, key=lambda y: y[0])])

            names.append(name)
            questions.append(question)
            contexts.append(context)
            topns.append([text for _, text, _ in topn])
            snippets.append(snippet)

        # Run qa pipeline and return answers
        return self.answers(names, questions, contexts, topns, snippets)

    def query(self, queries, texts):
        """
        Executes the extractor embeddings query. Returns results sorted by best match.

        Args:
            queries: list of embedding queries to run
            texts: list of text

        Returns:
            list of (id, text, score)
        """

        if not queries:
            return []

        # Tokenize text
        segments, tokenlist = [], []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if tokens:
                segments.append(text)
                tokenlist.append(tokens)

        # Add index id to segments to preserver ordering after filters
        segments = list(enumerate(segments))

        # Run batch queries for performance purposes
        if isinstance(self.similarity, Similarity):
            # Get list of (id, score) - sorted by highest score per query
            scores = self.similarity(queries, [t for _, t in segments])
        else:
            # Assume this is an embeddings instance, tokenize and run similarity queries
            scores = self.similarity.batchsimilarity([self.tokenizer.tokenize(x) for x in queries], tokenlist)

        # Build question-context pairs
        results = []
        for i, query in enumerate(queries):
            # Get list of required and prohibited tokens
            must = [token.strip("+") for token in query.split() if token.startswith("+") and len(token) > 1]
            mnot = [token.strip("-") for token in query.split() if token.startswith("-") and len(token) > 1]

            # List of matches
            matches = []
            for x, score in scores[i]:
                # Get segment text
                text = segments[x][1]

                # Add result if:
                #   - all required tokens are present or there are not required tokens AND
                #   - all prohibited tokens are not present or there are not prohibited tokens
                #   - score is above minimum score required
                #   - number of tokens is above minimum number of tokens required
                if (not must or all(token.lower() in text.lower() for token in must)) and (
                    not mnot or all(token.lower() not in text.lower() for token in mnot)
                ):
                    if score >= self.minscore and len(tokenlist[x]) >= self.mintokens:
                        matches.append(segments[x] + (score,))

            # Add query matches sorted by highest score
            results.append(matches)

        return results

    def answers(self, names, questions, contexts, topns, snippets):
        """
        Executes QA pipeline and formats extracted answers.

        Args:
            names: question identifiers/names
            questions: questions
            contexts: question context
            topns: same as question context but as a list with each candidate element
            snippets: flags to enable answer snippets per answer

        Returns:
            list of (name, answer)
        """

        results = []

        # Run qa pipeline
        answers = self.pipeline(questions, contexts)

        # Extract and format answer
        for x, answer in enumerate(answers):
            # Resolve snippet if necessary
            if answer and snippets[x]:
                answer = self.snippet(topns[x], answer)

            results.append((names[x], answer))

        return results

    def snippet(self, topn, answer):
        """
        Extracts text surrounding the answer within context.

        Args:
            topn: topn items used as a context
            answer: answer within context

        Returns:
            text surrounding answer as a snippet
        """

        # Searches for first sentence to contain answer
        if answer:
            for x in topn:
                if answer in x:
                    return x

        return answer

"""
Extractor module
"""

from transformers import AutoConfig

from ..base import Pipeline
from ..data import Tokenizer

from .generator import Generator
from .questions import Questions
from .sequences import Sequences
from .similarity import Similarity


class Extractor(Pipeline):
    """
    Finds answers for a set of queries/questions. The extractor is a combination of a similarity instance (embeddings or similarity pipeline)
    to build question context and a model that answers questions. The model can be either a prompt-driven large language model (LLM) or
    an extractive qa model.
    """

    # pylint: disable=R0913
    def __init__(
        self, similarity, path, quantize=False, gpu=True, model=None, tokenizer=None, minscore=None, mintokens=None, context=None, task=None
    ):
        """
        Builds a new extractor.

        Args:
            similarity: similarity instance (embeddings or similarity pipeline)
            path: path to model, supports Questions, Generator, Sequences or custom pipeline
            quantize: True if model should be quantized before inference, False otherwise.
            gpu: if gpu inference should be used (only works if GPUs are available)
            model: optional existing pipeline model to wrap
            tokenizer: Tokenizer class
            minscore: minimum score to include context match, defaults to None
            mintokens: minimum number of tokens to include context match, defaults to None
            context: topn context matches to include, defaults to 3
            task: model task (language-generation, sequence-sequence or question-answering), defaults to auto-detect
        """

        # Similarity instance
        self.similarity = similarity

        # Question-Answer model. Can be prompt-driven LLM or extractive qa
        self.model = self.load(path, quantize, gpu, model, task)

        # Tokenizer class use default method if not set
        self.tokenizer = tokenizer if tokenizer else Tokenizer() if hasattr(self.similarity, "scoring") and self.similarity.scoring else None

        # Minimum score to include context match
        self.minscore = minscore if minscore is not None else 0.0

        # Minimum number of tokens to include context match
        self.mintokens = mintokens if mintokens is not None else 0.0

        # Top n context matches to include for context
        self.context = context if context else 3

    def __call__(self, queue, texts=None):
        """
        Finds answers to input questions. This method runs queries to finds the top n best matches and uses that as the context.
        A model is then run against the context for each input question, with the answer returned.

        Args:
            queue: input question queue (name, query, question, snippet)
            texts: optional list of text for context, otherwise runs embeddings search

        Returns:
            list of (name, answer)
        """

        # Rank text by similarity for each query
        results = self.rank([query for _, query, _, _ in queue], texts)

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

        # Run pipeline and return answers
        return self.answers(names, questions, contexts, topns, snippets)

    def load(self, path, quantize, gpu, model, task):
        """
        Loads a question-answer model.

        Args:
            path: path to model, loads a Questions, Generator or Sequences pipeline
            quantize: True if model should be quantized before inference, False otherwise.
            gpu: if gpu inference should be used (only works if GPUs are available)
            model: optional existing pipeline model to wrap
            task: model task (language-generation, sequence-sequence or question-answering), defaults to auto-detect

        Returns:
            Generator, Sequences, Questions or custom pipeline
        """

        # If path is not a string, return input
        if not isinstance(path, str):
            return path

        # Autodetect task if not provided
        if not task:
            config = AutoConfig.from_pretrained(path)
            architecture = config.architectures[0] if config.architectures else None

            if any(x for x in ["LMHead", "CausalLM"] if x in architecture):
                task = "language-generation"
            elif "ConditionalGeneration" in architecture:
                task = "sequence-sequence"

        if task == "language-generation":
            return Generator(path, quantize, gpu, model)
        if task == "sequence-sequence":
            return Sequences(path, quantize, gpu, model)

        # Default to question-answering
        return Questions(path, quantize, gpu, model)

    def rank(self, queries, texts):
        """
        Rank texts against queries. If texts is not provided, an embeddings search will be executed.
        Returns results sorted by best match.

        Args:
            queries: list of queries
            texts: optional list of text

        Returns:
            list of (id, text, score) per query
        """

        if not queries:
            return []

        # Score text against queries
        scores, segments, tokenlist = self.score(queries, texts)

        # Build question-context pairs
        results = []
        for i, query in enumerate(queries):
            # Get list of required and prohibited tokens
            must = [token.strip("+") for token in query.split() if token.startswith("+") and len(token) > 1]
            mnot = [token.strip("-") for token in query.split() if token.startswith("-") and len(token) > 1]

            # Segment text is static when texts is passed in but different per query when an
            # embeddings search is run
            segment = segments[i] if isinstance(segments[i], list) else segments
            tokens = tokenlist[i] if isinstance(segments[i], list) else tokenlist

            # List of matches
            matches = []
            for x, score in scores[i]:
                # Get segment text
                text = segment[x][1]

                # Add result if:
                #   - all required tokens are present or there are not required tokens AND
                #   - all prohibited tokens are not present or there are not prohibited tokens
                #   - score is above minimum score required
                #   - number of tokens is above minimum number of tokens required
                if (not must or all(token.lower() in text.lower() for token in must)) and (
                    not mnot or all(token.lower() not in text.lower() for token in mnot)
                ):
                    if score >= self.minscore and len(tokens[x]) >= self.mintokens:
                        matches.append(segment[x] + (score,))

            # Add query matches sorted by highest score
            results.append(matches)

        return results

    def score(self, queries, texts):
        """
        Runs queries against texts (or an embeddings search if texts is empty) and builds list of
        similarity scores for each query-text combination.

        Args:
            queries: list of queries
            texts: optional list of text

        Returns:
            scores, segments, tokenlist
        """

        # Tokenize text
        segments, tokenlist = [], []
        if texts:
            for text in texts:
                # Run tokenizer method, if available, otherwise returns original text
                tokens = self.tokenize(text)
                if tokens:
                    segments.append(text)
                    tokenlist.append(tokens)

            # Add index id to segments to preserve ordering after filters
            segments = list(enumerate(segments))

        # Get list of (id, score) - sorted by highest score per query
        if isinstance(self.similarity, Similarity):
            # Score using similarity pipeline
            scores = self.similarity(queries, [t for _, t in segments])
        elif texts:
            # Score using embeddings.batchsimilarity
            scores = self.similarity.batchsimilarity([self.tokenize(x) for x in queries], tokenlist)
        else:
            # Score using embeddings.batchsearch
            scores, segments, tokenlist = self.batchsearch(queries)

        return scores, segments, tokenlist

    def batchsearch(self, queries):
        """
        Runs a batch embeddings search for a set of queries.

        Args:
            queries: list of queries to run

        Returns:
            scores, segments, tokenlist
        """

        scores, segments, tokenlist = [], [], []
        for results in self.similarity.batchsearch([self.tokenize(x) for x in queries], self.context):
            # Assume embeddings content is enabled and results are dictionaries
            scores.append([(x, result["score"]) for x, result in enumerate(results)])
            segments.append([(x, result["text"]) for x, result in enumerate(results)])
            tokenlist.append([self.tokenize(result["text"]) for result in results])

        return scores, segments, tokenlist

    def tokenize(self, text):
        """
        Tokenizes text. Returns original text if tokenizer is not available.

        Args:
            text: input text

        Returns:
            tokens if tokenizer available otherwise original text
        """

        return self.tokenizer(text) if self.tokenizer else text

    def answers(self, names, questions, contexts, topns, snippets):
        """
        Executes pipeline and formats extracted answers.

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

        # Run model inference for question-context pairs
        if isinstance(self.model, Questions):
            # Questions pipeline takes questions and contexts separately
            answers = self.model(questions, contexts)
        else:
            # Combine question and context into single text field for generative pipelines
            answers = self.model([f"{questions[x]} {context}" for x, context in enumerate(contexts)])

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

"""
Pipeline module
"""

import numpy as np
import torch

from transformers import pipeline

class Pipeline(object):
    """
    Light wrapper around Hugging Face's pipeline component for selected tasks. Adds support for model
    quantization and minor interface changes.
    """

    def __init__(self, task, path=None, quantize=False, gpu=False, model=None):
        """
        Loads a new pipeline model.

        Args:
            task: pipeline task or category
            path: optional path to model, accepts Hugging Face model hub id or local path,
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: if gpu inference should be used (only works if GPUs are available)
            model: optional existing pipeline model to wrap
        """

        if model:
            # Check if input model is a Pipeline or a HF pipeline
            self.pipeline = model.pipeline if isinstance(model, Pipeline) else model
        else:
            # Enable GPU inference if explicitly set and a GPU is available
            gpu = gpu and torch.cuda.is_available()

            # Transformer pipeline task
            self.pipeline = pipeline(task, model=path, tokenizer=path, device=0 if gpu else -1)

            # Model quantization. Compresses model to int8 precision, improves runtime performance. Only supported on CPU.
            if not gpu and quantize:
                # pylint: disable=E1101
                self.pipeline.model = torch.quantization.quantize_dynamic(self.pipeline.model, {torch.nn.Linear}, dtype=torch.qint8)

class Questions(Pipeline):
    """
    Runs extractive QA for a series of questions and contexts.
    """

    def __init__(self, path=None, quantize=False, gpu=False, model=None):
        super().__init__("question-answering", path, quantize, gpu, model)

    def __call__(self, questions, contexts):
        """
        Runs a extractive question-answering model against each question-context pair, finding the best answers.

        Args:
            questions: list of questions
            contexts: list of contexts to pull answers from

        Returns:
            list of answers
        """

        answers = []

        for x, question in enumerate(questions):
            if question and contexts[x]:
                # Run the QA pipeline
                result = self.pipeline(question=question, context=contexts[x])

                # Get answer and score
                answer, score = result["answer"], result["score"]

                # Require score to be at least 0.05
                if score < 0.05:
                    answer = None

                # Add answer
                answers.append(answer)
            else:
                answers.append(None)

        return answers

class Labels(Pipeline):
    """
    Applies a zero shot classifier to text using a list of labels.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("zero-shot-classification", path, quantize, gpu, model)

    def __call__(self, text, labels, multiclass=False):
        """
        Applies a zero shot classifier to text using a list of labels. Returns a list of
        (id, score) sorted by highest score, where id is the index in labels.

        This method supports text as a string or a list. If the input is a string, the return
        type is a 1D list of (id, score). If text is a list, a 2D list of (id, score) is
        returned with a row per string.

        Args:
            text: text|list
            labels: list of labels

        Returns:
            list of (id, score)
        """

        # Run ZSL pipeline
        results = self.pipeline(text, labels, multi_class=multiclass)

        # Convert results to a list if necessary
        if not isinstance(results, list):
            results = [results]

        # Build list of (id, score)
        scores = []
        for result in results:
            scores.append([(labels.index(label), result["scores"][x]) for x, label in enumerate(result["labels"])])

        return scores[0] if isinstance(text, str) else scores

class Similarity(Labels):
    """
    Computes similarity between query and list of text using a zero shot classifier.
    """

    def __call__(self, query, texts, multiclass=True):
        """
        Computes the similarity between query and list of text. Returns a list of
        (id, score) sorted by highest score, where id is the index in texts.

        This method supports query as a string or a list. If the input is a string,
        the return type is a 1D list of (id, score). If text is a list, a 2D list
        of (id, score) is returned with a row per string.

        Args:
            query: query text|list
            texts: list of texts

        Returns:
            list of (id, score)
        """

        # Call Labels pipeline for texts using input query as the candidate label
        scores = super().__call__(texts, [query] if isinstance(query, str) else query, multiclass)

        # Sort on query index id
        scores = [[score for _, score in sorted(row)] for row in scores]

        # Transpose axes to get a list of text scores for each query
        scores = np.array(scores).T.tolist()

        # Build list of (id, score) per query sorted by highest score
        scores = [sorted(enumerate(row), key=lambda x: x[1], reverse=True) for row in scores]

        return scores[0] if isinstance(query, str) else scores

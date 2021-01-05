"""
Pipeline module
"""

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

                # Require best score to be at least 0.05
                if score < 0.05:
                    answer = None

                # Add answer
                answers.append({"answer": answer, "score": score})
            else:
                answers.append({"answer": None, "score": 0.0})

        return answers

class Labels(Pipeline):
    """
    Applies labels to text sections using a zero shot classifier.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        super().__init__("zero-shot-classification", path, quantize, gpu, model)

    def __call__(self, section, labels):
        """
        Applies a zero shot classifier to a text section using a list of labels.

        Args:
            section: text|list
            labels: list of labels

        Returns:
            list of (label, score) for each text section
        """

        # Run pipeline
        results = self.pipeline(section, labels)

        # Pipeline returns a dict for single element inputs
        if not isinstance(results, list):
            return list(zip(results["labels"], results["scores"]))

        return [list(zip(result["labels"], result["scores"])) for result in results]

class Similarity(Labels):
    """
    Computes similarity between a query and a set of documents using a zero shot classifier.
    """

    def __call__(self, query, documents):
        """
        Computes the similarity between a query and a set of documents

        Args:
            query: query text
            documents: document text

        Returns:
            [computed similarity (0 - 1 with 1 being most similar)]
        """

        # Call Labels pipeline for documents using input query as the candidate label
        results = super().__call__(documents, [query])

        # Convert results to a list when documents only has a single input
        if not isinstance(results[0], list):
            results = [results]

        return [r[0][1] for r in results]

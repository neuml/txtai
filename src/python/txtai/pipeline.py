"""
Pipeline module
"""

import numpy as np
import regex as re
import torch

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class Pipeline(object):
    """
    Extractive question-answering model.

    Logic based on HuggingFace's transformers QuestionAnswering pipeline.
    """

    def __init__(self, path, quantize):
        """
        Loads a new pipeline model.

        Args:
            path: path to model
            quantize: if model should be quantized
        """

        self.model = AutoModelForQuestionAnswering.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        if quantize:
            # pylint: disable=E1101
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

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
            # Encode question and context using model tokenizer
            inputs = self.tokenizer.encode_plus(question, contexts[x], add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]

            try:
                # Run the input against the model, get candidate start-end pairs
                start, end = self.model(**inputs)
                start, end = start.detach().numpy(), end.detach().numpy()

                # Normalize start and end logits
                start = np.exp(start) / np.sum(np.exp(start))
                end = np.exp(end) / np.sum(np.exp(end))

                # Tokenized questions for BERT models take the format:
                # [CLS] Question [SEP] Answer [SEP]
                # This logic prevents the answer coming from the question
                separator = input_ids.index(self.tokenizer.sep_token_id)
                pmask = np.array([1 if x <= separator else 0 for x in range(len(input_ids))])

                # Mask the question tokens
                start, end = (start * np.abs(np.array(pmask) - 1), end * np.abs(np.array(pmask) - 1))

                tokens, answer = [], None

                start, end, score = self.score(start, end, 15)

                # Require best score to be at least 0.05
                if score >= 0.05:
                    # Get span tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[start:(end + 1)], skip_special_tokens=True)

                    # Build regex to match original string
                    answer = re.search(self.regex(tokens), contexts[x], re.IGNORECASE)
                    answer = answer[0]

                # Add answer
                answers.append({"answer": answer, "score": score})

            # pylint: disable=W0702
            except:
                answers.append({"answer": None, "score": 0.0})

        return answers

    def score(self, start, end, maxlength):
        """
        Scores all possible combinations of start and end index up to maxlength. Returns
        the best match.

        Args:
            start: start index scores
            end: end index scores
            maxlength: max number of tokens to allow in a match

        Returns:
            (start index, end index, score) of best scoring combination
        """

        # Score all possible combinations of start, end indices
        scores = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Zero out candidates with end < start and end - start > maxlength
        candidates = np.tril(np.triu(scores), maxlength - 1)

        # Get index of best rated combination
        # pylint: disable=E1101
        index = np.argmax(candidates.flatten())

        # Get (start, end, score) of best rated combination
        # pylint: disable=E1126, W0632
        start, end = np.unravel_index(index, candidates.shape)[1:]
        return start, end, candidates[0, start, end]

    def regex(self, tokens):
        """
        Builds a regular expression from tokens.

        Args:
            tokens: input tokens

        Returns:
            regex to use to extract match in original text
        """

        regex = []

        for token in tokens:
            # Escape regex characters
            token = re.escape(token)

            # Handle subwords
            if token.startswith("\\#\\#"):
                token = re.sub(r"^\\#\\#", "", token)

            regex.append(token)

        # Build and return complete regular expression
        return "\\s?".join(regex)

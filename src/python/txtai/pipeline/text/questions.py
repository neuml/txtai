"""
Questions module
"""

from ..hfmodel import HFModel

# Core library imports
from ...util import Library

torch = Library().torch()


class Questions(HFModel):
    """
    Runs extractive QA for a series of questions and contexts.
    """

    def __init__(self, path=None, quantize=False, gpu=True, batch=64, **kwargs):
        # Default model
        path = path if path else "distilbert-base-cased-distilled-squad"

        # Call parent constructor
        super().__init__(path, quantize, gpu, batch)

        # Load model and tokenizer
        self.model, self.tokenizer = self.load(path, "question-answering", **kwargs)

    def __call__(self, questions, contexts, **kwargs):
        """
        Runs a extractive question-answering model against each question-context pair, finding the best answers.

        Args:
            questions: list of questions
            contexts: list of contexts to pull answers from
            kwargs: additional keyword arguments

        Returns:
            list of answers
        """

        answers = []

        for x, question in enumerate(questions):
            if question and contexts[x]:
                # Tokenize inputs
                tokens = self.tokenizer(question, contexts[x], truncation="only_second", return_tensors="pt").to(self.device)

                # Generate outputs
                with torch.no_grad():
                    outputs = self.model(**tokens)

                # Unpack results
                startlogits, endlogits = (outputs.start_logits, outputs.end_logits) if hasattr(outputs, "start_logits") else outputs

                # Get best span as answer
                start = startlogits.argmax()
                end = endlogits.argmax()
                answer = self.answer(contexts[x], tokens, start, end)

                # Calculate span score
                startprob = torch.nn.functional.softmax(startlogits, dim=-1)[0]
                endprob = torch.nn.functional.softmax(endlogits, dim=-1)[0]
                score = startprob[start] * endprob[end]

                # Require score to be at least 0.05
                if score < 0.05:
                    answer = None

                # Add answer
                answers.append(answer)
            else:
                answers.append(None)

        return answers

    def answer(self, context, tokens, start, end):
        """
        Extracts an answer snippet from context.

        Args:
            context: context
            tokens: tokenized inputs
            start: start index of answer
            end: end index of answer

        Returns:
            answer
        """

        startword = tokens.token_to_word(start)
        endword = tokens.token_to_word(end)
        startindex = tokens.word_to_chars(startword, sequence_index=1)[0]
        endindex = tokens.word_to_chars(endword, sequence_index=1)[1]

        return context[startindex:endindex]

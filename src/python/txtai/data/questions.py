"""
Questions module
"""

from .base import Data


class Questions(Data):
    """
    Tokenizes question-answering datasets used as input for training question-answering models.
    """

    def __init__(self, tokenizer, columns, maxlength, stride):
        """
        Creates a new instance for tokenizing Questions training data.

        Args:
            tokenizer: model tokenizer
            columns: tuple of columns to use for question/context/answer
            maxlength: maximum sequence length
            stride: chunk size for splitting data for QA tasks
        """

        super().__init__(tokenizer, columns, maxlength)

        if not self.columns:
            self.columns = ("question", "context", "answers")

        self.question, self.context, self.answer = self.columns
        self.stride = stride
        self.rpad = tokenizer.padding_side == "right"

    def process(self, data):
        # Tokenize data
        tokenized = self.tokenize(data)

        # Get mapping of overflowing tokens and answer offsets
        samples = tokenized.pop("overflow_to_sample_mapping")
        offsets = tokenized.pop("offset_mapping")

        # Start/end positions
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for x, offset in enumerate(offsets):
            # Label NO ANSWER with CLS token
            inputids = tokenized["input_ids"][x]
            clstoken = inputids.index(self.tokenizer.cls_token_id)

            # Sequence ids
            sequences = tokenized.sequence_ids(x)

            # Get and format answer
            answers = self.answers(data, samples[x])

            # If no answers are given, set cls token as answer.
            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(clstoken)
                tokenized["end_positions"].append(clstoken)
            else:
                # Start/end character index of the answer in the text.
                startchar = answers["answer_start"][0]
                endchar = startchar + len(answers["text"][0])

                # Start token index of the current span in the text.
                start = 0
                while sequences[start] != (1 if self.rpad else 0):
                    start += 1

                # End token index of the current span in the text.
                end = len(inputids) - 1
                while sequences[end] != (1 if self.rpad else 0):
                    end -= 1

                # Label with CLS token if out of span
                if not (offset[start][0] <= startchar and offset[end][1] >= endchar):
                    tokenized["start_positions"].append(clstoken)
                    tokenized["end_positions"].append(clstoken)
                else:
                    # Map start character and end character to matching token index
                    while start < len(offset) and offset[start][0] <= startchar:
                        start += 1
                    tokenized["start_positions"].append(start - 1)

                    while offset[end][1] >= endchar:
                        end -= 1
                    tokenized["end_positions"].append(end + 1)

        return tokenized

    def tokenize(self, data):
        """
        Tokenizes batch of data

        Args:
            data: input data batch

        Returns:
            tokenized data
        """

        # Trim question whitespace
        data[self.question] = [x.lstrip() for x in data[self.question]]

        # Tokenize records
        return self.tokenizer(
            data[self.question if self.rpad else self.context],
            data[self.context if self.rpad else self.question],
            truncation="only_second" if self.rpad else "only_first",
            max_length=self.maxlength,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        )

    def answers(self, data, index):
        """
        Gets and formats an answer.

        Args:
            data: input examples
            index: answer index to retrieve

        Returns:
            answers dict
        """

        # Answer mappings
        answers = data[self.answer][index]
        context = data[self.context][index]

        # Handle mapping string answers to dict
        if not isinstance(answers, dict):
            if not answers:
                answers = {"text": [], "answer_start": []}
            else:
                answers = {"text": [answers], "answer_start": [context.index(answers)]}

        return answers

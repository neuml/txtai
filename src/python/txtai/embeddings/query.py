"""
Query module
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration


class Query:
    """
    Query translation model.
    """

    def __init__(self, path, prefix=None, maxlength=512):
        """
        Creates a query translation model.

        Args:
            path: path to query model
            prefix: text prefix
            maxlength: max sequence length to generate
        """

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)

        # Default prefix if not provided for T5 models
        if not prefix and isinstance(self.model, T5ForConditionalGeneration):
            prefix = "translate English to SQL: "

        self.prefix = prefix
        self.maxlength = maxlength

    def __call__(self, query):
        """
        Runs query translation model.

        Args:
            query: input query

        Returns:
            transformed query
        """

        # Add prefix, if necessary
        if self.prefix:
            query = f"{self.prefix}{query}"

        # Tokenize and generate text using model
        features = self.tokenizer([query], return_tensors="pt")
        output = self.model.generate(input_ids=features["input_ids"], attention_mask=features["attention_mask"], max_length=self.maxlength)

        # Decode tokens to text
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Clean and return generated text
        return self.clean(result)

    def clean(self, text):
        """
        Applies a series of rules to clean generated text.

        Args:
            text: input text

        Returns:
            clean text
        """

        return text.replace("$=", "<=")

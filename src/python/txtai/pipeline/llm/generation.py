"""
Generation module
"""

from ...util import TemplateFormatter


class Generation:
    """
    Base class for generative models. This class has common logic for building prompts and cleaning model results.
    """

    def __init__(self, path=None, template=None, **kwargs):
        """
        Creates a new Generation instance.

        Args:
            path: model path
            template: prompt template
            kwargs: additional keyword arguments
        """

        self.path = path
        self.template = template
        self.kwargs = kwargs

    def __call__(self, text, maxlength, **kwargs):
        """
        Generates text. Supports the following input formats:

          - String or list of strings
          - List of dictionaries with `role` and `content` key-values or lists of lists

        Args:
            text: text|list
            maxlength: maximum sequence length
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # Format inputs
        texts = [text] if isinstance(text, str) or isinstance(text[0], dict) else text

        # Apply template, if necessary
        if self.template:
            formatter = TemplateFormatter()
            texts = [formatter.format(self.template, text=x) for x in texts]

        # Run pipeline
        results = self.execute(texts, maxlength, **kwargs)

        # Clean generated text
        results = [self.clean(texts[x], result) for x, result in enumerate(results)]

        # Extract results based on inputs
        return results[0] if isinstance(text, str) or isinstance(text[0], dict) else results

    def execute(self, texts, maxlength, **kwargs):
        """
        Runs a list of prompts through a generative model.

        Args:
            texts: list of prompts to run
            maxlength: maximum sequence length
            kwargs: additional generation keyword arguments
        """

        raise NotImplementedError

    def clean(self, prompt, result):
        """
        Applies a series of rules to clean generated text.

        Args:
            prompt: original input prompt
            result: result text

        Returns:
            clean text
        """

        # Replace input prompt
        text = result.replace(prompt, "") if isinstance(prompt, str) else result

        # Apply text cleaning rules
        return text.replace("$=", "<=").strip()

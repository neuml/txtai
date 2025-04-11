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

    def __call__(self, text, maxlength, stream, stop, defaultrole, **kwargs):
        """
        Generates text. Supports the following input formats:

          - String or list of strings (instruction-tuned models must follow chat templates)
          - List of dictionaries with `role` and `content` key-values or lists of lists

        Args:
            text: text|list
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings
            defaultrole: default role to apply to text inputs (prompt for raw prompts (default) or user for user chat messages)
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # Format inputs
        texts = [text] if isinstance(text, str) or isinstance(text[0], dict) else text

        # Apply template, if necessary
        if self.template:
            formatter = TemplateFormatter()
            texts = [formatter.format(self.template, text=x) if isinstance(x, str) else x for x in texts]

        # Apply default role, if necessary
        if defaultrole == "user":
            texts = [[{"role": "user", "content": x}] if isinstance(x, str) else x for x in texts]

        # Run pipeline
        results = self.execute(texts, maxlength, stream, stop, **kwargs)

        # Streaming generation
        if stream:
            return results

        # Clean generated text
        results = [self.clean(texts[x], result) for x, result in enumerate(results)]

        # Extract results based on inputs
        return results[0] if isinstance(text, str) or isinstance(text[0], dict) else results

    def isvision(self):
        """
        Returns True if this LLM supports vision operations.

        Returns:
            True if this is a vision model
        """

        return False

    def execute(self, texts, maxlength, stream, stop, **kwargs):
        """
        Runs a list of prompts through a generative model.

        Args:
            texts: list of prompts to run
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings
            kwargs: additional generation keyword arguments

        Returns:
            generated text
        """

        # Streaming generation
        if stream:
            return self.stream(texts, maxlength, stream, stop, **kwargs)

        # Full response as content elements
        return list(self.stream(texts, maxlength, stream, stop, **kwargs))

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

    def response(self, result):
        """
        Parses response content from the result. This supports both standard and streaming
        generation.

        For standard generation, the full response is returned. For streaming generation,
        this method will stream chunks of content.

        Args:
            result: LLM response

        Returns:
            response
        """

        streamed = False
        for chunk in result:
            # Expects one of the following parameter paths
            #  - text
            #  - message.content
            #  - delta.content
            data = chunk["choices"][0]
            text = data.get("text", data.get("message", data.get("delta")))
            text = text if isinstance(text, str) else text.get("content")

            # Yield result if there is text AND it's not leading stream whitespace
            if text is not None and (streamed or text.strip()):
                yield (text.lstrip() if not streamed else text)
                streamed = True

    def stream(self, texts, maxlength, stream, stop, **kwargs):
        """
        Streams LLM responses.

        Args:
            texts: list of prompts to run
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings
            kwargs: additional generation keyword arguments

        Returns:
            responses
        """

        raise NotImplementedError

"""
Generation module
"""

import re

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

    def __call__(self, text, maxlength, stream, stop, defaultrole, stripthink, **kwargs):
        """
        Generates content. Supports the following input formats:

          - String or list of strings (instruction-tuned models must follow chat templates)
          - List of dictionaries with `role` and `content` key-values or lists of lists

        Args:
            text: text|list
            maxlength: maximum sequence length
            stream: stream response if True, defaults to False
            stop: list of stop strings
            defaultrole: default role to apply to text inputs (`auto` to infer (default), `user` for user chat messages or `prompt` for raw prompts)
            stripthink: strip thinking tags, defaults to False if stream is enabled, True otherwise
            kwargs: additional generation keyword arguments

        Returns:
            generated content
        """

        # Format inputs
        texts = [text] if isinstance(text, str) or isinstance(text[0], dict) else text

        # Apply template, if necessary
        if self.template:
            formatter = TemplateFormatter()
            texts = [formatter.format(self.template, text=x) if isinstance(x, str) else x for x in texts]

        # Run pipeline
        results = self.execute(self.format(texts, defaultrole), maxlength, stream, stop, **kwargs)

        # Streaming generation
        if stream:
            return self.cleanstream(results) if stripthink else results

        # Clean generated content
        results = [self.clean(texts[x], result, stripthink) for x, result in enumerate(results)]

        # Extract results based on inputs
        return results[0] if isinstance(text, str) or isinstance(text[0], dict) else results

    def ischat(self):
        """
        Returns True if this LLM supports chat.

        Returns:
            True if this a chat model
        """

        return True

    def isvision(self):
        """
        Returns True if this LLM supports vision.

        Returns:
            True if this is a vision model
        """

        return False

    def format(self, texts, defaultrole):
        """
        Formats inputs for LLM inference. This method handles wrapping string inputs as chat messages using the following rules.

          - defaultrole == "user" OR
          - defaultrole == "auto" AND model supports chat AND text doesn't start with an instruction token

        Args:
            texts: list of inputs
            defaultrole: default role to apply to text inputs (`auto` to infer (default), `user` for user chat messages or `prompt` for raw prompts)

        Returns:
            inputs ready for inference
        """

        # Instruction tokens
        instruct = ("<|im_start|>", "<|start|>", "<|start_of_role|>", "[INST]")

        results = []
        for text in texts:
            # Format chat messages using following rules
            #  - defaultrole == "user"
            #  - defaultrole == "auto" and text doesn't start with a instruction token
            if isinstance(text, str) and (
                defaultrole == "user" or (defaultrole == "auto" and self.ischat() and not text.strip().startswith(instruct))
            ):
                text = [{"role": "user", "content": text}]

            results.append(text)

        return results

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
            generated content
        """

        # Streaming generation
        if stream:
            return self.stream(texts, maxlength, stream, stop, **kwargs)

        # Full response as content elements
        return list(self.stream(texts, maxlength, stream, stop, **kwargs))

    def clean(self, prompt, result, stripthink):
        """
        Applies a series of rules to clean generated content.

        Args:
            prompt: original input prompt
            result: result text
            stripthink: removes thinking text if true

        Returns:
            clean content
        """

        # Replace input prompt
        text = result.replace(prompt, "") if isinstance(prompt, str) else result

        # Replace thinking text, if necessary
        if stripthink:
            text = self.cleanthink(text)

        # Apply text cleaning rules
        return text.replace("$=", "<=").strip()

    def cleanstream(self, results):
        """
        Cleans thinking tokens from streaming results and streams the remaining results.

        Args:
            results: results stream
        """

        # Consume "thinking" tokens
        text, buffer = None, ""
        for chunk in results:
            buffer += chunk
            text = self.cleanthink(buffer)
            if text != buffer:
                break

        # Yield remaining tokens
        if text.lstrip():
            yield text.lstrip()
        yield from results

    def cleanthink(self, text):
        """
        Clean thinking tokens from text.

        Args:
            text: input text

        Returns:
            text with thinking tokens removed
        """

        text = re.sub(r"(?s)<think>.+?</think>", "", text)
        text = text.split("<|channel|>final<|message|>", 1)
        text = text[1] if len(text) > 1 else text[0]
        return text

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

"""
Template module
"""

from string import Formatter

from .file import Task


class TemplateTask(Task):
    """
    Task that generates text from a template and task inputs. Templates can be used to prepare data for a number of tasks
    including generating large language model (LLM) prompts.
    """

    def register(self, template=None, rules=None, strict=True):
        """
        Read template parameters.

        Args:
            template: prompt template
            rules: parameter rules
            strict: requires all task inputs to be consumed by template, defaults to True
        """

        # pylint: disable=W0201
        # Template text
        self.template = template if template else self.defaulttemplate()

        # Template processing rules
        self.rules = rules if rules else self.defaultrules()

        # Create formatter
        self.formatter = TemplateFormatter() if strict else Formatter()

    def prepare(self, element):
        # Check if element matches any processing rules
        match = self.match(element)
        if match:
            return match

        # Apply template processing, if applicable
        if self.template:
            # Pass dictionary as named prompt template parameters
            if isinstance(element, dict):
                return self.formatter.format(self.template, **element)

            # Pass tuple as prompt template parameters (arg0 - argN)
            if isinstance(element, tuple):
                return self.formatter.format(self.template, **{f"arg{i}": x for i, x in enumerate(element)})

            # Default behavior is to use input as {text} parameter in prompt template
            return self.formatter.format(self.template, text=element)

        # Return original inputs when no prompt provided
        return element

    def defaulttemplate(self):
        """
        Generates a default template for this task. Base method returns None.

        Returns:
            default template
        """

        return None

    def defaultrules(self):
        """
        Generates a default rules for this task. Base method returns an empty dictionary.

        Returns:
            default rules
        """

        return {}

    def match(self, element):
        """
        Check if element matches any processing rules.

        Args:
            element: input element

        Returns:
            matching value if found, None otherwise
        """

        if self.rules and isinstance(element, dict):
            # Check if any rules are matched
            for key, value in self.rules.items():
                if element[key] == value:
                    return element[key]

        return None


class ExtractorTask(TemplateTask):
    """
    Template task that prepares input for an extractor pipeline.
    """

    def prepare(self, element):
        # Allow partial input with the "question" field used to complete prompt template
        if isinstance(element, dict):
            element["question"] = super().prepare(element["question"])
            return element

        # Default mode is to use element text for both query and question
        return {"query": element, "question": super().prepare(element)}


class TemplateFormatter(Formatter):
    """
    Helper class used to format template checks.
    """

    def check_unused_args(self, used_args, args, kwargs):
        difference = set(kwargs).difference(used_args)
        if difference:
            raise KeyError(difference)

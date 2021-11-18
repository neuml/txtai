"""
Hugging Face Transformers trainer wrapper module
"""

import sys

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, set_seed

from transformers import TrainingArguments as HFTrainingArguments

from ...data import Labels, Questions
from ...models import Models
from ..tensors import Tensors


class HFTrainer(Tensors):
    """
    Trains a new Hugging Face Transformer model using the Trainer framework.
    """

    def __call__(self, base, train, validation=None, columns=None, maxlength=None, stride=128, task="text-classification", **args):
        """
        Builds a new model using arguments.

        Args:
            base: path to base model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple
            train: training data
            validation: validation data
            columns: tuple of columns to use for text/label, defaults to (text, None, label)
            maxlength: maximum sequence length, defaults to tokenizer.model_max_length
            stride: chunk size for splitting data for QA tasks
            task: optional model task or category, determines the model type, defaults to "text-classification"
            args: training arguments

        Returns:
            (model, tokenizer)
        """

        # Parse TrainingArguments
        args = self.parse(args)

        # Set seed for model reproducibility
        set_seed(args.seed)

        # Load model configuration, tokenizer and max sequence length
        config, tokenizer, maxlength = self.load(base, maxlength)

        # List of labels (only for classification models)
        labels = None

        # Prepare datasets
        if task == "question-answering":
            process = Questions(tokenizer, columns, maxlength, stride)
        else:
            process = Labels(tokenizer, columns, maxlength)
            labels = process.labels(train)

        # Tokenize training and validation data
        train, validation = process(train, validation)

        # Create model to train
        model = self.model(task, base, config, labels)

        # Build trainer
        trainer = Trainer(model=model, tokenizer=tokenizer, args=args, train_dataset=train, eval_dataset=validation if validation else None)

        # Run training
        trainer.train()

        # Run evaluation
        if validation:
            trainer.evaluate()

        # Save model outputs
        if args.should_save:
            trainer.save_model()
            trainer.save_state()

        # Put model in eval mode to disable weight updates and return (model, tokenizer)
        return (model.eval(), tokenizer)

    def parse(self, updates):
        """
        Parses and merges custom arguments with defaults.

        Args:
            updates: custom arguments

        Returns:
            TrainingArguments
        """

        # Default training arguments
        args = {"output_dir": "", "save_strategy": "no", "report_to": "none", "log_level": "warning"}

        # Apply custom arguments
        args.update(updates)

        return TrainingArguments(**args)

    def load(self, base, maxlength):
        """
        Loads the base config and tokenizer.

        Args:
            base: base model - supports a file path or (model, tokenizer) tuple
            maxlength: maximum sequence length

        Returns:
            (config, tokenizer, maxlength)
        """

        if isinstance(base, (list, tuple)):
            # Unpack existing config and tokenizer
            model, tokenizer = base
            config = model.config
        else:
            # Load config
            config = AutoConfig.from_pretrained(base)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base)

        # Detect unbounded tokenizer
        Models.checklength(config, tokenizer)

        # Derive max sequence length
        maxlength = min(maxlength if maxlength else sys.maxsize, tokenizer.model_max_length)

        return (config, tokenizer, maxlength)

    def model(self, task, base, config, labels):
        """
        Loads the base model to train.

        Args:
            task: optional model task or category, determines the model type, defaults to "text-classification"
            base: base model - supports a file path or (model, tokenizer) tuple
            config: model configuration
            labels: number of labels

        Returns:
            model
        """

        if labels is not None:
            # Add number of labels to config
            config.update({"num_labels": labels})

        # pylint: disable=E1120
        # Unpack existing model or create new model from config
        if isinstance(base, (list, tuple)):
            return base[0]
        if task == "question-answering":
            return AutoModelForQuestionAnswering.from_pretrained(base, config=config)

        return AutoModelForSequenceClassification.from_pretrained(base, config=config)


class TrainingArguments(HFTrainingArguments):
    """
    Extends standard TrainingArguments to make the output directory optional for transient models.
    """

    @property
    def should_save(self):
        """
        Override should_save to disable model saving when output directory is None.

        Returns:
            If model should be saved
        """

        return super().should_save if self.output_dir else False

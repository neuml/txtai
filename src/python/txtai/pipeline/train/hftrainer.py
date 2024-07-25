"""
Hugging Face Transformers trainer wrapper module
"""

import os
import sys

import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForPreTraining,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, Trainer, set_seed
from transformers import TrainingArguments as HFTrainingArguments

# Conditional import
try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    # pylint: disable=C0412
    from transformers import BitsAndBytesConfig

    PEFT = True
except ImportError:
    PEFT = False

from ...data import Labels, Questions, Sequences, Texts
from ...models import Models, TokenDetection
from ..tensors import Tensors


class HFTrainer(Tensors):
    """
    Trains a new Hugging Face Transformer model using the Trainer framework.
    """

    # pylint: disable=R0913
    def __call__(
        self,
        base,
        train,
        validation=None,
        columns=None,
        maxlength=None,
        stride=128,
        task="text-classification",
        prefix=None,
        metrics=None,
        tokenizers=None,
        checkpoint=None,
        quantize=None,
        lora=None,
        **args
    ):
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
            prefix: optional source prefix
            metrics: optional function that computes and returns a dict of evaluation metrics
            tokenizers: optional number of concurrent tokenizers, defaults to None
            checkpoint: optional resume from checkpoint flag or path to checkpoint directory, defaults to None
            quantize: quantization configuration to pass to base model
            lora: lora configuration to pass to PEFT model
            args: training arguments

        Returns:
            (model, tokenizer)
        """

        # Quantization / LoRA support
        if (quantize or lora) and not PEFT:
            raise ImportError('PEFT is not available - install "pipeline" extra to enable')

        # Parse TrainingArguments
        args = self.parse(args)

        # Set seed for model reproducibility
        set_seed(args.seed)

        # Load model configuration, tokenizer and max sequence length
        config, tokenizer, maxlength = self.load(base, maxlength)

        # Default tokenizer pad token if it's not set
        tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token

        # Prepare parameters
        process, collator, labels = self.prepare(task, train, tokenizer, columns, maxlength, stride, prefix, args)

        # Tokenize training and validation data
        train, validation = process(train, validation, os.cpu_count() if tokenizers and isinstance(tokenizers, bool) else tokenizers)

        # Create model to train
        model = self.model(task, base, config, labels, tokenizer, quantize)

        # Default config pad token if it's not set
        model.config.pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id

        # Load as PEFT model, if necessary
        model = self.peft(task, lora, model)

        # Add model to collator
        if collator:
            collator.model = model

        # Build trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            args=args,
            train_dataset=train,
            eval_dataset=validation if validation else None,
            compute_metrics=metrics,
        )

        # Run training
        trainer.train(resume_from_checkpoint=checkpoint)

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
        args = {"output_dir": "", "save_strategy": "no", "report_to": "none", "log_level": "warning", "use_cpu": not Models.hasaccelerator()}

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

    def prepare(self, task, train, tokenizer, columns, maxlength, stride, prefix, args):
        """
        Prepares data for model training.

        Args:
            task: optional model task or category, determines the model type, defaults to "text-classification"
            train: training data
            tokenizer: model tokenizer
            columns: tuple of columns to use for text/label, defaults to (text, None, label)
            maxlength: maximum sequence length, defaults to tokenizer.model_max_length
            stride: chunk size for splitting data for QA tasks
            prefix: optional source prefix
            args: training arguments
        """

        process, collator, labels = None, None, None

        if task == "language-generation":
            process = Texts(tokenizer, columns, maxlength)
            collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if args.fp16 else None)
        elif task in ("language-modeling", "token-detection"):
            process = Texts(tokenizer, columns, maxlength)
            collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
        elif task == "question-answering":
            process = Questions(tokenizer, columns, maxlength, stride)
        elif task == "sequence-sequence":
            process = Sequences(tokenizer, columns, maxlength, prefix)
            collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
        else:
            process = Labels(tokenizer, columns, maxlength)
            labels = process.labels(train)

        return process, collator, labels

    def model(self, task, base, config, labels, tokenizer, quantize):
        """
        Loads the base model to train.

        Args:
            task: optional model task or category, determines the model type, defaults to "text-classification"
            base: base model - supports a file path or (model, tokenizer) tuple
            config: model configuration
            labels: number of labels
            tokenizer: model tokenizer
            quantize: quantization config

        Returns:
            model
        """

        if labels is not None:
            # Add number of labels to config
            config.update({"num_labels": labels})

        # Format quantization configuration
        quantization = self.quantization(quantize)

        # Clear quantization configuration if GPU is not available
        quantization = quantization if torch.cuda.is_available() else None

        # pylint: disable=E1120
        # Unpack existing model or create new model from config
        if isinstance(base, (list, tuple)) and not isinstance(base[0], str):
            return base[0]
        if task == "language-generation":
            return AutoModelForCausalLM.from_pretrained(base, config=config, quantization_config=quantization)
        if task == "language-modeling":
            return AutoModelForMaskedLM.from_pretrained(base, config=config, quantization_config=quantization)
        if task == "question-answering":
            return AutoModelForQuestionAnswering.from_pretrained(base, config=config, quantization_config=quantization)
        if task == "sequence-sequence":
            return AutoModelForSeq2SeqLM.from_pretrained(base, config=config, quantization_config=quantization)
        if task == "token-detection":
            return TokenDetection(
                AutoModelForMaskedLM.from_pretrained(base, config=config, quantization_config=quantization),
                AutoModelForPreTraining.from_pretrained(base, config=config, quantization_config=quantization),
                tokenizer,
            )

        # Default task
        return AutoModelForSequenceClassification.from_pretrained(base, config=config, quantization_config=quantization)

    def quantization(self, quantize):
        """
        Formats and returns quantization configuration.

        Args:
            quantize: input quantization configuration

        Returns:
            formatted quantization configuration
        """

        if quantize:
            # Default quantization settings when set to True
            if isinstance(quantize, bool):
                quantize = {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "bfloat16",
                }

            # Load dictionary configuration
            if isinstance(quantize, dict):
                quantize = BitsAndBytesConfig(**quantize)

        return quantize if quantize else None

    def peft(self, task, lora, model):
        """
        Wraps the input model as a PEFT model if lora configuration is set.

        Args:
            task: optional model task or category, determines the model type, defaults to "text-classification"
            lora: lora configuration
            model: transformers model

        Returns:
            wrapped model if lora configuration set, otherwise input model is returned
        """

        if lora:
            # Format LoRA configuration
            config = self.lora(task, lora)

            # Wrap as PeftModel
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)
            model.print_trainable_parameters()

        return model

    def lora(self, task, lora):
        """
        Formats and returns LoRA configuration.

        Args:
            task: optional model task or category, determines the model type, defaults to "text-classification"
            lora: lora configuration

        Returns:
            formatted lora configuration
        """

        if lora:
            # Default lora settings when set to True
            if isinstance(lora, bool):
                lora = {"r": 16, "lora_alpha": 8, "target_modules": "all-linear", "lora_dropout": 0.05, "bias": "none"}

            # Load dictionary configuration
            if isinstance(lora, dict):
                # Set task type if missing
                if "task_type" not in lora:
                    lora["task_type"] = self.loratask(task)

                lora = LoraConfig(**lora)

        return lora

    def loratask(self, task):
        """
        Looks up the corresponding LoRA task for input task.

        Args:
            task: optional model task or category, determines the model type, defaults to "text-classification"

        Returns:
            lora task
        """

        # Task mapping
        tasks = {
            "language-generation": TaskType.CAUSAL_LM,
            "language-modeling": TaskType.FEATURE_EXTRACTION,
            "question-answering": TaskType.QUESTION_ANS,
            "sequence-sequence": TaskType.SEQ_2_SEQ_LM,
            "text-classification": TaskType.SEQ_CLS,
            "token-detection": TaskType.FEATURE_EXTRACTION,
        }

        # Default task
        task = task if task in tasks else "text-classification"

        # Lookup and return task
        return tasks[task]


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

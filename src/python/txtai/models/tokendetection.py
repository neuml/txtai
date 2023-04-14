"""
Token Detection module
"""

import inspect
import os

import torch

from transformers import PreTrainedModel


class TokenDetection(PreTrainedModel):
    """
    Runs the replaced token detection training objective. This method was first proposed by the ELECTRA model.
    The method consists of a masked language model generator feeding data to a discriminator that determines
    which of the tokens are incorrect. More on this training objective can be found in the ELECTRA paper.
    """

    def __init__(self, generator, discriminator, tokenizer, weight=50.0):
        """
        Creates a new TokenDetection class.

        Args:
            generator: Generator model, must be a masked language model
            discriminator: Discriminator model, must be a model that can detect replaced tokens. Any model can
                           can be customized for this task. See ElectraForPretraining for more.
        """

        # Initialize model with discriminator config
        super().__init__(discriminator.config)

        self.generator = generator
        self.discriminator = discriminator

        # Tokenizer to save with generator and discriminator
        self.tokenizer = tokenizer

        # Discriminator weight
        self.weight = weight

        # Share embeddings if both models are the same type
        # Embeddings must be same size
        if self.generator.config.model_type == self.discriminator.config.model_type:
            self.discriminator.set_input_embeddings(self.generator.get_input_embeddings())

        # Set attention mask present flags
        self.gattention = "attention_mask" in inspect.signature(self.generator.forward).parameters
        self.dattention = "attention_mask" in inspect.signature(self.discriminator.forward).parameters

    # pylint: disable=E1101
    def forward(self, input_ids=None, labels=None, attention_mask=None, token_type_ids=None):
        """
        Runs a forward pass through the model. This method runs the masked language model then randomly samples
        the generated tokens and builds a binary classification problem for the discriminator (detecting if each token is correct).

        Args:
            input_ids: token ids
            labels: token labels
            attention_mask: attention mask
            token_type_ids: segment token indices

        Returns:
            (loss, generator outputs, discriminator outputs, discriminator labels)
        """

        # Copy input ids
        dinputs = input_ids.clone()

        # Run inputs through masked language model
        inputs = {"attention_mask": attention_mask} if self.gattention else {}
        goutputs = self.generator(input_ids, labels=labels, token_type_ids=token_type_ids, **inputs)

        # Get predictions
        preds = torch.softmax(goutputs[1], dim=-1)
        preds = preds.view(-1, self.config.vocab_size)

        tokens = torch.multinomial(preds, 1).view(-1)
        tokens = tokens.view(dinputs.shape[0], -1)

        # Labels have a -100 value to ignore loss from unchanged tokens
        mask = labels.ne(-100)

        # Replace the masked out tokens of the input with the generator predictions
        dinputs[mask] = tokens[mask]

        # Turn mask into new target labels - 1 (True) for corrupted, 0 otherwise.
        # If the prediction was correct, mark it as uncorrupted.
        correct = tokens == labels
        dlabels = mask.long()
        dlabels[correct] = 0

        # Run token classification, predict whether each token was corrupted
        inputs = {"attention_mask": attention_mask} if self.dattention else {}
        doutputs = self.discriminator(dinputs, labels=dlabels, token_type_ids=token_type_ids, **inputs)

        # Compute combined loss
        loss = goutputs[0] + self.weight * doutputs[0]
        return loss, goutputs[1], doutputs[1], dlabels

    def save_pretrained(self, output, state_dict=None, **kwargs):
        """
        Saves current model to output directory.

        Args:
            output: output directory
            state_dict: model state
            kwargs: additional keyword arguments
        """

        # Save combined model to support training from checkpoints
        super().save_pretrained(output, state_dict, **kwargs)

        # Save generator tokenizer and model
        gpath = os.path.join(output, "generator")
        self.tokenizer.save_pretrained(gpath)
        self.generator.save_pretrained(gpath)

        # Save discriminator tokenizer and model
        dpath = os.path.join(output, "discriminator")
        self.tokenizer.save_pretrained(dpath)
        self.discriminator.save_pretrained(dpath)

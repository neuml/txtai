"""
Train a bashsql model
"""

import sys

import pandas as pd

from txtai.pipeline import HFTrainer


def run(data, output):
    """
    Trains a txtsql model.

    Args:
        data: path to input data file
        output: model output path
    """

    train = HFTrainer()
    train(
        "t5-small",
        pd.read_csv(data),
        task="sequence-sequence",
        prefix="translate Bash to SQL: ",
        maxlength=512,
        per_device_train_batch_size=4,
        num_train_epochs=5,
        output_dir=output,
        overwrite_output_dir=True,
    )


if __name__ == "__main__":
    # Run train loop
    run(sys.argv[1], sys.argv[2])

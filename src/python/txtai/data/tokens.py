"""
Tokens module
"""

import torch


class Tokens(torch.utils.data.Dataset):
    """
    Default dataset used to hold tokenized data.
    """

    def __init__(self, columns):
        self.data = []

        # Map column-oriented data to rows
        for column in columns:
            for x, value in enumerate(columns[column]):
                if len(self.data) <= x:
                    self.data.append({})

                self.data[x][column] = value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

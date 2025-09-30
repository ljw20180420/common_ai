from typing import Literal
from torch import nn
import torch
from transformers.trainer_pt_utils import get_parameter_names
import optuna
import jsonargparse


class MyOptimizer:
    def __init__(
        self,
        name: Literal[
            "Adadelta",
            "Adafactor",
            "Adagrad",
            "Adam",
            "AdamW",
            # "SparseAdam", # SparseAdam does not support weight_decay
            "Adamax",
            "ASGD",
            # "LBFGS", # LBFGS does not support weight_decay
            "NAdam",
            "RAdam",
            "RMSprop",
            # "Rprop", # Rprop does not support weight_decay
            "SGD",
        ],
        learning_rate: float,
        weight_decay: float,
        **kwargs,
    ) -> None:
        """Parameters of optimizer.

        Args:
            name: Name of optimizer.
            learning_rate: Learn rate of the optimizer.
            weight_decay: The l2 regularization coefficient.
        """
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def __call__(self, model: nn.Module) -> None:
        decay_parameters = get_parameter_names(
            model=model,
            forbidden_layer_types=[nn.LayerNorm],
            forbidden_layer_names=[
                r"bias",
                r"layernorm",
                r"rmsnorm",
                r"(?:^|\.)norm(?:$|\.)",
                r"_norm(?:$|\.)",
            ],
        )
        params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = getattr(torch.optim, self.name)(
            params=params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def step(self) -> None:
        self.optimizer.step()

from typing import Literal

import jsonargparse
import optuna
from einops.layers.torch import EinMix
from torch import nn

from .generator import MyGenerator


class MyInitializer:
    def __init__(
        self,
        name: Literal[
            "uniform_",
            "normal_",
            "xavier_uniform_",
            "xavier_normal_",
            "kaiming_uniform_",
            "kaiming_normal_",
            "trunc_normal_",
        ],
        **kwargs,
    ) -> None:
        """Initializer arguments.

        Args:
            name: Name of the intialization method for model weights.
        """
        if name == "uniform_":
            self.method = lambda tensor, generator: nn.init.uniform_(
                tensor=tensor, a=-1.0, b=1.0, generator=generator
            )
            return
        self.method = lambda tensor, generator: getattr(nn.init, name)(
            tensor=tensor, generator=generator
        )

    def __call__(self, model: nn.Module, my_generator: MyGenerator) -> None:
        for m in model.modules():
            # linear layers
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Bilinear)
                or isinstance(m, EinMix)
            ):
                self.method(
                    m.weight,
                    my_generator.get_torch_generator_by_device(m.weight.device),
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            # embedding layers
            elif isinstance(m, nn.Embedding):
                self.method(
                    m.weight,
                    my_generator.get_torch_generator_by_device(m.weight.device),
                )
                if m.padding_idx is not None:
                    nn.init.constant_(m.weight[m.padding_idx], 0.0)
            # (transposed) convolution layers
            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.ConvTranspose3d)
            ):
                self.method(
                    m.weight,
                    my_generator.get_torch_generator_by_device(m.weight.device),
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            # norm layers
            elif isinstance(m, nn.RMSNorm):
                nn.init.constant_(m.weight, 1.0)

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        cfg.initializer.name = trial.suggest_categorical(
            "initializer/name",
            choices=[
                "uniform_",
                "normal_",
                "xavier_uniform_",
                "xavier_normal_",
                "kaiming_uniform_",
                "kaiming_normal_",
                "trunc_normal_",
            ],
        )

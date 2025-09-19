from typing import Literal
import torch
from .optimizer import MyOptimizer


class MyLrScheduler:
    def __init__(
        self,
        name: Literal[
            "ConstantLR",
            "CosineAnnealingWarmRestarts",
            "ReduceLROnPlateau",
        ],
        warmup_epochs: int,
        period_epochs: int,
    ) -> None:
        """Parameters for learning rate scheduler.

        Args:
            name: The scheduler type to use.
            warmup_epochs: Epochs used for a linear warmup from 0.1 to 1.0 factor of initial learning rate.
            period_epochs: The period to reset the learning rate for period scheduler.
        """
        self.name = name
        self.warmup_epochs = warmup_epochs
        self.period_epochs = period_epochs

    def __call__(self, my_optimizer: MyOptimizer) -> None:
        # SequentialLR does not support ReduceLROnPlateau
        if self.name == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=my_optimizer.optimizer
            )
            return

        warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=my_optimizer.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        if self.name == "ConstantLR":
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=my_optimizer.optimizer,
                schedulers=[
                    warm_up_scheduler,
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizer=my_optimizer.optimizer,
                        factor=1,
                        total_iters=0,
                    ),
                ],
                milestones=[self.warmup_epochs],
            )
            return

        if self.name == "CosineAnnealingWarmRestarts":
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=my_optimizer.optimizer,
                schedulers=[
                    warm_up_scheduler,
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer=my_optimizer.optimizer,
                        T_0=self.period_epochs,
                        eta_min=my_optimizer.optimizer.param_groups[0]["initial_lr"]
                        * 0.1,
                    ),
                ],
                milestones=[self.warmup_epochs],
            )
            return

        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=my_optimizer.optimizer,
            schedulers=[
                warm_up_scheduler,
                getattr(torch.optim.lr_scheduler, self.name)(
                    optimizer=my_optimizer.optimizer
                ),
            ],
            milestones=[self.warmup_epochs],
        )

    def state_dict(self) -> dict:
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.lr_scheduler.load_state_dict(state_dict)

    def step(self, loss: float):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

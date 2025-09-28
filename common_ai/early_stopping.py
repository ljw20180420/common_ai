from typing import Optional


class MyEarlyStopping:
    def __init__(
        self,
        patience: Optional[int],
        delta: float,
        **kwargs,
    ) -> None:
        """Early stopping arguments.

        Args:
            patience: early stopping patience.
            delta: early stopping loss improvement threshold.
        """
        self.patience = patience
        self.delta = delta
        self.remain_patience = patience
        self.best_loss = float("inf")

    def __call__(self, loss: float) -> bool:
        if self.patience is None:
            return False

        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.remain_patience = self.patience
            return False

        self.remain_patience -= 1
        if self.remain_patience > 0:
            return False

        # reset internal states
        self.remain_patience = self.patience
        self.best_loss = float("inf")
        return True

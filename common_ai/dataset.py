from abc import ABC, abstractmethod
import os


class MyDatasetAbstract(ABC):
    def __init__(
        self,
        data_file: os.PathLike,
        name: str,
        test_ratio: float,
        validation_ratio: float,
        seed: int,
    ):
        self.data_file = os.fspath(data_file)
        self.name = name
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.seed = seed

    @abstractmethod
    def __call__(self):
        pass

    @classmethod
    @abstractmethod
    def hpo(cls):
        pass

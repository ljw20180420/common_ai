from abc import ABC, abstractmethod


class MyDatasetAbstract(ABC):
    def __init__(
        self,
        name: str,
    ):
        self.name = name

    @abstractmethod
    def __call__(self):
        pass

    @classmethod
    @abstractmethod
    def hpo(cls):
        pass

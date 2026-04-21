from abc import ABC, abstractmethod


class MyMetricAbstract(ABC):
    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def epoch(self):
        pass

    @classmethod
    @abstractmethod
    def hpo(cls):
        pass

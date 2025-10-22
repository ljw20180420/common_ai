from abc import ABC, abstractmethod
import optuna
import jsonargparse


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

    # @classmethod
    # @abstractmethod
    # def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace):
    #     pass

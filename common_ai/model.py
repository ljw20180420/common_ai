from abc import ABC, abstractmethod


class MyModelAbstract(ABC):
    @abstractmethod
    def eval_output(self):
        pass

    @classmethod
    @abstractmethod
    def hpo(cls):
        pass

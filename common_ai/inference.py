from abc import ABC, abstractmethod


class MyInferenceAbstract(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

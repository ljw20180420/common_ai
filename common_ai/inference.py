from abc import ABC, abstractmethod
import jsonargparse
from .test import MyTest

class MyInferenceAbstract(ABC):
    def load_model(self, test_cfg: jsonargparse.Namespace, train_parser: jsonargparse.ArgumentParser) -> None:
        _, train_cfg, self.logger, self.model, self.my_generator = MyTest(
            **test_cfg.as_dict()
        ).load_model(train_parser)
        self.batch_size = train_cfg.train.batch_size

import importlib
import pathlib
import tempfile
from abc import ABC, abstractmethod

import jsonargparse

from .test import MyTest
from .inference import MyInferenceAbstract

class MyGradioFnAbstract(ABC):
    def __init__(
        self,
        app_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> None:
        self.DEFAULT_TEMP_DIR = pathlib.Path(tempfile.gettempdir())
        self.train_parser = train_parser
        self.inference_dict = {}
        for test_cfg, inference_cfg in zip(app_cfg.test, app_cfg.inference):
            test_cfg = test_cfg.init_args
            my_test = MyTest(**test_cfg.as_dict())
            _, train_cfg = my_test.load_train_cfg(train_parser)
            _, preprocess, _, model_cls = train_cfg.model.class_path.rsplit(".", 3)
            data_name = train_cfg.dataset.init_args.name
            repo_id = f"{preprocess}_{model_cls}_{data_name}"
            self.inference_dict[repo_id] = (inference_cfg, test_cfg)

    def load_inference(self, repo_id: str) -> MyInferenceAbstract:
        assert repo_id in self.inference_dict, f"repo id {repo_id} is not found"
        inference_cfg, test_cfg = self.inference_dict[repo_id]
        inference_module, inference_cls = inference_cfg.class_path.rsplit(".", 1)
        my_inference = getattr(
            importlib.import_module(inference_module), inference_cls
        )(**inference_cfg.init_args.as_dict())
        my_inference.load_model(test_cfg, self.train_parser)

        return my_inference

    @abstractmethod
    def launch(self):
        pass

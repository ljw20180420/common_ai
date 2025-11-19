import importlib
import jsonargparse
import pathlib
from abc import ABC, abstractmethod


class MyGradioFnAbstract(ABC):
    def __init__(
        self,
        app_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> None:
        self.train_parser = train_parser
        self.inference_dict = {}
        for test_cfg, inference_cfg in zip(app_cfg.test, app_cfg.inference):
            test_cfg = test_cfg.init_args
            checkpoints_path = pathlib.Path(test_cfg.checkpoints_path)
            train_cfg = train_parser.parse_path(
                checkpoints_path / "checkpoint-0" / "train.yaml"
            )
            _, preprocess, _, model_cls = train_cfg.model.class_path.rsplit(".", 3)
            data_name = train_cfg.dataset.init_args.name
            repo_id = f"{preprocess}_{model_cls}_{data_name}"
            self.inference_dict[repo_id] = (inference_cfg, test_cfg)

        self.repo_id = None

    def reload_inference(self, repo_id: str):
        if repo_id != self.repo_id:
            assert repo_id in self.inference_dict, f"repo id {repo_id} is not found"
            self.repo_id = repo_id
            inference_cfg, self.test_cfg = self.inference_dict[repo_id]
            inference_module, inference_cls = inference_cfg.class_path.rsplit(".", 1)
            self.my_inference = getattr(
                importlib.import_module(inference_module), inference_cls
            )(**inference_cfg.init_args.as_dict())

    @abstractmethod
    def launch(self):
        pass

import os
import importlib
import jsonargparse
import pathlib
from abc import ABC, abstractmethod
from huggingface_hub import HfFileSystem, hf_hub_download
from tbparse import SummaryReader
import tempfile


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

            if not os.path.exists(test_cfg.logs_path):
                # no local logs, assume logs_path as repo_id
                repo_id = test_cfg.logs_path
                hfs = HfFileSystem()
                train_logs = hfs.glob(f"{repo_id}/logs/train/*")
                assert len(train_logs) == 1, "train log not found or not unique"
                test_cfg.logs_path = pathlib.Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=f"logs/train/{train_logs[0].rsplit('/', 1)[-1]}",
                        repo_type="model",
                    )
                ).parent.parent.as_posix()

            df_train = SummaryReader(f"{test_cfg.logs_path}/train", pivot=True).scalars
            best_epoch = (
                df_train["step"]
                .iloc[df_train[f"eval/{test_cfg.target}"].argmin()]
                .item()
            )

            if not os.path.exists(test_cfg.checkpoints_path):
                # no local checkpoints, assume checkpoints_path as repo_id
                repo_id = test_cfg.checkpoints_path
                hfs = HfFileSystem()
                checkpoint_files = hfs.glob(
                    f"{repo_id}/checkpoints/checkpoint-{best_epoch}/*"
                )
                for checkpoint_file in checkpoint_files:
                    test_cfg.checkpoints_path = pathlib.Path(
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=f"checkpoints/checkpoint-{best_epoch}/{checkpoint_file.rsplit('/', 1)[-1]}",
                            repo_type="model",
                        )
                    ).parent.parent.as_posix()

            train_cfg = train_parser.parse_path(
                pathlib.Path(test_cfg.checkpoints_path)
                / f"checkpoint-{best_epoch}"
                / "train.yaml"
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

import os
import pathlib
from tqdm import tqdm
import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import importlib
import jsonargparse
from tbparse import SummaryReader
from .utils import instantiate_model, instantiate_metrics
from .logger import get_logger
from .generator import MyGenerator


class MyTest:
    def __init__(
        self,
        checkpoints_path: os.PathLike,
        logs_path: os.PathLike,
        target: str,
        overwrite: dict,
        **kwargs,
    ) -> None:
        """Test arguments.

        Args:
            checkpoints_path: path to the model checkpoints.
            logs_path: path to the model logs.
            target: target metric name.
            overwrite: a backdoor to replace train parameters when test.
        """
        self.checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
        self.logs_path = pathlib.Path(os.fspath(logs_path))
        self.target = target
        self.overwrite = overwrite

    def _overwrite_train_config(
        self, cfg: jsonargparse.Namespace
    ) -> jsonargparse.Namespace:
        for key, value in self.overwrite.items():
            cfg[key] = type(cfg[key])(value)

        return cfg

    def load_model(self, train_parser: jsonargparse.ArgumentParser) -> tuple:
        logdir = self.logs_path / "train"
        assert os.path.exists(logdir) and len(os.listdir(logdir)) > 0, "no train log"
        assert len(os.listdir(logdir)) == 1, "find more than one train log"
        df_train = SummaryReader(os.fspath(logdir), pivot=True).scalars
        best_epoch = (
            df_train["step"].iloc[df_train[f"eval/{self.target}"].argmin()].item()
        )

        cfg = train_parser.parse_path(
            self.checkpoints_path / f"checkpoint-{best_epoch}" / "train.yaml"
        )
        cfg = self._overwrite_train_config(cfg)
        logger = get_logger(**cfg.logger.as_dict())

        logger.info("instantiate model")
        model = instantiate_model(cfg)
        my_generator = MyGenerator(**cfg.generator.as_dict())

        logger.info("load model checkpoint")
        checkpoint = torch.load(
            self.checkpoints_path / f"checkpoint-{best_epoch}" / "checkpoint.pt",
            map_location=cfg.train.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        my_generator.load_state_dict(checkpoint["generator"])

        logger.info("set model device")
        setattr(model, "device", cfg.train.device)
        if isinstance(model, nn.Module):
            model = model.to(cfg.train.device)
            model.eval()

        return best_epoch, cfg, logger, model, my_generator

    @torch.no_grad()
    def __call__(
        self,
        train_parser: jsonargparse.ArgumentParser,
    ) -> int:
        best_epoch, cfg, logger, model, my_generator = self.load_model(train_parser)

        logger.info("instantiate metrics")
        metrics = instantiate_metrics(cfg)

        logger.info("setup data loader")
        dataset_module, dataset_cls = cfg.dataset.class_path.rsplit(".", 1)
        dataset = getattr(importlib.import_module(dataset_module), dataset_cls)(
            **cfg.dataset.init_args.as_dict()
        )()
        test_dataloader = DataLoader(
            dataset=dataset["test"],
            batch_size=cfg.train.batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("test model")
        for examples in tqdm(test_dataloader):
            batch = model.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            df = model.eval_output(examples, batch, my_generator)
            for metric_cls, metric_inst in metrics.items():
                metric_inst.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        logger.info("save metrics")
        if os.path.exists(self.logs_path / "test" / self.target):
            if os.path.exists(self.logs_path / "test" / f"{self.target}.bak"):
                shutil.rmtree(self.logs_path / "test" / f"{self.target}.bak")
            os.rename(
                self.logs_path / "test" / self.target,
                self.logs_path / "test" / f"{self.target}.bak",
            )

        try:
            tensorboard_writer = SummaryWriter(self.logs_path / "test" / self.target)
            _, preprocess, _, model_cls = cfg.model.class_path.rsplit(".", 3)
            tensorboard_writer.add_hparams(
                hparam_dict={
                    "preprocess": preprocess,
                    "model_cls": model_cls,
                    "target": self.target,
                },
                metric_dict={
                    f"test/{metric_cls}": metric_inst.epoch()
                    for metric_cls, metric_inst in metrics.items()
                },
                global_step=best_epoch,
            )
            tensorboard_writer.close()
            if os.path.exists(self.logs_path / "test" / f"{self.target}.bak"):
                shutil.rmtree(self.logs_path / "test" / f"{self.target}.bak")
        except Exception as err:
            tensorboard_writer.close()
            if os.path.exists(self.logs_path / "test" / f"{self.target}.bak"):
                if os.path.exists(self.logs_path / "test" / self.target):
                    shutil.rmtree(self.logs_path / "test" / self.target)
                os.rename(
                    self.logs_path / "test" / f"{self.target}.bak",
                    self.logs_path / "test" / self.target,
                )

            raise err

        return best_epoch

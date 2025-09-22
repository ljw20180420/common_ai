import re
import os
import pathlib
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Literal
import importlib
import jsonargparse
import datasets
from .utils import instantiate_model, target_to_epoch
from .logger import get_logger
from .generator import MyGenerator


class MyTest:
    def __init__(
        self,
        model_path: os.PathLike,
        target: str,
        batch_size: int,
        device: Literal["cpu", "cuda"],
    ) -> None:
        """Test arguments.

        Args:
            model_path: Path to the model.
            target: target metric name.
            batch_size: Batch size.
            device: Device.
        """
        self.model_path = pathlib.Path(os.fspath(model_path))
        self.target = target
        self.batch_size = batch_size
        self.device = device

    def get_best_cfg(
        self,
        train_parser: jsonargparse.ArgumentParser,
    ) -> jsonargparse.Namespace:
        best_epoch = target_to_epoch(
            self.model_path / "checkpoints", target=self.target
        )
        cfg = train_parser.parse_path(
            self.model_path / "checkpoints" / f"checkpoint-{best_epoch}" / "train.yaml"
        )
        return cfg

    @torch.no_grad()
    def __call__(
        self,
        cfg: jsonargparse.Namespace,
        dataset: datasets.Dataset,
    ) -> None:
        logger = get_logger(**cfg.logger.as_dict())
        logger.info("instantiate model")
        model, _ = instantiate_model(cfg)

        setattr(model, "device", self.device)
        if isinstance(model, nn.Module):
            model = model.to(self.device)
            model.eval()

        logger.info("instantiate components")
        my_generator = MyGenerator(**cfg.generator.as_dict())

        logger.info("instantiate metrics")
        metrics = {}
        for metric in cfg.metric:
            metric_module, metric_cls = metric.class_path.rsplit(".", 1)
            metrics[metric_cls] = getattr(
                importlib.import_module(metric_module), metric_cls
            )(**metric.init_args.as_dict())

        logger.info("load checkpoint")
        checkpoint = torch.load(
            self.model_path
            / "checkpoints"
            / f"checkpoint-{cfg.train.last_epoch}"
            / "checkpoint.pt",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        my_generator.load_state_dict(checkpoint["generator"])

        logger.info("load dataset")
        dl = DataLoader(
            dataset=dataset["test"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("test model")
        for examples in tqdm(dl):
            batch = model.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            df = model.eval_output(examples, batch, my_generator)
            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )
        metric_loss_dict = {"name": [], "loss": []}
        for metric_name, metric_fun in metrics.items():
            metric_loss_dict["name"].append(metric_name)
            metric_loss_dict["loss"].append(metric_fun.epoch())

        logger.info("output metrics")
        pd.DataFrame(metric_loss_dict).to_csv(
            self.model_path / f"{self.target}_test_result.csv",
            index=False,
        )

import os
import pathlib
from tqdm import tqdm
import shutil
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Literal
import importlib
import jsonargparse
from .utils import instantiate_model, target_to_epoch
from .logger import get_logger
from .generator import MyGenerator


class MyTest:
    def __init__(
        self,
        checkpoints_path: os.PathLike,
        logs_path: os.PathLike,
        target: str,
        **kwargs,
    ) -> None:
        """Test arguments.

        Args:
            checkpoints_path: path to the model checkpoints.
            logs_path: path to the model logs.
            target: target metric name.
        """
        self.checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
        self.logs_path = pathlib.Path(os.fspath(logs_path))
        self.target = target

    @torch.no_grad()
    def __call__(
        self,
        train_parser: jsonargparse.ArgumentParser,
    ) -> tuple[int, pathlib.Path]:
        best_epoch = target_to_epoch(self.logs_path, target=self.target)
        cfg = train_parser.parse_path(
            self.checkpoints_path / f"checkpoint-{best_epoch}" / "train.yaml"
        )
        logger = get_logger(**cfg.logger.as_dict())

        logger.info("instantiate model and random generator")
        model, _, _ = instantiate_model(cfg)
        my_generator = MyGenerator(**cfg.generator.as_dict())

        logger.info("instantiate metrics")
        metrics = {}
        for metric in cfg.metric:
            metric_module, metric_cls = metric.class_path.rsplit(".", 1)
            metrics[metric_cls] = getattr(
                importlib.import_module(metric_module), metric_cls
            )(**metric.init_args.as_dict())

        logger.info("load checkpoint for model and random generator")
        checkpoint = torch.load(
            self.checkpoints_path / f"checkpoint-{best_epoch}" / "checkpoint.pt",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        my_generator.load_state_dict(checkpoint["generator"])

        logger.info("set model device")
        setattr(model, "device", cfg.train.device)
        if isinstance(model, nn.Module):
            model = model.to(cfg.train.device)
            model.eval()

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
            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        logger.info("save metrics")
        logdir = self.logs_path / "test" / self.target
        if os.path.exists(logdir):
            logger.warning(f"{logdir.as_posix()} already exits, delete it.")
            shutil.rmtree(logdir)
        tensorboard_writer = SummaryWriter(logdir)
        metric_dict = {
            f"test/{metric_name}": metric_fun.epoch()
            for metric_name, metric_fun in metrics.items()
        }
        for metric_name in metrics.keys():
            tensorboard_writer.add_scalar(
                tag=f"test/{metric_name}",
                scalar_value=metric_dict[f"test/{metric_name}"],
                global_step=best_epoch,
            )
        _, preprocess, _, model_cls = cfg.model.class_path.rsplit(".", 3)
        tensorboard_writer.add_hparams(
            hparam_dict={
                "preprocess": preprocess,
                "model_cls": model_cls,
                "target": self.target,
            },
            metric_dict=metric_dict,
            global_step=best_epoch,
        )
        tensorboard_writer.close()

        return best_epoch, logdir

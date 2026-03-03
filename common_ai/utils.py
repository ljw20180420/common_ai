import importlib
import os
import pathlib

import datasets
import jsonargparse
import numpy as np
from einops.layers.torch import EinMix
from torch import nn

from .model import MyModelAbstract


def instantiate_model(cfg: jsonargparse.Namespace) -> MyModelAbstract:
    model_module, model_cls = cfg.model.class_path.rsplit(".", 1)
    model = getattr(importlib.import_module(model_module), model_cls)(
        **cfg.model.init_args.as_dict(),
    )

    return model


def instantiate_metrics(cfg: jsonargparse.Namespace) -> dict:
    metrics = {}
    for metric in cfg.metric:
        metric_module, metric_cls = metric.class_path.rsplit(".", 1)
        metrics[metric_cls] = getattr(
            importlib.import_module(metric_module), metric_cls
        )(**metric.init_args.as_dict())

    return metrics


def get_save_path(cfg: jsonargparse.Namespace) -> tuple[pathlib.Path]:
    model_module, model_cls = cfg.model.class_path.rsplit(".", 1)
    _, preprocess, _ = model_module.rsplit(".", 2)
    output_dir = pathlib.Path(os.fspath(cfg.train.output_dir))
    checkpoints_path = (
        output_dir
        / "checkpoints"
        / preprocess
        / model_cls
        / cfg.dataset.init_args.name
        / cfg.train.trial_name
    )
    logs_path = (
        output_dir
        / "logs"
        / preprocess
        / model_cls
        / cfg.dataset.init_args.name
        / cfg.train.trial_name
    )

    return checkpoints_path, logs_path


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


class ElasticNet(nn.Module):
    def __init__(self, reg_l1: float, reg_l2: float) -> None:
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2

    def forward(self, model: nn.Module):
        elastic_net_loss = 0.0
        for m in model.modules():
            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Bilinear)
                or isinstance(m, EinMix)
                or isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.ConvTranspose3d)
            ):
                if not m.weight.requires_grad:
                    continue
                elastic_net_loss += (
                    self.reg_l1 * m.weight.flatten().abs().sum()
                    + self.reg_l2 * m.weight.flatten().pow(2).sum()
                )

        return elastic_net_loss


class SeqTokenizer:
    def __init__(self, alphabet: str) -> None:
        self.ascii_code = np.frombuffer(alphabet.encode(), dtype=np.int8)
        self.int2idx = np.zeros(self.ascii_code.max() + 1, dtype=int)
        for i, c in enumerate(self.ascii_code):
            self.int2idx[c] = i

    def __call__(self, seq: str) -> np.ndarray:
        return self.int2idx[np.frombuffer(seq.encode(), dtype=np.int8)]


def split_train_valid_test(
    ds: datasets.Dataset, validation_ratio: float, test_ratio: float, seed: int
) -> datasets.Dataset:
    ds = ds["train"].train_test_split(
        test_size=test_ratio + validation_ratio, seed=seed
    )
    ds2 = ds.pop("test").train_test_split(
        test_size=test_ratio / (test_ratio + validation_ratio),
        seed=seed,
    )
    ds["validation"] = ds2.pop("train")
    ds["test"] = ds2.pop("test")
    return ds

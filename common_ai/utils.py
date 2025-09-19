from torch import nn

import re
import os
import pathlib
import numpy as np
import logging
import sys
import json
from typing import Literal
import datasets
import importlib
import jsonargparse


def instantiate_model(cfg: jsonargparse.Namespace) -> tuple:
    reg_obj = re.search(
        r"^AI\.preprocess\.(.+)\.model\.(.+)Model$", cfg.model.class_path
    )
    preprocess = reg_obj.group(1)
    model_type = reg_obj.group(2)
    model_module = importlib.import_module(f"AI.preprocess.{preprocess}.model")
    model = getattr(model_module, f"{model_type}Model")(
        **cfg.model.init_args.as_dict(),
    )
    assert (
        preprocess == model.data_collator.preprocess and model_type == model.model_type
    ), "preprocess or model type is inconsistent"

    output_dir = pathlib.Path(os.fspath(cfg.train.output_dir))
    model_path = (
        output_dir / preprocess / model_type / cfg.dataset.name / cfg.train.trial_name
    )

    return model, model_path


def target_to_epoch(checkpoints_path: os.PathLike, target: str) -> int:
    """
    Infer the epoch with the loweset metric (including loss).
    """
    checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
    if not os.path.exists(checkpoints_path):
        return -1
    check_epochs = [
        check_epoch
        for check_epoch in os.listdir(checkpoints_path)
        if re.search(r"^checkpoint-(\d+)$", check_epoch)
    ]
    if len(check_epochs) == 0:
        return -1

    metric_value_min = np.inf
    for check_epoch in check_epochs:
        with open(checkpoints_path / check_epoch / "performance.json", "r") as fd:
            performance = json.load(fd)
        if target == "loss":
            metric_value = performance["eval"]["loss"] / performance["eval"]["loss_num"]
        else:
            metric_value = performance["eval"][target]
        if metric_value <= metric_value_min:
            metric_value_min = metric_value
            epoch = int(check_epoch.split("-")[1])

    return epoch


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x) + x


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

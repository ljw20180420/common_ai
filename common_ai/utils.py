from torch import nn

import re
import os
import pathlib
import numpy as np
import datasets
import importlib
import jsonargparse
from typing import Optional
from tbparse import SummaryReader


def instantiate_model(cfg: jsonargparse.Namespace) -> tuple:
    model_module, model_cls = cfg.model.class_path.rsplit(".", 1)
    _, preprocess, _ = model_module.rsplit(".", 2)
    model = getattr(importlib.import_module(model_module), model_cls)(
        **cfg.model.init_args.as_dict(),
    )

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

    return model, checkpoints_path, logs_path


def get_latest_event_file(logdir: os.PathLike) -> Optional[pathlib.Path]:
    logdir = pathlib.Path(os.fspath(logdir))
    if not os.path.exists(logdir):
        return None
    reg_obj = re.compile(r"^events\.out\.tfevents\.(\d+)\.")
    latest_event_file, latest_time = None, 0
    for event_file in os.listdir(logdir):
        if os.path.isdir(logdir / event_file):
            continue
        mat = reg_obj.search(event_file)
        assert mat is not None, "cannot deteramine latest_event_file"
        current_file_time = int(mat.group(1))
        if current_file_time > latest_time:
            latest_time = current_file_time
            latest_event_file = event_file

    if latest_event_file is None:
        return None
    return logdir / latest_event_file


def target_to_epoch(logs_path: os.PathLike, target: str) -> int:
    """
    Infer the epoch with the loweset metric (including loss).
    """
    logs_path = pathlib.Path(os.fspath(logs_path))
    latest_event_file_train = get_latest_event_file(logs_path / "train")
    df = SummaryReader(latest_event_file_train.as_posix(), pivot=True).scalars
    epoch = df["step"].iloc[df[f"eval/{target}"].argmin()].item()

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

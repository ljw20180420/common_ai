from torch import nn

import re
import os
import pathlib
import numpy as np
import json
import datasets
import importlib
import jsonargparse
from tbparse import SummaryReader


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


def target_to_epoch(model_path: os.PathLike, target: str) -> int:
    """
    Infer the epoch with the loweset metric (including loss).
    """

    def get_latest_event_file(logdir: os.PathLike) -> tuple[pathlib.Path, int]:
        reg_obj = re.compile(r"^events\.out\.tfevents\.(\d+)\.")
        logdir = pathlib.Path(os.fspath(logdir))
        latest_time = 0
        for event_file in os.listdir(logdir):
            mat = reg_obj.search(event_file)
            assert mat is not None, "cannot deteramine latest_event_file"
            current_file_time = int(mat.group(1))
            if current_file_time > latest_time:
                latest_time = current_file_time
                latest_event_file = event_file

        return logdir / latest_event_file, latest_time

    model_path = pathlib.Path(os.fspath(model_path))
    latest_event_file_train, latest_time_train = get_latest_event_file(
        model_path / "log" / "train"
    )
    latest_event_file_eval, latest_time_eval = get_latest_event_file(
        model_path / "log" / "eval"
    )

    df_train = SummaryReader(latest_event_file_train.as_posix(), pivot=True).scalars
    if latest_time_eval > latest_time_train:
        df_eval = SummaryReader(latest_event_file_eval.as_posix(), pivot=True).scalars
        for column in df_eval.columns:
            df_train[column] = df_eval[column]

    epoch = df_train["step"].iloc[df_train[f"eval/{target}"].argmin()].item()

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

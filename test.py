import re
import numpy as np
import os
import pathlib
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Literal
import importlib
import jsonargparse
import datasets
from .utils import get_logger, target_to_epoch


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
        if os.path.exists(self.model_path / "checkpoints"):
            best_epoch = target_to_epoch(
                self.model_path / "checkpoints", target=self.target
            )
            cfg = train_parser.parse_path(
                self.model_path
                / "checkpoints"
                / f"checkpoint-{best_epoch}"
                / "train.yaml"
            )
        else:
            cfg = train_parser.parse_path(self.model_path / "train.yaml")
        return cfg

    @torch.no_grad()
    def __call__(
        self,
        cfg: jsonargparse.Namespace,
        dataset: datasets.Dataset,
    ) -> None:
        logger = get_logger(**cfg.logger.as_dict())

        logger.info("load dataset")
        dl = DataLoader(
            dataset=dataset["test"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("load metric")
        metrics = {}
        for metric in cfg.metric:
            metric_module, metric_cls = metric.class_path.rsplit(".", 1)
            metrics[metric_cls] = getattr(
                importlib.import_module(metric_module), metric_cls
            )(**metric.init_args.as_dict())

        logger.info("load model")
        reg_obj = re.search(
            r"^AI\.preprocess\.(.+)\.model\.(.+)Config$", cfg.model.class_path
        )
        preprocess = reg_obj.group(1)
        model_type = reg_obj.group(2)
        model_module = importlib.import_module(f"AI.preprocess.{preprocess}.model")
        model = getattr(model_module, f"{model_type}Model")(
            getattr(model_module, f"{model_type}Config")(
                **cfg.model.init_args.as_dict(),
            )
        )
        if hasattr(model, "my_load_model"):
            model.my_load_model(self.model_path)
        else:
            checkpoint = torch.load(
                self.model_path
                / "checkpoints"
                / f"checkpoint-{cfg.train.last_epoch}"
                / "checkpoint.pt",
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model"])
            model = model.to(self.device)
            model.eval()

        logger.info("test model")
        metric_dfs, accum_sample_idx, dfs = [], 0, []
        for examples in tqdm(dl):
            current_batch_size = len(examples)
            batch = model.data_collator(examples, output_label=True)
            df = model.eval_output(examples, batch)
            dfs.append(df)
            metric_df = pd.DataFrame({"sample_idx": np.arange(current_batch_size)})
            for metric_name, metric_fun in metrics.items():
                if (
                    hasattr(metric_fun, "global_metric")
                    and metric_fun.global_metric == True
                ):
                    continue
                metric_loss, metric_loss_num = metric_fun(
                    df=df,
                    examples=examples,
                    batch=batch,
                )
                metric_df[f"{metric_name}_loss"] = metric_loss
                metric_df[f"{metric_name}_loss_num"] = metric_loss_num
            metric_df["sample_idx"] = metric_df["sample_idx"] + accum_sample_idx
            accum_sample_idx += current_batch_size
            metric_dfs.append(metric_df)

        logger.info("output metrics")
        pd.concat(metric_dfs).to_csv(
            self.model_path / "test_result.csv",
            index=False,
        )

        logger.info("output global metrics")
        df_global = pd.concat(dfs).reset_index(drop=True)
        metric_df_global = {"metric_name": [], "metric_loss": []}
        for metric_name, metric_fun in metrics.items():
            if (
                not hasattr(metric_fun, "global_metric")
                or metric_fun.global_metric == False
            ):
                continue
            metric_loss = metric_fun(
                df=df_global,
                examples=examples,
                batch=batch,
            )
            metric_df_global["metric_name"].append(metric_name)
            metric_df_global["metric_loss"].append(metric_loss)
        pd.DataFrame(metric_df_global).to_csv(
            self.model_path / "test_result_global.csv", index=False
        )

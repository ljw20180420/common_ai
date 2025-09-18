import re
import torch
from torch import nn
import os
import pathlib
from torch.utils.data import DataLoader
import json
from typing import Literal, Callable, Generator
from numbers import Number
import importlib
from transformers.trainer_pt_utils import get_parameter_names
from tqdm import tqdm
import logging
import jsonargparse
import datasets
from .utils import instantiate_model, get_logger, MyGenerator


class MyTrain:
    def __init__(
        self,
        output_dir: os.PathLike,
        trial_name: str,
        batch_size: int,
        num_epochs: int,
        last_epoch: int,
        clip_value: float,
        accumulate_steps: int,
        device: Literal["cpu", "cuda"],
        evaluation_only: bool,
    ):
        """Train arguments.

        Args:
            output_dir: Output directory.
            trial_name: name of the training trial
            batch_size: Batch size.
            num_epochs: Total number of training epochs to perform.
            last_epoch: The last trained epochs.
            clip_value: clip the norm of gradients.
            accumulate_steps: Accumulate gradients for these steps before update parameters.
            device: Device.
            evaluation_only: Only redo evaluation for existing checkpoints instead of training.
        """
        self.output_dir = pathlib.Path(os.fspath(output_dir))
        self.trial_name = trial_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.last_epoch = last_epoch
        self.clip_value = clip_value
        self.accumulate_steps = accumulate_steps
        self.device = device
        self.evaluation_only = evaluation_only

    def get_initializer(
        self,
        name: Literal[
            "uniform_",
            "normal_",
            "xavier_uniform_",
            "xavier_normal_",
            "kaiming_uniform_",
            "kaiming_normal_",
            "trunc_normal_",
        ],
    ) -> Callable:
        """Initializer arguments.

        Args:
            name: Name of the intialization method for model weights.
        """
        generator = self.my_generator.get_torch_generator_by_device(self.device)
        if name == "uniform_":
            return lambda tensor, generator=generator: nn.init.uniform_(
                tensor=tensor, a=-1.0, b=1.0, generator=generator
            )
        return lambda tensor, generator=generator: getattr(nn.init, name)(
            tensor=tensor, generator=generator
        )

    def get_optimizer(
        self,
        name: Literal[
            "Adadelta",
            "Adafactor",
            "Adagrad",
            "Adam",
            "AdamW",
            # "SparseAdam", # SparseAdam does not support weight_decay
            "Adamax",
            "ASGD",
            # "LBFGS", # LBFGS does not support weight_decay
            "NAdam",
            "RAdam",
            "RMSprop",
            # "Rprop", # Rprop does not support weight_decay
            "SGD",
        ],
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        """Parameters of optimizer.

        Args:
            name: Name of optimizer.
            learning_rate: Learn rate of the optimizer.
            weight_decay: The l2 regularization coefficient.
        """
        decay_parameters = get_parameter_names(
            model=self.model,
            forbidden_layer_types=[nn.LayerNorm],
            forbidden_layer_names=[
                r"bias",
                r"layernorm",
                r"rmsnorm",
                r"(?:^|\.)norm(?:$|\.)",
                r"_norm(?:$|\.)",
            ],
        )
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        return getattr(torch.optim, name)(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def get_lr_scheduler(
        self,
        name: Literal[
            "ConstantLR",
            "CosineAnnealingWarmRestarts",
            "ReduceLROnPlateau",
        ],
        warmup_epochs: int,
        period_epochs: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Parameters for learning rate scheduler.

        Args:
            name: The scheduler type to use.
            warmup_epochs: Epochs used for a linear warmup from 0.1 to 1.0 factor of initial learning rate.
            period_epochs: The period to reset the learning rate for period scheduler.
        """
        # SequentialLR does not support ReduceLROnPlateau
        if name == "ReduceLROnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer)
        warm_up_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        if name == "ConstantLR":
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=[
                    warm_up_scheduler,
                    torch.optim.lr_scheduler.ConstantLR(
                        optimizer=self.optimizer,
                        factor=1,
                        total_iters=0,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        if name == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=[
                    warm_up_scheduler,
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer=self.optimizer,
                        T_0=period_epochs,
                        eta_min=self.optimizer.param_groups[0]["initial_lr"] * 0.1,
                    ),
                ],
                milestones=[warmup_epochs],
            )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=self.optimizer,
            schedulers=[
                warm_up_scheduler,
                getattr(torch.optim.lr_scheduler, name)(optimizer=self.optimizer),
            ],
            milestones=[warmup_epochs],
        )

    def instantiate_components(self, cfg: jsonargparse.Namespace) -> None:
        self.my_generator = MyGenerator(**cfg.generator.as_dict())

        self.metrics = {}
        for metric in cfg.metric:
            metric_module, metric_cls = metric.class_path.rsplit(".", 1)
            self.metrics[metric_cls] = getattr(
                importlib.import_module(metric_module), metric_cls
            )(**metric.init_args.as_dict())

        if isinstance(self.model, nn.Module):
            self.initializer = self.get_initializer(**cfg.initializer.as_dict())
            self.optimizer = self.get_optimizer(**cfg.optimizer.as_dict())
            self.lr_scheduler = self.get_lr_scheduler(**cfg.lr_scheduler.as_dict())

    def load_checkpoint(self, model_path: os.PathLike, epoch: int):
        model_path = pathlib.Path(os.fspath(model_path))
        checkpoint = torch.load(
            model_path / "checkpoints" / f"checkpoint-{epoch}" / "checkpoint.pt",
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.my_generator.load_state_dict(checkpoint["generator"])
        if isinstance(self.model, nn.Module):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def __call__(
        self,
        train_parser: jsonargparse.ArgumentParser,
        cfg: jsonargparse.Namespace,
        dataset: datasets.Dataset,
    ) -> Generator:
        logger = get_logger(**cfg.logger.as_dict())
        logger.info("instantiate model")
        self.model, model_path = instantiate_model(cfg)

        if not self.evaluation_only:
            for performance in self.my_train_model(
                dataset, train_parser, cfg, model_path, logger
            ):
                yield performance
        else:
            for performance in self.my_eval_model(
                dataset, train_parser, cfg, model_path, logger
            ):
                yield performance

    @staticmethod
    def my_initialize_model(model: nn.Module, initializer: Callable) -> None:
        for m in model.modules():
            # linear layers
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # (transposed) convolution layers
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.ConvTranspose3d)
            ):
                initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def my_train_epoch(
        self,
    ) -> tuple[float]:
        self.model.train()
        self.model.zero_grad()  # optimizer.zero_grad() is different when multiple models share a common optimizer
        train_loss, train_loss_num, grad_norm = 0.0, 0.0, 0.0
        for step, examples in tqdm(enumerate(self.train_dataloader)):
            batch = self.model.data_collator(examples, output_label=True)
            result = self.model(
                input=batch["input"],
                label=batch["label"],
                my_generator=self.my_generator,
            )

            result["loss"].backward()
            if (step + 1) % self.accumulate_steps == 0 or step == len(
                self.train_dataloader
            ):
                grad_norm += nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.clip_value
                ).item()
                self.optimizer.step()
                self.model.zero_grad()
            train_loss += (
                result["loss"].item()
                if not isinstance(result["loss"], Number)
                else result["loss"]
            )
            train_loss_num += (
                result["loss_num"].item()
                if not isinstance(result["loss_num"], Number)
                else result["loss_num"]
            )
        return train_loss, train_loss_num, grad_norm

    def my_eval_epoch(self):
        with torch.no_grad():
            if isinstance(self.model, nn.Module):
                self.model.eval()
            eval_loss, eval_loss_num = 0.0, 0.0
            for examples in tqdm(self.eval_dataloader):
                batch = self.model.data_collator(examples, output_label=True)
                result = self.model(
                    input=batch["input"],
                    label=batch["label"],
                    my_generator=self.my_generator,
                )

                eval_loss += (
                    result["loss"].item()
                    if not isinstance(result["loss"], Number)
                    else result["loss"]
                )
                eval_loss_num += (
                    result["loss_num"].item()
                    if not isinstance(result["loss_num"], Number)
                    else result["loss_num"]
                )
                df = self.model.eval_output(examples, batch)
                for metric_name, metric_fun in self.metrics.items():
                    metric_fun.step(
                        df=df,
                        examples=examples,
                        batch=batch,
                    )

            metric_loss_dict = {}
            for metric_name, metric_fun in self.metrics.items():
                metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, eval_loss_num, metric_loss_dict

    def my_train_model(
        self,
        dataset: datasets.Dataset,
        train_parser: jsonargparse.ArgumentParser,
        cfg: jsonargparse.Namespace,
        model_path: os.PathLike,
        logger: logging.Logger,
    ) -> Generator:
        # Move model to the device as soon as possible so that get_optimizer (for Adagrad) and initializer can work correctly.
        setattr(self.model, "device", self.device)
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)

        logger.info("instantiate components")
        self.instantiate_components(cfg)

        if self.last_epoch >= 0:
            logger.info("load checkpoint")
            self.load_checkpoint(model_path, self.last_epoch)
        else:
            if isinstance(self.model, nn.Module):
                logger.info("initialize model weights")
                self.my_initialize_model(self.model, self.initializer)

        self.train_dataloader = DataLoader(
            dataset=dataset["train"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
            shuffle=True,
            generator=self.my_generator.torch_c_rng,
        )
        self.eval_dataloader = DataLoader(
            dataset=dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("train loop")
        for epoch in tqdm(range(self.last_epoch + 1, self.num_epochs)):
            logger.info(f"train epoch {epoch}")
            if hasattr(self.model, "my_train_epoch"):
                train_loss, train_loss_num, grad_norm = self.model.my_train_epoch(self)
            else:
                train_loss, train_loss_num, grad_norm = self.my_train_epoch()
            print({"train_loss": train_loss / train_loss_num})

            logger.info(f"eval epoch {epoch}")
            if hasattr(self.model, "my_eval_epoch"):
                eval_loss, eval_loss_num, metric_loss_dict = self.model.my_eval_epoch(
                    self
                )
            else:
                eval_loss, eval_loss_num, metric_loss_dict = self.my_eval_epoch()
            print({"eval_loss": eval_loss / eval_loss_num})

            logger.info(f"update learning rate for {epoch}")
            if isinstance(self.model, nn.Module):
                if isinstance(
                    self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.lr_scheduler.step(eval_loss / eval_loss_num)
                else:
                    self.lr_scheduler.step()

            logger.info(f"save epoch {epoch}")
            os.makedirs(
                model_path / "checkpoints" / f"checkpoint-{epoch}", exist_ok=True
            )

            cfg.train.last_epoch = epoch
            train_parser.save(
                cfg=cfg,
                path=model_path / "checkpoints" / f"checkpoint-{epoch}" / "train.yaml",
                overwrite=True,
            )

            performance = {
                "train": {
                    "loss": train_loss,
                    "loss_num": train_loss_num,
                    "grad_num": grad_norm,
                },
                "eval": {
                    "loss": eval_loss,
                    "loss_num": eval_loss_num,
                    **metric_loss_dict,
                },
            }
            with open(
                model_path / "checkpoints" / f"checkpoint-{epoch}" / "performance.json",
                "w",
            ) as fd:
                json.dump(performance, fd, indent=4)

            obj = {
                "model": self.model.state_dict(),
                "generator": self.my_generator.state_dict(),
            }
            if isinstance(self.model, nn.Module):
                obj.update(
                    {
                        "optimizer": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    }
                )

            torch.save(
                obj=obj,
                f=model_path / "checkpoints" / f"checkpoint-{epoch}" / "checkpoint.pt",
            )

            yield performance

    def my_eval_model(
        self,
        dataset: datasets.Dataset,
        train_parser: jsonargparse.ArgumentParser,
        cfg: jsonargparse.Namespace,
        model_path: os.PathLike,
        logger: logging.Logger,
    ) -> Generator:
        setattr(self.model, "device", self.device)
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)

        logger.info("instantiate components")
        self.instantiate_components(cfg)

        logger.info("eval loop")
        for epoch in tqdm(range(self.last_epoch + 1, self.num_epochs)):
            logger.info("check config consistency")
            cfg_train = train_parser.parse_path(
                model_path / "checkpoints" / f"checkpoint-{epoch}" / "train.yaml"
            )
            for key in cfg.keys():
                if key in [
                    "config",
                    "train.last_epoch",
                    "train.evaluation_only",
                    "model.__path__",
                    "metric",
                ]:
                    continue
                assert (
                    cfg[key] == cfg_train[key]
                ), f"train and evaluation configuration of {key} is not consistent"

            logger.info("load checkpoint")
            self.load_checkpoint(model_path, epoch)

            logger.info("set dataloader")
            self.train_dataloader = DataLoader(
                dataset=dataset["train"],
                batch_size=self.batch_size,
                collate_fn=lambda examples: examples,
                shuffle=True,
                generator=self.my_generator.torch_c_rng,
            )
            self.eval_dataloader = DataLoader(
                dataset=dataset["validation"],
                batch_size=self.batch_size,
                collate_fn=lambda examples: examples,
            )

            logger.info(f"eval epoch {epoch}")
            if hasattr(self.model, "my_eval_epoch"):
                eval_loss, eval_loss_num, metric_loss_dict = self.model.my_eval_epoch(
                    self
                )
            else:
                eval_loss, eval_loss_num, metric_loss_dict = self.my_eval_epoch()
            print({"eval_loss": eval_loss / eval_loss_num})

            logger.info(f"update config and evaluation for epoch {epoch}")
            cfg.train.last_epoch = epoch
            train_parser.save(
                cfg=cfg,
                path=model_path / "checkpoints" / f"checkpoint-{epoch}" / "train.yaml",
                overwrite=True,
            )

            with open(
                model_path / "checkpoints" / f"checkpoint-{epoch}" / "performance.json",
                "r",
            ) as fd:
                performance = json.load(fd)

            performance["eval"] = {
                "loss": eval_loss,
                "loss_num": eval_loss_num,
                **metric_loss_dict,
            }
            with open(
                model_path / "checkpoints" / f"checkpoint-{epoch}" / "performance.json",
                "w",
            ) as fd:
                json.dump(performance, fd, indent=4)

            yield performance

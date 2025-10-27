import pandas as pd
import torch
from torch import nn
import os
import pathlib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import shutil
from typing import Literal, Generator
from numbers import Number
import importlib
from tqdm import tqdm
import logging
import jsonargparse
from tbparse import SummaryReader
import re
from .utils import instantiate_model, instantiate_metrics, get_save_path
from .logger import get_logger
from .generator import MyGenerator
from .initializer import MyInitializer
from .optimizer import MyOptimizer
from .lr_scheduler import MyLrScheduler
from .early_stopping import MyEarlyStopping
from .profiler import MyProfiler


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
        **kwargs,
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

    def __call__(
        self,
        train_parser: jsonargparse.ArgumentParser,
    ) -> Generator:
        cfg = train_parser.parse_args(train_parser.args)
        logger = get_logger(**cfg.logger.as_dict())

        logger.info("open tensorboard writer")
        checkpoints_path, logs_path = get_save_path(cfg)
        if os.path.exists(logs_path / "train"):
            if os.path.exists(logs_path / "train.bak"):
                shutil.rmtree(logs_path / "train.bak")
            os.rename(logs_path / "train", logs_path / "train.bak")
        if self.evaluation_only:
            assert os.path.exists(logs_path / "train.bak"), "no train log"
            logger.info("read logging directory")
            tbdf = SummaryReader(os.fspath(logs_path / "train.bak"), pivot=True).scalars
        else:
            if os.path.exists(logs_path / "profile"):
                if os.path.exists(logs_path / "profile.bak"):
                    shutil.rmtree(logs_path / "profile.bak")
                os.rename(logs_path / "profile", logs_path / "profile.bak")
        tensorboard_writer = SummaryWriter(logs_path / "train")

        try:
            if not self.evaluation_only:
                for epoch in self.my_train_model(
                    train_parser,
                    cfg,
                    checkpoints_path,
                    logs_path,
                    logger,
                    tensorboard_writer,
                ):
                    yield epoch
            else:
                for epoch in self.my_eval_model(
                    train_parser,
                    cfg,
                    checkpoints_path,
                    logger,
                    tensorboard_writer,
                    tbdf,
                ):
                    yield epoch

            logger.info("close tensorboard writer")
            tensorboard_writer.close()
            for pn in ["train", "profile"]:
                if os.path.exists(logs_path / f"{pn}.bak"):
                    shutil.rmtree(logs_path / f"{pn}.bak")
        except Exception as err:
            logger.info("close tensorboard writer")
            tensorboard_writer.close()
            for pn in ["train", "profile"]:
                if os.path.exists(logs_path / f"{pn}.bak"):
                    if os.path.exists(logs_path / pn):
                        shutil.rmtree(logs_path / pn)
                    os.rename(logs_path / f"{pn}.bak", logs_path / pn)

            raise err

    def my_train_epoch(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
        my_profiler: MyProfiler,
    ) -> tuple[float]:
        model.train()
        model.zero_grad()  # optimizer.zero_grad() is different when multiple models share a common optimizer
        train_loss, train_loss_num, grad_norm = 0.0, 0.0, 0.0
        for step, examples in tqdm(enumerate(train_dataloader)):
            batch = model.data_collator(
                examples, output_label=True, my_generator=my_generator
            )

            my_profiler.start()

            result = model(
                input=batch["input"],
                label=batch["label"],
                my_generator=my_generator,
            )

            result["loss"].backward()
            if (step + 1) % self.accumulate_steps == 0 or step == len(train_dataloader):
                grad_norm += nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=self.clip_value
                ).item()
                my_optimizer.step()
                model.zero_grad()

            my_profiler.stop()

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

    def my_eval_epoch(
        self,
        model: object,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ) -> tuple[float | dict]:
        with torch.no_grad():
            if isinstance(model, nn.Module):
                model.eval()
            eval_loss, eval_loss_num = 0.0, 0.0
            for examples in tqdm(eval_dataloader):
                batch = model.data_collator(
                    examples, output_label=True, my_generator=my_generator
                )
                result = model(
                    input=batch["input"],
                    label=batch["label"],
                    my_generator=my_generator,
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
                df = model.eval_output(examples, batch, my_generator)
                for metric_name, metric_fun in metrics.items():
                    metric_fun.step(
                        df=df,
                        examples=examples,
                        batch=batch,
                    )

            metric_loss_dict = {}
            for metric_name, metric_fun in metrics.items():
                metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, eval_loss_num, metric_loss_dict

    def my_train_model(
        self,
        train_parser: jsonargparse.ArgumentParser,
        cfg: jsonargparse.Namespace,
        checkpoints_path: os.PathLike,
        logs_path: os.PathLike,
        logger: logging.Logger,
        tensorboard_writer: SummaryWriter,
    ) -> Generator:
        logger.info("instantiate model")
        model = instantiate_model(cfg)
        my_generator = MyGenerator(**cfg.generator.as_dict())
        metrics = instantiate_metrics(cfg)
        my_optimizer = MyOptimizer(**cfg.optimizer.as_dict())
        my_lr_scheduler = MyLrScheduler(**cfg.lr_scheduler.as_dict())
        my_early_stopping = MyEarlyStopping(**cfg.early_stopping.as_dict())
        my_profiler = MyProfiler(**cfg.profiler.as_dict()).set_logs_path(logs_path)

        checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
        if self.last_epoch >= 0:
            logger.info("load checkpoint for model and random generator")
            checkpoint = torch.load(
                checkpoints_path / f"checkpoint-{self.last_epoch}" / "checkpoint.pt",
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model"])
            my_generator.load_state_dict(checkpoint["generator"])
            if isinstance(model, nn.Module):
                my_optimizer.load_state_dict(checkpoint["optimizer"])
                my_lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        else:
            logger.info("initialize model weights")
            my_initializer = MyInitializer(**cfg.initializer.as_dict())
            if hasattr(model, "my_initialize_model"):
                model.my_initialize_model(my_initializer, my_generator)
            else:
                my_initializer(model, my_generator)

        # Move model to the device before setup optimizer because some optimizers like Adagrad use device information.
        logger.info("set model device")
        setattr(model, "device", self.device)
        if isinstance(model, nn.Module):
            model = model.to(self.device)
            logger.info("setup optimizer and lr_scheduler")
            my_optimizer(model)
            my_lr_scheduler(my_optimizer)

        logger.info("setup data loader")
        dataset_module, dataset_cls = cfg.dataset.class_path.rsplit(".", 1)
        dataset = getattr(importlib.import_module(dataset_module), dataset_cls)(
            **cfg.dataset.init_args.as_dict()
        )()
        train_dataloader = torch.utils.data.DataLoader(
            dataset=dataset["train"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
            shuffle=True,
            generator=my_generator.torch_c_rng,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            dataset=dataset["validation"],
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        logger.info("train loop")
        for epoch in tqdm(range(self.last_epoch + 1, self.num_epochs)):
            logger.info(f"train epoch {epoch}")
            if hasattr(model, "my_train_epoch"):
                train_loss, train_loss_num, grad_norm = model.my_train_epoch(
                    self,
                    train_dataloader,
                    eval_dataloader,
                    my_generator,
                    my_optimizer,
                    my_profiler,
                )
            else:
                train_loss, train_loss_num, grad_norm = self.my_train_epoch(
                    model,
                    train_dataloader,
                    my_generator,
                    my_optimizer,
                    my_profiler,
                )

            logger.info(f"eval epoch {epoch}")
            if hasattr(model, "my_eval_epoch"):
                eval_loss, eval_loss_num, metric_loss_dict = model.my_eval_epoch(
                    self, eval_dataloader, my_generator, metrics
                )
            else:
                eval_loss, eval_loss_num, metric_loss_dict = self.my_eval_epoch(
                    model, eval_dataloader, my_generator, metrics
                )

            tensorboard_writer.add_scalar("train/loss", train_loss, epoch)
            tensorboard_writer.add_scalar("train/loss_num", train_loss_num, epoch)
            tensorboard_writer.add_scalar(
                "train/mean_loss", train_loss / train_loss_num, epoch
            )
            tensorboard_writer.add_scalar("train/grad_norm", grad_norm, epoch)

            tensorboard_writer.add_scalar("eval/loss", eval_loss, epoch)
            tensorboard_writer.add_scalar("eval/loss_num", eval_loss_num, epoch)
            tensorboard_writer.add_scalar(
                "eval/mean_loss", eval_loss / eval_loss_num, epoch
            )
            for metric_name, metric_val in metric_loss_dict.items():
                tensorboard_writer.add_scalar(f"eval/{metric_name}", metric_val, epoch)

            if isinstance(model, nn.Module):
                tensorboard_writer.add_scalar(
                    "train/learning_rate",
                    my_lr_scheduler.get_last_lr()[0],
                    epoch,
                )
                logger.info(f"update learning rate for epoch {epoch}")
                my_lr_scheduler.step(eval_loss / eval_loss_num)

            logger.info(f"flush tensorboard log for epoch {epoch}")
            tensorboard_writer.flush()

            logger.info(f"save epoch {epoch}")
            os.makedirs(checkpoints_path / f"checkpoint-{epoch}", exist_ok=True)

            cfg.train.last_epoch = epoch
            train_parser.save(
                cfg=cfg,
                path=checkpoints_path / f"checkpoint-{epoch}" / "train.yaml",
                overwrite=True,
            )

            obj = {
                "model": model.state_dict(),
                "generator": my_generator.state_dict(),
            }
            if isinstance(model, nn.Module):
                obj.update(
                    {
                        "optimizer": my_optimizer.state_dict(),
                        "lr_scheduler": my_lr_scheduler.state_dict(),
                    }
                )

            torch.save(
                obj=obj,
                f=checkpoints_path / f"checkpoint-{epoch}" / "checkpoint.pt",
            )

            yield epoch

            if my_early_stopping(eval_loss / eval_loss_num):
                logger.info(f"Early stop at epoch {epoch}")
                break

    def my_eval_model(
        self,
        train_parser: jsonargparse.ArgumentParser,
        cfg: jsonargparse.Namespace,
        checkpoints_path: os.PathLike,
        logger: logging.Logger,
        tensorboard_writer: SummaryWriter,
        tbdf: pd.DataFrame,
    ) -> Generator:
        logger.info("instantiate model")
        model = instantiate_model(cfg)
        my_generator = MyGenerator(**cfg.generator.as_dict())
        metrics = instantiate_metrics(cfg)

        logger.info("eval loop")
        checkpoints_path = pathlib.Path(os.fspath(checkpoints_path))
        for epoch in tqdm(range(self.last_epoch + 1, self.num_epochs)):
            logger.info("check config consistency")
            cfg_train = train_parser.parse_path(
                checkpoints_path / f"checkpoint-{epoch}" / "train.yaml"
            )
            for key in cfg.keys():
                if key in [
                    "config",
                    "train.last_epoch",
                    "train.evaluation_only",
                    "model.__path__",
                    "metric",
                    "profiler.repeat",
                ] or re.search(r"^.+\.__path__$", key):
                    continue
                assert (
                    cfg[key] == cfg_train[key]
                ), f"train and evaluation configuration of {key} is not consistent"

            logger.info("load checkpoint for model and random generator")
            checkpoint = torch.load(
                checkpoints_path / f"checkpoint-{epoch}" / "checkpoint.pt",
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model"])
            my_generator.load_state_dict(checkpoint["generator"])

            logger.info("set model device")
            setattr(model, "device", self.device)
            if isinstance(model, nn.Module):
                model = model.to(self.device)

            logger.info("setup data loader")
            dataset_module, dataset_cls = cfg.dataset.class_path.rsplit(".", 1)
            dataset = getattr(importlib.import_module(dataset_module), dataset_cls)(
                **cfg.dataset.init_args.as_dict()
            )()
            eval_dataloader = DataLoader(
                dataset=dataset["validation"],
                batch_size=self.batch_size,
                collate_fn=lambda examples: examples,
            )

            logger.info(f"eval epoch {epoch}")
            if hasattr(model, "my_eval_epoch"):
                eval_loss, eval_loss_num, metric_loss_dict = model.my_eval_epoch(
                    self, eval_dataloader, my_generator, metrics
                )
            else:
                eval_loss, eval_loss_num, metric_loss_dict = self.my_eval_epoch(
                    model, eval_dataloader, my_generator, metrics
                )

            tbdf.loc[tbdf["step"] == epoch, f"eval/loss"] = eval_loss
            tbdf.loc[tbdf["step"] == epoch, f"eval/loss_num"] = eval_loss_num
            tbdf.loc[tbdf["step"] == epoch, f"eval/mean_loss"] = (
                eval_loss / eval_loss_num
            )
            for metric_name, metric_val in metric_loss_dict.items():
                tbdf.loc[tbdf["step"] == epoch, f"eval/{metric_name}"] = metric_val

            logger.info(f"update config for epoch {epoch}")
            cfg.train.last_epoch = epoch
            train_parser.save(
                cfg=cfg,
                path=checkpoints_path / f"checkpoint-{epoch}" / "train.yaml",
                overwrite=True,
            )

            yield epoch

        logger.info("save tensorboard log")
        for tag in tbdf.columns:
            if tag == "step":
                continue
            for epoch, scalar_value in zip(tbdf["step"], tbdf[tag]):
                tensorboard_writer.add_scalar(tag, scalar_value, global_step=epoch)

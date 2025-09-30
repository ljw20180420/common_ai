#!/usr/bin/env python

import importlib
import os
import pathlib
from typing import Literal, Callable

import jsonargparse
import optuna
from torch import nn
import yaml
import importlib
from torch.utils.tensorboard import SummaryWriter
from tbparse import SummaryReader
import logging
from .test import MyTest
from .train import MyTrain
from .logger import get_logger
from .utils import get_latest_event_file


# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)


class Objective:
    def __init__(
        self,
        hpo_parser: jsonargparse.ArgumentParser,
        get_train_parser: Callable,
        target: str,
        logger: logging.Logger,
    ) -> None:
        self.hpo_parser = hpo_parser
        self.get_train_parser = get_train_parser
        self.target = target
        self.logger = logger
        cfg = hpo_parser.parse_args(hpo_parser.args).train
        _, preprocess, _, model_cls = cfg.model.class_path.rsplit(".", 3)
        self.checkpoints_parent = (
            pathlib.Path(cfg.train.output_dir)
            / "checkpoints"
            / preprocess
            / model_cls
            / cfg.dataset.init_args.name
        )
        self.logs_parent = (
            pathlib.Path(cfg.train.output_dir)
            / "logs"
            / preprocess
            / model_cls
            / cfg.dataset.init_args.name
        )

    def __call__(self, trial: optuna.Trial):
        cfg = self.hpo_parser.parse_args(self.hpo_parser.args).train
        trial_name = f"{cfg.train.trial_name}-{trial._trial_id}"
        checkpoints_path = self.checkpoints_parent / trial_name
        logs_path = self.logs_parent / trial_name
        os.makedirs(checkpoints_path, exist_ok=True)
        model_module, model_cls = cfg.model.class_path.rsplit(".", 1)

        self.logger.info("mandatory train config")
        cfg.train.trial_name = trial_name
        cfg.train.last_epoch = -1
        cfg.train.evaluation_only = False

        if issubclass(
            getattr(importlib.import_module(model_module), model_cls), nn.Module
        ):
            self.logger.info("choose initializer config")
            cfg.initializer.name = trial.suggest_categorical(
                "initializer/name",
                choices=[
                    "uniform_",
                    "normal_",
                    "xavier_uniform_",
                    "xavier_normal_",
                    "kaiming_uniform_",
                    "kaiming_normal_",
                    "trunc_normal_",
                ],
            )

            self.logger.info("choose optimizer config")
            cfg.optimizer.name = trial.suggest_categorical(
                "optimizer/name",
                choices=[
                    "Adadelta",
                    "Adafactor",
                    "Adagrad",
                    "Adam",
                    "AdamW",
                    "Adamax",
                    "ASGD",
                    "NAdam",
                    "RAdam",
                    "RMSprop",
                    "SGD",
                ],
            )
            cfg.optimizer.learning_rate = trial.suggest_float(
                "optimizer/learning_rate", 1e-5, 1e-2, log=True
            )

            self.logger.info("choose lr_scheduler config")
            cfg.lr_scheduler.name = trial.suggest_categorical(
                "lr_scheduler/name",
                choices=[
                    "CosineAnnealingWarmRestarts",
                    "ConstantLR",
                    "ReduceLROnPlateau",
                ],
            )

        self.logger.info("choose dataset config")
        dataset_module, dataset_cls = cfg.dataset.class_path.rsplit(".", 1)
        getattr(importlib.import_module(dataset_module), dataset_cls).hpo(trial, cfg)

        self.logger.info("choose metric config")
        for metric in cfg.metric:
            metric_module, metric_cls = metric.class_path.rsplit(".", 1)
            getattr(importlib.import_module(metric_module), metric_cls).hpo(trial, cfg)

        self.logger.info("choose model config")
        getattr(importlib.import_module(model_module), model_cls).hpo(trial, cfg)

        self.logger.info("write train config")
        train_parser = self.get_train_parser()
        train_parser.save(cfg, checkpoints_path / "train.yaml")
        train_parser.args = ["--config", (checkpoints_path / "train.yaml").as_posix()]

        self.logger.info("write test config")
        with open(checkpoints_path / "test.yaml", "w") as fd:
            yaml.dump(
                {
                    "checkpoints_path": checkpoints_path,
                    "logs_path": logs_path,
                    "target": self.target,
                },
                fd,
            )

        # train
        for epoch, logdir in MyTrain(**cfg.train.as_dict())(train_parser):
            latest_event_file = get_latest_event_file(logdir)
            df = SummaryReader(latest_event_file.as_posix(), pivot=True).scalars
            trial.report(
                value=df.loc[df["step"] == epoch, f"eval/{self.target}"].item(),
                step=epoch,
            )
            if trial.should_prune():
                break

        # test
        epoch, logdir = MyTest(
            checkpoints_path=checkpoints_path,
            logs_path=logs_path,
            target=self.target,
        )(train_parser)
        latest_event_file = get_latest_event_file(logdir)
        df = SummaryReader(latest_event_file.as_posix(), pivot=True).scalars
        target_metric_val = df.loc[df["step"] == epoch, f"test/{self.target}"].item()
        tensorboard_writer = SummaryWriter(self.logs_parent / "hpo")
        tensorboard_writer.add_hparams(
            hparam_dict=trial.params,
            metric_dict={f"test/{self.target}": target_metric_val},
            global_step=trial._trial_id,
        )
        tensorboard_writer.close()

        return target_metric_val


class MyHpo:
    def __init__(
        self,
        target: str,
        study_name: str,
        n_trials: int,
        sampler: Literal[
            "GridSampler",
            "RandomSampler",
            "TPESampler",
            "CmaEsSampler",
            "GPSampler",
            "PartialFixedSampler",
            "NSGAIISampler",
            "QMCSampler",
        ],
        pruner: Literal[
            "MedianPruner",
            "NopPruner",
            "PatientPruner",
            "PercentilePruner",
            "SuccessiveHalvingPruner",
            "HyperbandPruner",
            "ThresholdPruner",
            "WilcoxonPruner",
        ],
        load_if_exists: bool,
        **kwargs,
    ):
        """Arguments of Hpo.

        Args:
            target: target metric name.
            study_name: the name of the study.
            n_trials: the total number of trials in the study.
            sampler: sampler continually narrows down the search space using the records of suggested parameter values and evaluated objective values.
            pruner: pruner to stop unpromising trials at the early stages.
            load_if_exists: flag to control the behavior to handle a conflict of study names. In the case where a study named study_name already exists in the storage.
        """
        self.target = target
        self.study_name = study_name
        self.n_trials = n_trials
        self.sampler = sampler
        self.pruner = pruner
        self.load_if_exists = load_if_exists

    def __call__(
        self,
        hpo_parser: jsonargparse.ArgumentParser,
        get_train_parser: Callable,
    ) -> None:
        cfg = hpo_parser.parse_args(hpo_parser.args).train
        logger = get_logger(**cfg.logger.as_dict())

        logger.info("instantiate objective")
        objective = Objective(
            hpo_parser=hpo_parser,
            get_train_parser=get_train_parser,
            target=self.target,
            logger=logger,
        )

        logger.info("instantiate study")
        os.makedirs(objective.logs_parent, exist_ok=True)
        study = optuna.create_study(
            storage=optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    (objective.logs_parent / "optuna_journal_storage.log").as_posix()
                ),
            ),
            sampler=getattr(importlib.import_module("optuna.samplers"), self.sampler)(),
            pruner=getattr(importlib.import_module("optuna.pruners"), self.pruner)(),
            study_name=self.study_name,
            load_if_exists=self.load_if_exists,
        )

        logger.info("start study")
        study.optimize(
            func=objective,
            n_trials=self.n_trials,
        )

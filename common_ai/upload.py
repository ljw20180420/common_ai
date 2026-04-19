#!/usr/bin/env python

import os
import pathlib
import shutil
from typing import Literal
import time

from huggingface_hub import (
    upload_folder,
    upload_large_folder,
    whoami,
    errors,
)


class MyUpload:
    def __init__(self,
        output_dir: os.PathLike,
        run_type: Literal["formal", "unittest"],
        run_name: str,
        preprocess: str,
        model_cls: str,
        data_name: str,
        trial_name: str,
        ignore_patterns: list[str],
        **kwargs,
    ) -> None:
        """Upload arguments.

        Args:
            output_dir: root dir saves all results.
            run_type: formal or unittest.
            run_name: name for the run task.
            preprocess: preprocess name for dataset.
            model_cls: model class.
            data_name: name of dataset.
            trial_name: name of trial.
            ignore_patterns: file pattern to ignore from uploading.
        """
        username = whoami()["name"]
        self.repo_id = f"{username}/{preprocess}_{model_cls}_{data_name}"
        self.output_dir = pathlib.Path(os.fspath(output_dir))
        self.run_type = run_type
        self.run_name = run_name
        self.preprocess = preprocess
        self.model_cls = model_cls
        self.data_name = data_name
        self.trial_name = trial_name
        self.ignore_patterns = ignore_patterns

    def __call__(self) -> None:
        self.upload()

        success=False
        while not success:
            try:
                self.delete()
                success=True
            except errors.HfHubHTTPError as e:
                print(e)
                time.sleep(1)

    def _comp_path(self, comp: Literal["checkpoints", "logs"]) -> os.PathLike:
        return self.output_dir / self.run_type / self.run_name / comp / self.preprocess / self.model_cls / self.data_name / self.trial_name

    def upload(self) -> None:
        upload_path = self.output_dir / "upload" / self.repo_id
        if os.path.exists(upload_path):
            shutil.rmtree(upload_path)
        os.makedirs(upload_path, exist_ok=True)
        for comp in ["checkpoints", "logs"]:
            shutil.copytree(
                src=self._comp_path(comp),
                dst=upload_path / comp,
            )

        upload_large_folder(
            repo_id=self.repo_id,
            folder_path=upload_path,
            repo_type="model",
            ignore_patterns=self.ignore_patterns,
        )


    def delete(self) -> None:
        for comp in ["checkpoints", "logs"]:
            upload_folder(
                repo_id=self.repo_id,
                folder_path=self._comp_path(comp),
                path_in_repo=comp,
                ignore_patterns=self.ignore_patterns,
                delete_patterns="*",
            )

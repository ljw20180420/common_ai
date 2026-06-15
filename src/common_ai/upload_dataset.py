import os
import jsonargparse
import importlib
import httpx
import time
import datasets
from huggingface_hub import upload_file

class MyUploadDataset:
    def __init__(
        self,
        repo_id: str,
        config_name: str,
        readme: os.PathLike,
        **kwargs,
    ):
        """Upload dataset arguments.

        Args:
            repo_id: huggingface repository id user/dataset_name.
            config_name: subdataset name.
            readme: path to local dataset card README.md file.
        """
        self.repo_id = repo_id
        self.config_name = config_name
        self.readme = readme

    def __call__(self, cfg_dataset: jsonargparse.Namespace):
        dataset_module, dataset_cls = cfg_dataset.class_path.rsplit(".", 1)
        dataset: datasets.DatasetDict = getattr(importlib.import_module(dataset_module), dataset_cls)(
            **cfg_dataset.init_args.as_dict()
        )()

        while True:
            try:
                dataset.push_to_hub(repo_id=self.repo_id, config_name=self.config_name)
                break
            except KeyboardInterrupt as e:
                print(e)
                break
            except (RuntimeError, httpx.ConnectError) as e:
                print(e)
                time.sleep(1)

        upload_file(
            path_or_fileobj=self.readme,
            path_in_repo="README.md",
            repo_id=self.repo_id,
            repo_type="dataset",
        )

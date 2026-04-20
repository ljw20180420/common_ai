import os
import pathlib
from huggingface_hub import whoami, snapshot_download

class MyDownload:
    def __init__(self,
        output_dir: os.PathLike,
        preprocess: str,
        model_cls: str,
        data_name: str,
    ) -> None:
        """Download arguments.

        Args:
            output_dir: root dir saves all results.
            preprocess: preprocess name for dataset.
            model_cls: model class.
            data_name: name of dataset.
        """
        username = whoami()["name"]
        self.repo_id = f"{username}/{preprocess}_{model_cls}_{data_name}"
        self.output_dir = pathlib.Path(os.fspath(output_dir))

    def __call__(self) -> None:
        download_path = self.output_dir / "download" / self.repo_id
        snapshot_download(repo_id=self.repo_id, repo_type="model", local_dir=download_path)

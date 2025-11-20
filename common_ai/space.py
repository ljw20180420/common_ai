from huggingface_hub import HfApi
from huggingface_hub import HfFileSystem


def space(
    preprocess: str,
    model_name: str,
    data_name: str,
    owner: str,
    device: str,
):
    common_packages = [
        "accelerate",
        "transformers",
        "diffusers",
        "torch",
        "einops",
        "ViennaRNA",
        "scikit-learn",
        "more_itertools",
        "biopython",
    ]
    specific_packages = {
        "FOREcasT": [],
        "DeepHF": ["ViennaRNA", "scikit-learn", "more_itertools", "biopython"],
        "inDelphi": [],
    }

    api = HfApi()
    fs = HfFileSystem()
    while True:
        try:
            api.create_repo(
                repo_id=f"{owner}/{preprocess}_{model_name}_{data_name}",
                repo_type="space",
                exist_ok=True,
                space_sdk="gradio",
            )

            api.upload_file(
                repo_id=f"{owner}/{preprocess}_{model_name}_{data_name}",
                repo_type="space",
                path_or_fileobj="preprocess/app.py",
                path_in_repo="preprocess/app.py",
            )

            with fs.open(
                f"spaces/{owner}/{preprocess}_{model_name}_{data_name}/app.py", "w"
            ) as fd:
                fd.write(
                    f"""
from preprocess.app import app
app(
    preprocess="{preprocess}",
    model_name="{model_name}",
    data_name="{data_name}",
    owner="{owner}",
    device="{device}",
)
                    """
                )
            with fs.open(
                f"spaces/{owner}/{preprocess}_{model_name}_{data_name}/requirements.txt",
                "w",
            ) as fd:
                fd.write("\n".join(common_packages + specific_packages[preprocess]))
            break
        except Exception as err:
            print(err)
            print("retry")

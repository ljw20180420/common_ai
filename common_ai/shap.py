import os
import pathlib
from abc import ABC, abstractmethod
import importlib
from typing import Literal
import pandas as pd
import jsonargparse
import shap


class MyShapAbstract(ABC):
    def __init__(
        self,
        explainer_cls: Literal["SamplingExplainer"],
        load_only: bool,
        shap_target: str,
        nsamples_per_feature: int,
        seed: int,
    ) -> None:
        """SHAP arguments.

        Args:
            explainer_cls: the model agnostic explainer method.
            load_only: only load existing explanation.
            shap_target: shap target.
            nsamples_per_feature: number of sampling for each feature while explaining.
            seed: seed for reproducibility.
        """
        self.explainer_cls = explainer_cls
        self.load_only = load_only
        self.shap_target = shap_target
        self.nsamples_per_feature = nsamples_per_feature
        self.seed = seed

    @abstractmethod
    def dataset2pandas(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def save_explanation(
        self,
        explain_parser: jsonargparse.ArgumentParser,
        explain_cfg: jsonargparse.Namespace,
        explanation: shap.Explanation,
    ) -> None:
        logs_path = pathlib.Path(os.fspath(explain_cfg.test.logs_path))
        os.makedirs(logs_path / "explain" / self.shap_target, exist_ok=True)
        explain_parser.save(
            cfg=explain_cfg,
            path=logs_path / "explain" / self.shap_target / "explain.yaml",
            overwrite=True,
        )

        with pd.HDFStore(
            logs_path / "explain" / self.shap_target / "explanation.h5"
        ) as store:
            store["values"] = pd.DataFrame(
                data=explanation.values, columns=explanation.feature_names
            )
            store["base_values"] = pd.Series(
                data=[explanation.base_values], name="base_values"
            )
            store["data"] = pd.DataFrame(
                data=explanation.data, columns=explanation.feature_names
            )

    def load_explanation(self, explain_cfg: jsonargparse.Namespace) -> shap.Explanation:
        logs_path = pathlib.Path(os.fspath(explain_cfg.test.logs_path))

        with pd.HDFStore(
            logs_path / "explain" / self.shap_target / "explanation.h5"
        ) as store:
            explanation = shap.Explanation(
                values=store["values"].values,
                base_values=store["base_values"].item(),
                data=store["data"].values,
                feature_names=store["values"].columns.to_list(),
            )

        return explanation

    def __call__(
        self,
        explain_parser: jsonargparse.ArgumentParser,
        train_parser: jsonargparse.ArgumentParser,
    ) -> shap.Explanation:
        explain_cfg = explain_parser.parse_args(explain_parser.args)
        if self.load_only:
            return self.load_explanation(explain_cfg)

        inference_module, inference_cls = explain_cfg.inference.class_path.rsplit(
            ".", 1
        )
        my_inference = getattr(
            importlib.import_module(inference_module), inference_cls
        )(**explain_cfg.inference.init_args.as_dict())

        dataset_module, dataset_cls = explain_cfg.dataset.class_path.rsplit(".", 1)
        dataset = getattr(importlib.import_module(dataset_module), dataset_cls)(
            **explain_cfg.dataset.init_args.as_dict()
        )()

        X = self.dataset2pandas([dataset["test"]], my_inference)
        back = self.dataset2pandas(
            [dataset["train"], dataset["validation"]], my_inference
        )

        ShapExplainer = getattr(importlib.import_module("shap"), self.explainer_cls)
        shap_explainer = ShapExplainer(
            model=lambda X, my_inference=my_inference, test_cfg=explain_cfg.test, train_parser=train_parser: self.predict(
                X, my_inference, test_cfg, train_parser
            ),
            data=back,
            seed=self.seed,
        )

        explanation = shap_explainer(X, nsamples=self.nsamples_per_feature * X.shape[1])
        self.save_explanation(explain_parser, explain_cfg, explanation)

        return explanation

import os
import pathlib
from abc import ABC, abstractmethod
import importlib
from typing import Literal
import pandas as pd
import numpy as np
import jsonargparse
import shap
import copy
import matplotlib.pyplot as plt
from common_ai.prs import my_text


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

    def visualize(
        self,
        explain_parser: jsonargparse.ArgumentParser,
        train_parser: jsonargparse.ArgumentParser,
        explanation: shap.Explanation,
        local_idxs: list[int],
    ) -> None:
        explain_cfg = explain_parser.parse_args(explain_parser.args)
        logs_path = pathlib.Path(os.fspath(explain_cfg.test.logs_path))

        # inference_module, inference_cls = explain_cfg.inference.class_path.rsplit(
        #     ".", 1
        # )
        # my_inference = getattr(
        #     importlib.import_module(inference_module), inference_cls
        # )(**explain_cfg.inference.init_args.as_dict())
        # y = self.predict(explanation.data, my_inference, explain_cfg.test, train_parser)

        explanation = copy.deepcopy(explanation)
        if isinstance(explanation.base_values, float):
            explanation.base_values = np.array(
                [explanation.base_values] * explanation.shape[0]
            )
        os.makedirs(logs_path / "explain" / self.shap_target / "figures", exist_ok=True)

        self.bar_plot(explanation, local_idxs, logs_path)
        self.waterfall_plot(explanation, local_idxs, logs_path)
        self.beeswarm_plot(explanation, logs_path)
        self.decision_plot(explanation, logs_path)
        self.force_plot(explanation, local_idxs, logs_path)
        self.heatmap_plot(explanation, logs_path)
        self.scatter_plot(explanation, logs_path)
        self.violin_plot(explanation, logs_path)
        self.embedding_plot(explanation, logs_path)

    def bar_plot(
        self,
        explanation: shap.Explanation,
        local_idxs: list[int],
        logs_path: pathlib.Path,
    ) -> None:
        ax = shap.plots.bar(
            explanation,
            max_display=explanation.shape[1],
            show=False,
        )
        ax.get_figure().savefig(
            logs_path
            / "explain"
            / self.shap_target
            / "figures"
            / f"global_bar_plot.pdf"
        )
        plt.close("all")

        for local_idx in local_idxs:
            ax = shap.plots.bar(
                explanation[local_idx],
                max_display=explanation.shape[1],
                show=False,
            )
            ax.get_figure().savefig(
                logs_path
                / "explain"
                / self.shap_target
                / "figures"
                / f"local_bar_plot_{local_idx}.pdf"
            )
            plt.close("all")

    def waterfall_plot(
        self,
        explanation: shap.Explanation,
        local_idxs: list[int],
        logs_path: pathlib.Path,
    ):
        for local_idx in local_idxs:
            ax = shap.plots.waterfall(
                explanation[local_idx],
                max_display=explanation.shape[1],
                show=False,
            )
            ax.get_figure().savefig(
                logs_path
                / "explain"
                / self.shap_target
                / "figures"
                / f"waterfall_plot_{local_idx}.pdf"
            )
            plt.close("all")

    def beeswarm_plot(
        self,
        explanation: shap.Explanation,
        logs_path: pathlib.Path,
    ) -> None:
        ax = shap.plots.beeswarm(
            explanation,
            max_display=explanation.shape[1],
            show=False,
        )
        ax.get_figure().savefig(
            logs_path / "explain" / self.shap_target / "figures" / f"beeswarm_plot.pdf"
        )
        plt.close("all")

        ax = shap.plots.beeswarm(
            explanation.abs,
            max_display=explanation.shape[1],
            show=False,
        )
        ax.get_figure().savefig(
            logs_path
            / "explain"
            / self.shap_target
            / "figures"
            / f"beeswarm_abs_plot.pdf"
        )
        plt.close("all")

    def decision_plot(
        self,
        explanation: shap.Explanation,
        logs_path: pathlib.Path,
    ) -> None:
        shap.plots.decision(
            base_value=explanation.base_values[0],
            shap_values=explanation.values,
            features=explanation.data,
            feature_names=explanation.feature_names,
            feature_display_range=slice(None, None, -1),
            show=False,
        )
        plt.savefig(
            logs_path / "explain" / self.shap_target / "figures" / "decision_plot.pdf"
        )
        plt.close("all")

    def force_plot(
        self,
        explanation: shap.Explanation,
        local_idxs: list[int],
        logs_path: pathlib.Path,
    ) -> None:
        for local_idx in local_idxs:
            ax = (
                shap.plots.force(
                    explanation[local_idx],
                    matplotlib=True,
                    show=False,
                )
                .get_figure()
                .savefig(
                    logs_path
                    / "explain"
                    / self.shap_target
                    / "figures"
                    / f"local_force_plot_{local_idx}.pdf"
                )
            )
            plt.close("all")

        out_dir = logs_path / "explain" / self.shap_target / "figures"
        shap.save_html(
            out_file=(out_dir / "global_force_plot.html").as_posix(),
            plot=shap.plots.force(
                explanation,
                show=False,
            ),
        )

    def heatmap_plot(
        self,
        explanation: shap.Explanation,
        logs_path: pathlib.Path,
    ) -> None:
        ax = shap.plots.heatmap(
            shap_values=explanation,
            max_display=explanation.shape[1],
            show=False,
        )
        ax.get_figure().savefig(
            logs_path / "explain" / self.shap_target / "figures" / "heatmap_plot.pdf",
        )
        plt.close("all")

    def scatter_plot(
        self,
        explanation: shap.Explanation,
        logs_path: pathlib.Path,
    ) -> None:
        ax = shap.plots.scatter(
            shap_values=explanation[:, explanation.abs.mean(0).argsort[-1]],
            color=explanation,
            x_jitter=0.5,
            show=False,
        )
        ax.get_figure().savefig(
            logs_path / "explain" / self.shap_target / "figures" / "scatter_plot.pdf",
        )
        plt.close("all")

    def violin_plot(
        self,
        explanation: shap.Explanation,
        logs_path: pathlib.Path,
    ):
        shap.plots.violin(
            shap_values=explanation,
            max_display=explanation.shape[1],
            show=False,
        )
        plt.savefig(
            logs_path / "explain" / self.shap_target / "figures" / "violin_plot.pdf",
        )
        plt.close("all")

    def embedding_plot(
        self,
        explanation: shap.Explanation,
        logs_path: pathlib.Path,
    ):
        shap.plots.embedding(
            ind="rank(0)",
            shap_values=explanation.values,
            feature_names=explanation.feature_names,
            method="pca",
            show=False,
        )
        plt.axis("on")
        plt.savefig(
            logs_path / "explain" / self.shap_target / "figures" / "embedding_plot.pdf"
        )
        plt.close("all")

    def text_plot(
        self,
        explanation: shap.Explanation,
        local_idxs: list[int],
        logs_path: pathlib.Path,
    ) -> None:
        for local_idx in local_idxs:
            with open(
                logs_path
                / "explain"
                / self.shap_target
                / "figures"
                / f"text_plot_{local_idx}.html",
                "w",
            ) as fd:
                fd.write(my_text(explanation[local_idx], display=False))

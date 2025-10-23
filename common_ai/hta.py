from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_bar, geom_histogram, geom_area
from hta.trace_analysis import TraceAnalysis
import re
import gzip
import pathlib
import os


class MyHta:
    def __init__(
        self,
        trace_dir: os.PathLike,
        runtime_cutoff: int,
        launch_delay_cutoff: int,
    ) -> None:
        """Hta arguments.

        Args:
            trace_dir: directory containing time and space profiles.
            runtime_cutoff: threshold for short kernel.
            launch_delay_cutoff: threshold for kernel delay.
        """
        self.trace_dir = pathlib.Path(os.fspath(trace_dir))
        self.runtime_cutoff = runtime_cutoff
        self.launch_delay_cutoff = launch_delay_cutoff
        self.analyzer = TraceAnalysis(trace_dir=(trace_dir / "time").as_posix())

    def __call__(self) -> None:
        self.get_temporal_breakdown()
        self.get_idle_time_breakdown()
        self.get_gpu_kernel_breakdown()
        self.get_memory_bw_time_series()
        self.get_memory_bw_summary()
        self.get_queue_length_time_series()
        self.get_queue_length_summary()
        self.get_cuda_kernel_launch_stats()
        self.export_memory_timeline("cpu")
        self.export_memory_timeline("cuda:0")

    # Temporal breakdown
    def get_temporal_breakdown(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        df = self.analyzer.get_temporal_breakdown(visualize=False)
        df.to_csv(
            self.trace_dir / "time" / "output" / "temporal_breakdown.df", index=False
        )
        (
            ggplot(
                data=pd.melt(
                    df,
                    id_vars="rank",
                    value_vars=df.columns.drop("rank"),
                    var_name="type",
                    value_name="time",
                ),
                mapping=aes(x="rank", y="time", fill="type"),
            )
            + geom_bar(position="stack", stat="identity")
        ).save(self.trace_dir / "time" / "output" / "temporal_breakdown.png")

    # Idle time breakdown
    def get_idle_time_breakdown(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        df, _ = self.analyzer.get_idle_time_breakdown(visualize=False)
        df.to_csv(
            self.trace_dir / "time" / "output" / "idle_time_breakdown.df", index=False
        )
        (
            ggplot(
                data=df,
                mapping=aes(x="stream", y="idle_time", fill="idle_category"),
            )
            + geom_bar(position="stack", stat="identity")
        ).save(self.trace_dir / "time" / "output" / "idle_time_breakdown.png")

    # Kernel breakdown
    def get_gpu_kernel_breakdown(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        df, _ = self.analyzer.get_gpu_kernel_breakdown(visualize=False)
        df.to_csv(
            self.trace_dir / "time" / "output" / "gpu_kernel_breakdown.df", index=False
        )
        fig, ax = plt.subplots()
        ax.pie(df["percentage"], labels=df["kernel_type"])
        fig.savefig(self.trace_dir / "time" / "output" / "gpu_kernel_breakdown.png")

    # Memory bandwidth time series
    def get_memory_bw_time_series(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        dfs = self.analyzer.get_memory_bw_time_series()
        for rank, df in dfs.items():
            df.to_csv(
                self.trace_dir / "time" / "output" / f"memory_bw_time_series_{rank}.df",
                index=False,
            )

    # Memory bandwidth summary
    def get_memory_bw_summary(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        df = self.analyzer.get_memory_bw_summary()
        df.to_csv(self.trace_dir / "time" / "output" / "memory_bw_summary.df")

    # Queue length time series
    def get_queue_length_time_series(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        dfs = self.analyzer.get_queue_length_time_series()
        for rank, df in dfs.items():
            df.to_csv(
                self.trace_dir
                / "time"
                / "output"
                / f"queue_length_time_series_{rank}.df",
                index=False,
            )

    # Queue length summary
    def get_queue_length_summary(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        df = self.analyzer.get_queue_length_summary()
        df.to_csv(self.trace_dir / "time" / "output" / "queue_length_summary.df")

    # CUDA kernel launch statistics
    def get_cuda_kernel_launch_stats(self) -> None:
        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        dfs = self.analyzer.get_cuda_kernel_launch_stats(visualize=False)
        for rank, df in dfs.items():
            df.to_csv(
                self.trace_dir
                / "time"
                / "output"
                / f"uda_kernel_launch_stats_{rank}.df",
                index=False,
            )

            short_kernels = df.query(
                "cpu_duration <= @self.runtime_cutoff and gpu_duration < cpu_duration"
            )
            (
                ggplot(
                    data=short_kernels,
                    mapping=aes(x="cpu_duration"),
                )
                + geom_histogram()
            ).save(
                self.trace_dir
                / "time"
                / "output"
                / f"cuda_kernel_launch_stats_{rank}_short_kernels.png"
            )

            runtime_outliers = df.query("cpu_duration > @self.runtime_cutoff")
            (
                ggplot(
                    data=runtime_outliers,
                    mapping=aes(x="cpu_duration"),
                )
                + geom_histogram()
            ).save(
                self.trace_dir
                / "time"
                / "output"
                / f"cuda_kernel_launch_stats_{rank}_runtime_outliers.png"
            )

            launch_delay_outliers = df.query("launch_delay > @self.launch_delay_cutoff")
            (
                ggplot(
                    data=launch_delay_outliers,
                    mapping=aes(x="launch_delay"),
                )
                + geom_histogram()
            ).save(
                self.trace_dir
                / "time"
                / "output"
                / f"cuda_kernel_launch_stats_{rank}_launch_delay_outliers.png"
            )

    def export_memory_timeline(self, device: Literal["cpu", "cuda:0"]) -> None:
        os.makedirs(self.trace_dir / "space" / "output", exist_ok=True)
        df = pd.read_json(self.trace_dir / "space" / f"{device}.json.gz")

        df = (
            pd.DataFrame(
                df.loc[1].to_list(),
                index=df.loc[0].rename("timestamp"),
                columns=[
                    "baseline",
                    "parameter",
                    "optimizer_state",
                    "input",
                    "temporary",
                    "activation",
                    "gradient",
                    "autograd_detail",
                    "unknown",
                ],
            )
            .reset_index(drop=False)
            .drop(columns="baseline")
        )

        (
            ggplot(
                data=df.melt(
                    id_vars="timestamp",
                    value_vars=df.columns.drop("timestamp"),
                    var_name="type",
                    value_name="memory",
                ),
                mapping=aes(x="timestamp", y="memory", fill="type"),
            )
            + geom_area()
        ).save(self.trace_dir / "space" / "output" / f"{device}.png")

import pandas as pd
import matplotlib.pyplot as plt
from plotnine import (
    ggplot,
    aes,
    geom_bar,
    geom_histogram,
    geom_area,
    theme,
    element_text,
)
from hta.trace_analysis import TraceAnalysis
import re
import pathlib
import os


class MyHta:
    def __init__(
        self,
        trace_dir: os.PathLike,
        runtime_cutoff: int,
        launch_delay_cutoff: int,
        **kwargs,
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

        if not os.path.exists(self.trace_dir):
            return

        self.analyzers: dict[int, TraceAnalysis] = {}
        for trace_file in os.listdir(self.trace_dir / "time"):
            if os.path.isfile(self.trace_dir / "time" / trace_file):
                rank = int(re.search(r"^step(\d)\.", trace_file).group(1))
                trace_file = (self.trace_dir / "time" / trace_file).as_posix()
                self.analyzers[rank] = TraceAnalysis(trace_files={rank: trace_file})

        os.makedirs(self.trace_dir / "time" / "output", exist_ok=True)
        os.makedirs(self.trace_dir / "space" / "output", exist_ok=True)

    def __call__(self) -> None:
        if not os.path.exists(self.trace_dir):
            return

        self.get_temporal_breakdown()
        self.get_idle_time_breakdown()
        self.get_gpu_kernel_breakdown()
        self.get_memory_bw_time_series()
        self.get_memory_bw_summary()
        self.get_queue_length_time_series()
        self.get_queue_length_summary()
        self.get_cuda_kernel_launch_stats()
        self.export_memory_timeline()

    # Temporal breakdown
    def get_temporal_breakdown(self) -> None:
        for rank, analyzer in self.analyzers.items():
            df = analyzer.get_temporal_breakdown(visualize=False)
            df.to_csv(
                self.trace_dir / "time" / "output" / f"temporal_breakdown.{rank}.df",
                index=False,
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
            ).save(
                self.trace_dir / "time" / "output" / f"temporal_breakdown.{rank}.png",
                verbose=False,
            )

    # Idle time breakdown
    def get_idle_time_breakdown(self) -> None:
        for rank, analyzer in self.analyzers.items():
            df, _ = analyzer.get_idle_time_breakdown(ranks=[rank], visualize=False)
            df.to_csv(
                self.trace_dir / "time" / "output" / f"idle_time_breakdown.{rank}.df",
                index=False,
            )
            (
                ggplot(
                    data=df,
                    mapping=aes(x="stream", y="idle_time", fill="idle_category"),
                )
                + geom_bar(position="stack", stat="identity")
            ).save(
                self.trace_dir / "time" / "output" / f"idle_time_breakdown.{rank}.png",
                verbose=False,
            )

    # Kernel breakdown
    def get_gpu_kernel_breakdown(self) -> None:
        for rank, analyzer in self.analyzers.items():
            df, _ = analyzer.get_gpu_kernel_breakdown(visualize=False)
            df.to_csv(
                self.trace_dir / "time" / "output" / f"gpu_kernel_breakdown.{rank}.df",
                index=False,
            )
            fig, ax = plt.subplots()
            ax.pie(df["percentage"], labels=df["kernel_type"])
            fig.savefig(
                self.trace_dir / "time" / "output" / f"gpu_kernel_breakdown.{rank}.png"
            )

    # Memory bandwidth time series
    def get_memory_bw_time_series(self) -> None:
        for rank, analyzer in self.analyzers.items():
            dfs = analyzer.get_memory_bw_time_series(ranks=[rank])
            for _, df in dfs.items():
                df.to_csv(
                    self.trace_dir
                    / "time"
                    / "output"
                    / f"memory_bw_time_series.{rank}.df",
                    index=False,
                )

    # Memory bandwidth summary
    def get_memory_bw_summary(self) -> None:
        for rank, analyzer in self.analyzers.items():
            df = analyzer.get_memory_bw_summary(ranks=[rank])
            df.to_csv(
                self.trace_dir / "time" / "output" / f"memory_bw_summary.{rank}.df"
            )

    # Queue length time series
    def get_queue_length_time_series(self) -> None:
        for rank, analyzer in self.analyzers.items():
            dfs = analyzer.get_queue_length_time_series(ranks=[rank])
            for _, df in dfs.items():
                df.to_csv(
                    self.trace_dir
                    / "time"
                    / "output"
                    / f"queue_length_time_series.{rank}.df",
                    index=False,
                )

    # Queue length summary
    def get_queue_length_summary(self) -> None:
        for rank, analyzer in self.analyzers.items():
            df = analyzer.get_queue_length_summary(ranks=[rank])
            df.to_csv(
                self.trace_dir / "time" / "output" / f"queue_length_summary.{rank}.df"
            )

    # CUDA kernel launch statistics
    def get_cuda_kernel_launch_stats(self) -> None:
        for rank, analyzer in self.analyzers.items():
            dfs = analyzer.get_cuda_kernel_launch_stats(ranks=[rank], visualize=False)
            for _, df in dfs.items():
                df.to_csv(
                    self.trace_dir
                    / "time"
                    / "output"
                    / f"uda_kernel_launch_stats.{rank}.df",
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
                    + geom_histogram(bins=300)
                ).save(
                    self.trace_dir
                    / "time"
                    / "output"
                    / f"cuda_kernel_launch_stats.{rank}.short_kernels.png",
                    verbose=False,
                )

                runtime_outliers = df.query("cpu_duration > @self.runtime_cutoff")
                (
                    ggplot(
                        data=runtime_outliers,
                        mapping=aes(x="cpu_duration"),
                    )
                    + geom_histogram(bins=300)
                ).save(
                    self.trace_dir
                    / "time"
                    / "output"
                    / f"cuda_kernel_launch_stats.{rank}.runtime_outliers.png",
                    verbose=False,
                )

                launch_delay_outliers = df.query(
                    "launch_delay > @self.launch_delay_cutoff"
                )
                (
                    ggplot(
                        data=launch_delay_outliers,
                        mapping=aes(x="launch_delay"),
                    )
                    + geom_histogram(bins=300)
                ).save(
                    self.trace_dir
                    / "time"
                    / "output"
                    / f"cuda_kernel_launch_stats.{rank}.launch_delay_outliers.png",
                    verbose=False,
                )

    def export_memory_timeline(self) -> None:
        for json_gz in os.listdir(self.trace_dir / "space"):
            if not os.path.isfile(self.trace_dir / "space" / json_gz):
                continue

            device, step, _ = os.fspath(json_gz).split(".", 2)
            df = pd.read_json(self.trace_dir / "space" / json_gz)

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
                + theme(axis_text_x=element_text(ha="left", rotation=-45))
            ).save(
                self.trace_dir / "space" / "output" / f"{device}.{step}.png",
                verbose=False,
            )

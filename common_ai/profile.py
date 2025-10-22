import os
import re
import gzip
import pathlib
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_bar, geom_histogram, geom_area
from hta.trace_analysis import TraceAnalysis
import sys


def decode_trace(trace_dir: os.PathLike) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "decode", exist_ok=True)
    for trace_file in os.listdir(trace_dir):
        if os.path.isfile(trace_dir / trace_file):
            with gzip.open(trace_dir / trace_file, "rb") as fd, gzip.open(
                trace_dir / "time" / "decode" / trace_file, "wb"
            ) as wd:
                _ = wd.write(
                    re.sub(
                        r"[\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f]",
                        "sc",
                        fd.read().decode(encoding="ascii", errors="ignore"),
                    ).encode(encoding="ascii")
                )


# Temporal breakdown
def get_temporal_breakdown(analyzer: TraceAnalysis, trace_dir: os.PathLike) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    df = analyzer.get_temporal_breakdown(visualize=False)
    df.to_csv(trace_dir / "time" / "output" / "temporal_breakdown.df", index=False)
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
    ).save(trace_dir / "time" / "output" / "temporal_breakdown.png")


# Idle time breakdown
def get_idle_time_breakdown(analyzer: TraceAnalysis, trace_dir: os.PathLike) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    df, _ = analyzer.get_idle_time_breakdown(visualize=False)
    df.to_csv(trace_dir / "time" / "output" / "idle_time_breakdown.df", index=False)
    (
        ggplot(
            data=df,
            mapping=aes(x="stream", y="idle_time", fill="idle_category"),
        )
        + geom_bar(position="stack", stat="identity")
    ).save(trace_dir / "time" / "output" / "idle_time_breakdown.png")


# Kernel breakdown
def get_gpu_kernel_breakdown(analyzer: TraceAnalysis, trace_dir: os.PathLike) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    df, _ = analyzer.get_gpu_kernel_breakdown(visualize=False)
    df.to_csv(trace_dir / "time" / "output" / "gpu_kernel_breakdown.df", index=False)
    fig, ax = plt.subplots()
    ax.pie(df["percentage"], labels=df["kernel_type"])
    fig.savefig(trace_dir / "time" / "output" / "gpu_kernel_breakdown.png")


# Memory bandwidth time series
def get_memory_bw_time_series(analyzer: TraceAnalysis, trace_dir: os.PathLike) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    dfs = analyzer.get_memory_bw_time_series()
    for rank, df in dfs.items():
        df.to_csv(
            trace_dir / "time" / "output" / f"memory_bw_time_series_{rank}.df",
            index=False,
        )


# Memory bandwidth summary
def get_memory_bw_summary(analyzer: TraceAnalysis, trace_dir: os.PathLike) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    df = analyzer.get_memory_bw_summary()
    df.to_csv(trace_dir / "time" / "output" / "memory_bw_summary.df")


# Queue length time series
def get_queue_length_time_series(
    analyzer: TraceAnalysis, trace_dir: os.PathLike
) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    dfs = analyzer.get_queue_length_time_series()
    for rank, df in dfs.items():
        df.to_csv(
            trace_dir / "time" / "output" / f"queue_length_time_series_{rank}.df",
            index=False,
        )


# Queue length summary
def get_queue_length_summary(analyzer: TraceAnalysis, trace_dir: os.PathLike) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    df = analyzer.get_queue_length_summary()
    df.to_csv(trace_dir / "time" / "output" / "queue_length_summary.df")


# CUDA kernel launch statistics
def get_cuda_kernel_launch_stats(
    analyzer: TraceAnalysis, trace_dir: os.PathLike
) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "time" / "output", exist_ok=True)
    dfs = analyzer.get_cuda_kernel_launch_stats(visualize=False)
    for rank, df in dfs.items():
        df.to_csv(
            trace_dir / "time" / "output" / f"uda_kernel_launch_stats_{rank}.df",
            index=False,
        )
        runtime_cutoff = 50
        launch_delay_cutoff = 100

        short_kernels = df.query(
            "cpu_duration <= @runtime_cutoff and gpu_duration < cpu_duration"
        )
        (
            ggplot(
                data=short_kernels,
                mapping=aes(x="cpu_duration"),
            )
            + geom_histogram()
        ).save(
            trace_dir
            / "time"
            / "output"
            / f"cuda_kernel_launch_stats_{rank}_short_kernels.png"
        )

        runtime_outliers = df.query("cpu_duration > @runtime_cutoff")
        (
            ggplot(
                data=runtime_outliers,
                mapping=aes(x="cpu_duration"),
            )
            + geom_histogram()
        ).save(
            trace_dir
            / "time"
            / "output"
            / f"cuda_kernel_launch_stats_{rank}_runtime_outliers.png"
        )

        launch_delay_outliers = df.query("launch_delay > @launch_delay_cutoff")
        (
            ggplot(
                data=launch_delay_outliers,
                mapping=aes(x="launch_delay"),
            )
            + geom_histogram()
        ).save(
            trace_dir
            / "time"
            / "output"
            / f"cuda_kernel_launch_stats_{rank}_launch_delay_outliers.png"
        )


def export_memory_timeline(
    device: Literal["cpu", "cuda:0"], trace_dir: os.PathLike
) -> None:
    trace_dir = pathlib.Path(os.fspath(trace_dir))
    os.makedirs(trace_dir / "memory" / "output", exist_ok=True)
    df = pd.read_json(trace_dir / "memory" / f"{device}.json.gz")

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
    ).save(trace_dir / "memory" / "output" / f"{device}.png")


if __name__ == "__main__":
    trace_dir = pathlib.Path(sys.argv[1])
    analyzer = TraceAnalysis(trace_dir=(trace_dir / "time").as_posix())
    get_temporal_breakdown(analyzer, trace_dir)
    get_idle_time_breakdown(analyzer, trace_dir)
    get_gpu_kernel_breakdown(analyzer, trace_dir)
    get_memory_bw_time_series(analyzer, trace_dir)
    get_memory_bw_summary(analyzer, trace_dir)
    get_queue_length_time_series(analyzer, trace_dir)
    get_queue_length_summary(analyzer, trace_dir)
    get_cuda_kernel_launch_stats(analyzer, trace_dir)
    export_memory_timeline("cpu", trace_dir)
    export_memory_timeline("cuda:0", trace_dir)

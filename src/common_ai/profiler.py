import torch
import os
import pathlib


class MyProfiler:
    def __init__(
        self,
        warmup: int,
        active: int,
        repeat: int,
        **kwargs,
    ) -> None:
        """Profiler arguments.

        Args:
            warmup: warmup steps in each rounds.
            active: record steps in each rounds.
            repeat: total rounds. 0 means infinity. negative means turn off.
        """
        self.step = -1
        self.repeat = repeat
        if repeat >= 0:
            self.schedule = torch.profiler.schedule(
                wait=0, warmup=warmup, active=active, repeat=repeat
            )

    def set_logs_path(self, logs_path: os.PathLike) -> object:
        self.logs_path = pathlib.Path(os.fspath(logs_path))
        return self

    def start(self) -> None:
        if self.repeat < 0:
            return

        self.step += 1
        if self.schedule(self.step) in [
            torch.profiler.ProfilerAction.NONE,
            torch.profiler.ProfilerAction.WARMUP,
        ]:
            return

        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.logs_path / "profile" / "time",
                worker_name=f"step{self.step}",
                use_gzip=True,
            ),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )
        self.prof.start()

    def stop(self) -> None:
        if self.repeat < 0:
            return

        if self.schedule(self.step) in [
            torch.profiler.ProfilerAction.NONE,
            torch.profiler.ProfilerAction.WARMUP,
        ]:
            return

        self.prof.stop()
        os.makedirs(self.logs_path / "profile" / "space", exist_ok=True)
        self.prof.export_memory_timeline(
            (
                self.logs_path / "profile" / "space" / f"cpu.{self.step}.json.gz"
            ).as_posix(),
            device="cpu",
        )
        self.prof.export_memory_timeline(
            (
                self.logs_path / "profile" / "space" / f"cuda:0.{self.step}.json.gz"
            ).as_posix(),
            device="cuda:0",
        )

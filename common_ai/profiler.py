import torch
import os
import pathlib


class MyProfiler:
    def __init__(
        self,
        wait: int,
        warmup: int,
        active: int,
        repeat: int,
        skip_first: int,
        skip_first_wait: int,
        use_profiler: bool,
    ) -> None:
        """Profiler arguments.

        Args:
            wait: skip steps in each round.
            warmup: warmup steps in each round.
            active: profile steps in each round.
            repeat: total round number, 0 means infinity rounds.
            skip_first: additional skip steps before first round.
            skip_first_wait: whether skip wait before first round.
            use_profiler: whether to use profiler.
        """
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.skip_first = skip_first
        self.skip_first_wait = skip_first_wait
        self.use_profiler = use_profiler

    def start(self, logs_path: os.PathLike) -> None:
        if not self.use_profiler or hasattr(self, "prof"):
            return

        self.logs_path = pathlib.Path(os.fspath(logs_path))
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                self.wait,
                self.warmup,
                self.active,
                self.repeat,
                self.skip_first,
                self.skip_first_wait,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.logs_path / "profile" / "time", use_gzip=True
            ),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )
        self.prof.start()

    def step(self) -> None:
        if not self.use_profiler or hasattr(self, "prof"):
            return
        self.prof.step()

    def stop(self) -> None:
        if not self.use_profiler or hasattr(self, "prof"):
            return
        self.prof.stop()
        os.makedirs(self.logs_path / "profile" / "space", exist_ok=True)
        self.prof.export_memory_timeline(
            (self.logs_path / "profile" / "space" / "cpu.json.gz").as_posix(),
            device="cpu",
        )
        self.prof.export_memory_timeline(
            (self.logs_path / "profile" / "space" / "cuda:0.json.gz").as_posix(),
            device="cuda:0",
        )

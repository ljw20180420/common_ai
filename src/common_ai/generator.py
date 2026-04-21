import torch
import numpy as np


class MyGenerator:
    def __init__(
        self,
        seed: int,
        **kwargs,
    ) -> None:
        """Generator arguments.

        Args:
            seed: Random seed.
        """
        self.seed = seed
        self.np_rng = np.random.default_rng(self.seed)
        self.torch_c_rng = torch.Generator(device="cpu").manual_seed(self.seed)
        if torch.cuda.is_available():
            self.torch_g_rng = torch.Generator(device="cuda").manual_seed(self.seed)

    def get_torch_generator_by_device(
        self, device: str | torch.device
    ) -> torch.Generator:
        if device == "cpu" or device == torch.device("cpu"):
            return self.torch_c_rng
        return self.torch_g_rng

    def state_dict(self) -> dict:
        state_dict = {
            "np_rng": self.np_rng.bit_generator.state,
            "torch_c_rng": self.torch_c_rng.get_state(),
        }
        if torch.cuda.is_available():
            state_dict.update({"torch_g_rng": self.torch_g_rng.get_state()})
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        self.np_rng.bit_generator.state = state_dict["np_rng"]
        self.torch_c_rng.set_state(state_dict["torch_c_rng"])
        if torch.cuda.is_available():
            self.torch_g_rng.set_state(state_dict["torch_g_rng"])

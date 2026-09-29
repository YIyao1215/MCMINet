"""Explicit seed initialization and epoch-boundary RNG state capture."""
from mcminet.config import default
import platform
import random
import numpy as np
import torch
import torchvision
import torch_geometric
from mcminet.training.training_config import DEFAULT_NOTICE


def seed_everything(seed, *, deterministic_algorithms=default("reproducibility.deterministic_algorithms")):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if type(deterministic_algorithms) is not bool:
        raise ValueError("deterministic_algorithms must be bool")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic_algorithms


def seeded_generator(seed):
    """Local generator: no mutation of global Python/NumPy/torch RNGs."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    return torch.Generator(device="cpu").manual_seed(seed)


def seed_worker(worker_id):
    """Optional DataLoader worker_init_fn; use a seeded DataLoader generator."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def run_metadata(config, device):
    return dict(seed=config.seed, config=config.to_dict(), device=str(torch.device(device)),
                numerical_policy=DEFAULT_NOTICE,
                software=dict(python=platform.python_version(), torch=str(torch.__version__),
                              numpy=np.__version__, torchvision=str(torchvision.__version__),
                              torch_geometric=str(torch_geometric.__version__)),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic)


def capture_rng(generator=None):
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("Data/sampler generator must be CPU")
    np_state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=[np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
        torch_cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        generator=None if generator is None else generator.get_state(),
    )


def restore_rng(state, generator=None):
    """Explicit resume action. Missing/changed device topology fails, never degrades."""
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "cuda", "mps", "generator"}:
        raise ValueError("RNG schema mismatch")
    if len(state["cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("CUDA RNG topology mismatch")
    if (state["mps"] is not None) != torch.backends.mps.is_available():
        raise ValueError("MPS RNG availability mismatch")
    if (state["generator"] is not None) != (generator is not None):
        raise ValueError("Sampler generator ownership mismatch")
    # Validate host states with local generators before mutating the global streams.
    random.Random().setstate(state["python"])
    np_state = state["numpy"]
    local_np = np.random.RandomState()
    local_np.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.Generator().set_state(state["torch_cpu"])
    if generator is not None:
        if generator.device.type != "cpu":
            raise ValueError("Data/sampler generator must be CPU")
        torch.Generator().set_state(state["generator"])
    random.setstate(state["python"])
    np.random.set_state(local_np.get_state())
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])
    if generator is not None:
        generator.set_state(state["generator"])

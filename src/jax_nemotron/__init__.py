"""
jax_nemotron — a public, Karpathy-style JAX/Flax(NNX) reimplementation of
NVIDIA Nemotron-3-Nano-Omni.

This package exposes the HF-faithful LLM backbone, its config, and the omni
(text + vision + sound) wrapper:
    from jax_nemotron.config import NemotronHConfig
    from jax_nemotron.nemotron_h import NemotronHModel
    from jax_nemotron.nemotron_omni import NemotronOmni, NemotronOmniConfig
"""

from .config import NemotronHConfig  # noqa: F401
from .nemotron_omni import NemotronOmni, NemotronOmniConfig  # noqa: F401

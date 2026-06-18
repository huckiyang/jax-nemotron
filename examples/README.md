# jax-nemotron examples

Runnable, laptop-CPU demos. No checkpoint and no GPU required — weights are
random-initialized, so the numbers are meaningless; the point is that the
shapes line up and the full pipeline executes.

- `run_tiny_cpu.py` — Builds the **tiny** `NemotronOmni` preset (tiny hybrid
  Mamba/Attention/MoE LLM + tiny vision & audio encoders), prints the config
  and the parsed hybrid layer pattern (`MEM*EM`), then runs ONE fused
  `[vision | sound | text]` forward pass and confirms the logits are finite.
  Ends with `OK`.

Run it from the repo root with the project venv (CPU JAX):

```
.venv/bin/python examples/run_tiny_cpu.py
```

(or just `python examples/run_tiny_cpu.py` if you `pip install -e .`'d the package.)

# Makefile for jax-nemotron.
#
# The project venv at .venv is CPU JAX (jax 0.10.1 / flax 0.12.7). Tests and the
# converter fix up sys.path to find src/ themselves, so `install` is optional for
# running the gates -- but `pip install -e .` makes `import jax_nemotron` work
# from anywhere.

# Prefer the project venv's python so the gates run against the confirmed CPU JAX
# stack regardless of the caller's active environment.
PYTHON ?= .venv/bin/python

# Real HF checkpoint dir (sibling of this repo) used by the dry-run gate.
CKPT_DIR ?= ../Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
OUT      ?= gs://bucket/nemotron-omni-30b-orbax
PRESET   ?= omni_30b
DTYPE    ?= bf16

.PHONY: help install test dry-run

help:
	@echo "Targets:"
	@echo "  install   editable install (uv pip if available, else pip): pip install -e ."
	@echo "  test      run all four standalone shape/coverage gates with $(PYTHON)"
	@echo "  dry-run   converter --dry-run against $(CKPT_DIR) (no weights written)"

# Editable install via uv if present (faster), otherwise the venv pip.
install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv pip install -e ."; \
		uv pip install -e .; \
	else \
		echo "$(PYTHON) -m pip install -e ."; \
		$(PYTHON) -m pip install -e .; \
	fi

# Each tests/*.py is a runnable script (no pytest). Run all four; fail on first error.
test:
	$(PYTHON) tests/test_shape_gate.py
	$(PYTHON) tests/test_omni_shape.py
	$(PYTHON) tests/test_name_coverage.py
	$(PYTHON) tests/test_converter_units.py

# Dry-run validates the name-map bijection + shapes against the index; writes nothing.
dry-run:
	$(PYTHON) scripts/convert_hf_to_orbax.py \
		--ckpt-dir $(CKPT_DIR) \
		--out $(OUT) \
		--preset $(PRESET) \
		--dtype $(DTYPE) \
		--dry-run

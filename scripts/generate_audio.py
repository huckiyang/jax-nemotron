#!/usr/bin/env python3
"""
Audio -> text inference with the converted Nemotron-3-Nano-Omni checkpoints.

Loads the LLM backbone (GS_OUT) and the Parakeet sound encoder + projector
(GS_OUT_SOUND), encodes a .wav, splices the projected audio tokens into the text
sequence at sound_context_token_id (27), and greedy-decodes a response. This is
the audio analogue of generate.py / infer_text.py.

It deliberately rebuilds ONLY the audio path (LLM + sound), NOT the full
NemotronOmni, so the vision tower is never allocated — important because the LLM
alone is ~60GB and lives in host RAM (run on CPU: JAX_PLATFORMS=cpu).

    JAX_PLATFORMS=cpu python scripts/generate_audio.py \
        --orbax       gs://bucket/nemotron-omni-30b-orbax \
        --orbax-sound gs://bucket/nemotron-omni-30b-orbax-sound \
        --tokenizer   /path/to/tokenizer_dir \
        --audio       when_does_cafe_macs_close.wav \
        --prompt "Transcribe this audio and identify key action items." \
        --max-new-tokens 128

NOTE (coherence): the in-script log-mel uses the checkpoint's stored mel
filterbank/window. The exact ParakeetFeatureExtractor normalization is the one
unverified detail — if a transcription comes out garbled, that normalization (or
the audio token framing/markers) is the first thing to revisit.
"""

from __future__ import annotations

import argparse
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# =============================================================================
# Flat-path helpers (slash paths match what the converter wrote)
# =============================================================================


def _normalize_path(path: str) -> str:
    return path if "://" in path else os.path.abspath(path)


def _slash(key_tuple) -> str:
    return "/".join(str(k) for k in key_tuple)


def _insert(tree: dict, slash_path: str, value) -> None:
    node = tree
    parts = slash_path.split("/")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _abstract_target(abstract_state):
    """{"params": <nested ShapeDtypeStruct tree>} for one module's abstract state."""
    import jax
    params: dict = {}
    for key_tuple, var in abstract_state.flat_state():
        val = getattr(var, "value", var)
        _insert(params, _slash(key_tuple), jax.ShapeDtypeStruct(tuple(val.shape), val.dtype))
    return params


def _scatter(abstract_state, restored_nested, graphdef):
    """Set each abstract leaf from restored_nested (looked up by slash path) and merge."""
    import jax.numpy as jnp
    from flax import nnx
    flat = dict(abstract_state.flat_state())
    for key_tuple, var in flat.items():
        node = restored_nested
        for seg in _slash(key_tuple).split("/"):
            node = node[seg]
        var.value = jnp.asarray(node)
    return nnx.merge(graphdef, nnx.State.from_flat_path(flat))


# =============================================================================
# Load the LLM backbone and the sound encoder + projector
# =============================================================================


def load_llm(orbax_path: str, cfg):
    """Restore the NemotronH backbone from the LLM Orbax checkpoint (GS_OUT)."""
    from flax import nnx
    import orbax.checkpoint as ocp
    from jax_nemotron.nemotron_h import NemotronHModel

    abstract = nnx.eval_shape(lambda: NemotronHModel(rngs=nnx.Rngs(0), config=cfg))
    graphdef, abstract_state = nnx.split(abstract)
    print(f"[audio] restoring LLM backbone from {orbax_path!r} ...")
    restored = ocp.StandardCheckpointer().restore(
        _normalize_path(orbax_path), {"params": _abstract_target(abstract_state)}
    )["params"]
    return _scatter(abstract_state, restored, graphdef)


def load_sound(orbax_path: str, omni_cfg):
    """Restore the AudioEncoder + SoundProjector from the sound Orbax checkpoint
    (GS_OUT_SOUND). The checkpoint nests them under "sound_encoder"/"sound_projection"
    (the prefixes the converter wrote); the projector prefix differs from the omni
    attribute name (sound_projector), which is exactly why we restore them here as
    standalone modules rather than into a NemotronOmni."""
    from flax import nnx
    import orbax.checkpoint as ocp
    from jax_nemotron.audio_encoder import AudioEncoder
    from jax_nemotron.nemotron_omni import SoundProjector

    enc_abstract = nnx.eval_shape(lambda: AudioEncoder(omni_cfg.sound, rngs=nnx.Rngs(0)))
    enc_graphdef, enc_state = nnx.split(enc_abstract)

    def _mk_proj():
        return SoundProjector(
            rngs=nnx.Rngs(0),
            in_dim=omni_cfg.sound_proj_in,
            mid_dim=omni_cfg.sound_projector_hidden,
            out_dim=omni_cfg.llm.hidden_size,
            eps=omni_cfg.llm.norm_eps,
        )

    proj_abstract = nnx.eval_shape(_mk_proj)
    proj_graphdef, proj_state = nnx.split(proj_abstract)

    # The checkpoint params tree: {"sound_encoder": <enc>, "sound_projection": <proj>}.
    target = {"params": {
        "sound_encoder": _abstract_target(enc_state),
        "sound_projection": _abstract_target(proj_state),
    }}
    print(f"[audio] restoring sound encoder + projector from {orbax_path!r} ...")
    restored = ocp.StandardCheckpointer().restore(
        _normalize_path(orbax_path), target
    )["params"]

    encoder = _scatter(enc_state, restored["sound_encoder"], enc_graphdef)
    projector = _scatter(proj_state, restored["sound_projection"], proj_graphdef)
    return encoder, projector


# =============================================================================
# Audio loading (pure numpy/scipy; resample to 16 kHz mono)
# =============================================================================


def load_waveform(path: str, target_sr: int = 16000):
    """Read a .wav to a (1, T) float32 array at target_sr, mono."""
    import numpy as np
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim == 2:                      # stereo -> mono
        data = data.mean(axis=1)
    # int PCM -> float [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    print(f"[audio] loaded {path}: {len(data)} samples @ {target_sr} Hz "
          f"({len(data)/target_sr:.1f}s)")
    return data[None, :]                    # (1, T)


# =============================================================================
# Decode
# =============================================================================


def generate_audio(lm, cfg, omni_cfg, input_ids, aud_tokens, max_new_tokens, eos_id=None):
    """Greedy decode with the (precomputed) audio tokens spliced at id-27.

    Each step re-embeds the growing text, splices the fixed sound tokens in place
    at id-27 positions, and runs the LLM backbone (mirrors NemotronOmni.__call__
    minus vision)."""
    import jax.numpy as jnp
    import numpy as np
    from jax_nemotron.nemotron_omni import _splice_modality

    snd_id = omni_cfg.sound_context_token_id
    chunk = cfg.chunk_size
    n_aud = int(aud_tokens.shape[1])

    ids = [int(x) for x in np.asarray(input_ids).reshape(-1)]
    n_placeholders = sum(1 for t in ids if t == snd_id)
    if n_placeholders != n_aud:
        raise ValueError(
            f"prompt has {n_placeholders} sound placeholders (id {snd_id}) but the "
            f"encoder produced {n_aud} tokens; they must match for the splice.")

    def _forward(seq_ids):
        x = lm.embeddings(jnp.asarray(seq_ids, dtype=jnp.int32)[None, :])
        x = _splice_modality(x, jnp.asarray(seq_ids)[None, :], snd_id, aud_tokens)
        for layer in lm.layers:
            x = layer(x)
        return lm.lm_head(lm.norm_f(x))     # (1, L, vocab)

    generated = []
    for _ in range(max_new_tokens):
        real_len = len(ids)
        padded_len = ((real_len + chunk - 1) // chunk) * chunk
        seq = ids + [0] * (padded_len - real_len)   # pad with 0 (not the sound id)
        logits = _forward(seq)
        next_id = int(np.argmax(np.asarray(logits[0, real_len - 1])))
        generated.append(next_id)
        ids.append(next_id)
        if eos_id is not None and next_id == int(eos_id):
            break
    return generated


# =============================================================================
# Main
# =============================================================================


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--orbax", required=True, help="LLM Orbax checkpoint (GS_OUT).")
    p.add_argument("--orbax-sound", required=True, help="Sound Orbax checkpoint (GS_OUT_SOUND).")
    p.add_argument("--tokenizer", required=True, help="Tokenizer dir or HF repo id.")
    p.add_argument("--audio", required=True, help="Path to a .wav file.")
    p.add_argument("--preset", default="omni_30b")
    p.add_argument("--prompt", default="Transcribe this audio.",
                   help="Text instruction placed AFTER the audio tokens.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    from jax_nemotron.nemotron_omni import NemotronOmniConfig

    omni_cfg = NemotronOmniConfig.from_preset(args.preset)
    omni_cfg.validate()

    lm = load_llm(args.orbax, omni_cfg.llm)
    encoder, projector = load_sound(args.orbax_sound, omni_cfg)
    print(f"[audio] loading tokenizer {args.tokenizer!r} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    waveform = load_waveform(args.audio, omni_cfg.sound.sample_rate)

    # Encode the audio ONCE -> projected sound tokens (1, n_aud, llm_hidden).
    import jax.numpy as jnp
    aud_tokens = projector(encoder(jnp.asarray(waveform)))
    n_aud = int(aud_tokens.shape[1])
    print(f"[audio] sound encoder -> {n_aud} tokens of width {aud_tokens.shape[-1]}")

    # Build the sequence: [sound placeholders] + [text instruction]. (A first-cut
    # layout; the exact HF processor wraps the audio in <so_start>/<so_end> and the
    # chat template — revisit here if transcription quality is poor.)
    text_ids = list(tok(args.prompt, return_tensors="np")["input_ids"][0])
    input_ids = [omni_cfg.sound_context_token_id] * n_aud + text_ids

    gen = generate_audio(lm, omni_cfg.llm, omni_cfg, input_ids, aud_tokens,
                         args.max_new_tokens, tok.eos_token_id)
    print("=" * 70)
    print(f"AUDIO   : {args.audio}")
    print(f"PROMPT  : {args.prompt}")
    print(f"RESPONSE: {tok.decode(gen, skip_special_tokens=True)}")
    print("=" * 70)
    print("[audio] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Milestone-2 Audio: Nemotron-Omni sound path (HF -> Orbax)

Build spec for converting the Parakeet FastConformer sound encoder + projector.
Ground truth: the HF checkpoint's `audio_model.py`, `modeling.py`, `processing.py`,
`config.json` (`sound_config`), and `model.safetensors.index.json`.

## Real config (config.json `sound_config`)
- hidden 1024, heads 8, head_dim 128, layers 24, d_ff 4096
- depthwise conv_kernel = **9** (NOT 31), convolution_bias false
- subsampling: conv_channels 256, kernel 3, stride 2, factor 8 (3 stride-2 stages, 2-D)
- num_mel_bins 128, projection_hidden 4096, projection_bias false, sample_rate 16000
- window 400 (25ms), hop 160 (10ms), n_fft 512 (rfft 257)

## Tensor inventory (713 = 710 encoder + 3 projection)
- feature_extractor: `featurizer.fb (1,128,257)`, `featurizer.window (400,)` — frozen buffers (raw)
- subsampling (12): `layers.{0,2,5}.{weight (256,1,3,3),bias}` depthwise conv2d stride2;
  `layers.{3,6}.{weight (256,256,1,1),bias}` pointwise conv2d; `linear.{weight (1024,4096),bias}`
- layers.{0..23} (29 each = 696): macaron post-norm conformer
  - norm_feed_forward1 {weight,bias} (LayerNorm); feed_forward1.linear1 (4096,1024), linear2 (1024,4096)
  - norm_self_att {weight,bias}; self_attn {q,k,v,o}_proj (1024,1024) no bias,
    relative_k_proj (1024,1024) no bias, bias_u (8,128), bias_v (8,128)
  - norm_conv {weight,bias}; conv.pointwise_conv1 (2048,1024,1), conv.depthwise_conv (1024,1,9),
    conv.norm BatchNorm1d {weight,bias,running_mean,running_var, num_batches_tracked=SKIP},
    conv.pointwise_conv2 (1024,1024,1)
  - norm_feed_forward2 {weight,bias}; feed_forward2.linear1 (4096,1024), linear2 (1024,4096)
  - norm_out {weight,bias}
- sound_projection (3, top-level): norm.weight (1024) RMSNorm; linear1 (4096,1024); linear2 (2688,4096); no bias.
  Forward order: **norm(1024) -> linear1 -> relu(x)^2 -> linear2**.

## Transforms
- raw: all LayerNorm/BatchNorm weight+bias, BN running_mean/var, bias_u/bias_v, RMSNorm scale, fb/window, conv biases
- T: all Linear weights (FF, q/k/v/o/relative_k proj, subsampling.linear, projection linear1/2)
- conv1d (= existing `conv`, axes 2,1,0): pointwise_conv1/2, depthwise_conv
- conv2d (NEW, axes 2,3,1,0): subsampling layers.{0,2,5,3,6}.weight
- SKIP: `conv.norm.num_batches_tracked` (I64 buffer, not a param) — expected sole unconsumed tensor

## Faithfulness gaps in src/jax_nemotron/audio_encoder.py (full rewrite)
RMSNorm+prenorm -> LayerNorm post-norm macaron; plain MHA -> Transformer-XL rel-pos (add relative_k_proj,
bias_u, bias_v); RMSNorm conv module -> two Conv1d pointwise + depthwise k=9 + BatchNorm1d (running stats,
inference uses them); 1-D subsampling -> 2-D Conv2d depthwise/pointwise stack + final linear; load stored
`featurizer.fb`/`window` instead of synthesizing; no pre-emphasis. Fix nemotron_omni SoundProjector order
(norm->linear1->relu^2->linear2, mid=4096, no bias) and omni preset (conv_kernel 31->9).

## Audio preprocessing (generate_audio.py, pure JAX/numpy)
librosa.load sr=16000 mono -> STFT n_fft=512 win=400 hop=160 center=True -> |.|^2 -> mel via stored fb
(1,128,257) -> log -> (Parakeet per-feature normalization: CONFIRM against ParakeetFeatureExtractor at
coherence) -> (B, n_frames, 128). n_frames = 1 + L//160; tokens = apply 3x (L-1)//2+1.

## Fusion
sound_context_token_id=27. tokens-per-clip via subsampling math; splice via in-place scatter at id-27
positions (nemotron_omni._splice_modality already faithful). Assert sound_mask.sum()==n_sound_tokens.

## Gate (objective, CPU, no weights)
eval_shape the faithful encoder+projector -> flat param paths; bijection vs index keys under
`sound_encoder.`/`sound_projection.` minus num_batches_tracked. Mirror tests/test_name_coverage.py.
Plus tiny CPU forward of fused [sound|text] -> finite logits.

## Open item (coherence-time)
Exact ParakeetFeatureExtractor normalization/log/window semantics unread (transformers not in venv) —
a mel-norm mismatch is a right-shape/wrong-value trap; confirm before/at the coherence decode.

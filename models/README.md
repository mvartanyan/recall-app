# Vendored speaker model

`spkrec-ecapa-voxceleb.onnx` is the official WeSpeaker
ECAPA-TDNN-512 large-margin model trained on VoxCeleb2 Dev. The upstream file
is `voxceleb_ECAPA512_LM.onnx` from:

https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM

Recall keeps the historical local filename so existing Tauri resource paths do
not move. `scripts/add_sherpa_metadata.py` adds only the metadata required by
sherpa-onnx; it does not modify graph nodes or weights.

- Upstream SHA-256 before metadata: `d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0`
- Vendored SHA-256 after metadata: `da11c87ed452e72087beb6f2fe8a2abc0ef722c2f9a641c373678a0917a07e07`
- Recall embedding pipeline: `wespeaker-ecapa512-lm-v4-vad`
- Output dimension: 192

The upstream model card declares CC BY 4.0 and contains the model's citation
and benchmark details.

## Silero voice activity detector

`silero_vad.onnx` is the sherpa-onnx export of Silero VAD distributed by the
k2-fsa project:

https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

- SHA-256: `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6`
- Input sample rate: 16 kHz mono
- Upstream license: MIT
- Recall embedding pipeline: `wespeaker-ecapa512-lm-v4-vad`

Run `python3 scripts/fetch_vad_model.py` to fetch and verify the pinned file.

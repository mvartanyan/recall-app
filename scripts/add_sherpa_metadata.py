#!/usr/bin/env python3
"""Add the metadata sherpa-onnx needs to the vendored SpeechBrain ECAPA model.

This is a one-time, reproducible model packaging step. It changes only ONNX
metadata; model weights and graph nodes are left untouched.
"""

from pathlib import Path

import onnx


MODEL = Path(__file__).resolve().parents[1] / "models" / "spkrec-ecapa-voxceleb.onnx"
METADATA = {
    "framework": "wespeaker",
    "language": "multilingual",
    "url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb",
    "comment": "SpeechBrain ECAPA-TDNN exported for Recall",
    "sample_rate": "16000",
    "output_dim": "192",
    "normalize_samples": "0",
}


def main() -> None:
    model = onnx.load(MODEL)
    existing = {item.key: item for item in model.metadata_props}
    for key, value in METADATA.items():
        if key in existing:
            existing[key].value = value
        else:
            item = model.metadata_props.add()
            item.key = key
            item.value = value
    onnx.save(model, MODEL)
    print(f"Updated sherpa-onnx metadata in {MODEL}")


if __name__ == "__main__":
    main()

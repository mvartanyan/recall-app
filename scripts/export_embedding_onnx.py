"""
Export SpeechBrain ECAPA (spkrec-ecapa-voxceleb) to ONNX for embedding inference.
Outputs models/spkrec-ecapa-voxceleb.onnx and stores downloaded weights under models/cache.
"""
import sys
import os
import pathlib
import torch
from speechbrain.pretrained import EncoderClassifier

if sys.version_info[:2] != (3, 11):
    raise SystemExit("Use Python 3.11 for export; torch/onnx export path is unstable on 3.12/3.13")

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
CACHE_DIR = MODEL_DIR / "cache"
ONNX_PATH = MODEL_DIR / "spkrec-ecapa-voxceleb.onnx"

MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Force legacy exporter to avoid torch.export symbolic errors on variable-length audio.
os.environ.setdefault("TORCH_ONNX_DISABLE_DYNAMO_EXPORT", "1")

print("Loading SpeechBrain encoder (this may download weights)…")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(CACHE_DIR),
    run_opts={"device": "cpu"},
)

class EncoderWrapper(torch.nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def forward(self, waveforms: torch.Tensor):
        # waveforms: [batch, time]
        emb = self.cls.encode_batch(waveforms)
        # encode_batch returns [batch, 1, emb] sometimes; squeeze to [batch, emb]
        return emb.squeeze(1)

wrapper = EncoderWrapper(classifier)
wrapper.eval()

sample_rate = classifier.audio_normalizer.sample_rate
example = torch.randn(1, sample_rate * 3)  # 3s of audio

# Trace to freeze shapes and bypass torch.export ONNX path.
traced = torch.jit.trace(wrapper, example)
traced = torch.jit.freeze(traced)

print(f"Exporting to ONNX at {ONNX_PATH}…")
torch.onnx.export(
    traced,
    example,
    ONNX_PATH,
    input_names=["waveform"],
    output_names=["embedding"],
    dynamic_axes=None,  # fixed-size export; caller should pad/trim to sample_rate * 3
    opset_version=14,
    )
print("Done. ONNX saved.")

#!/usr/bin/env python3
"""Fetch Recall's pinned sherpa-onnx-compatible Silero VAD model."""

from hashlib import sha256
from pathlib import Path
from subprocess import run


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "silero_vad.onnx"
URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/silero_vad.onnx"
)
SHA256 = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL.with_suffix(".download")
    try:
        run(
            ["curl", "-L", "--fail", "--silent", "--show-error", URL, "-o", str(temporary)],
            check=True,
        )
        actual = digest(temporary)
        if actual != SHA256:
            raise SystemExit(
                f"Refusing unexpected VAD model: wanted {SHA256}, got {actual}"
            )
        temporary.replace(MODEL)
        print(f"Fetched {MODEL} ({actual})")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

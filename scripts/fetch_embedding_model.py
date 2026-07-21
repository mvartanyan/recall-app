#!/usr/bin/env python3
"""Fetch and package Recall's pinned official WeSpeaker ECAPA ONNX model."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

from add_sherpa_metadata import MODEL, main as add_metadata


URL = (
    "https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/main/"
    "voxceleb_ECAPA512_LM.onnx"
)
UPSTREAM_SHA256 = "d71b85d9b48058ef68004f04f1b78acebefb9dfcf542e19b976a12a5ad1f10b0"
PACKAGED_SHA256 = "da11c87ed452e72087beb6f2fe8a2abc0ef722c2f9a641c373678a0917a07e07"


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    temporary = MODEL.with_suffix(".download")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(URL) as response, temporary.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        actual = digest(temporary)
        if actual != UPSTREAM_SHA256:
            raise SystemExit(
                f"Refusing unexpected upstream model: wanted {UPSTREAM_SHA256}, got {actual}"
            )
        temporary.replace(MODEL)
        add_metadata()
        packaged = digest(MODEL)
        if packaged != PACKAGED_SHA256:
            raise SystemExit(
                f"Packaged model checksum changed: wanted {PACKAGED_SHA256}, got {packaged}"
            )
        print(f"Packaged {MODEL} ({packaged})")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

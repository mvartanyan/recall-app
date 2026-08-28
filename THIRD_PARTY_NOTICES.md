# Recall third-party notices

Recall-owned source is licensed separately under MIT OR Apache-2.0. The
following material and dependencies are not relicensed by Recall.

## WeSpeaker ECAPA-TDNN-512 large-margin speaker model

Recall includes the WeSpeaker ECAPA-TDNN-512 large-margin model trained on
VoxCeleb2 Dev. The original `voxceleb_ECAPA512_LM.onnx` model is published by
the WeSpeaker project at:

https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM

The model is made available under the Creative Commons Attribution 4.0
International license (CC BY 4.0):

https://creativecommons.org/licenses/by/4.0/

Recall distributes it as `spkrec-ecapa-voxceleb.onnx`. The ONNX graph and
weights are unchanged. Recall adds sherpa-onnx compatibility metadata and uses
the model locally to generate speaker embeddings. Provenance and checksums are
recorded in `models/README.md` in the Recall source repository.

## Silero voice activity detector

Recall includes the sherpa-onnx export of Silero VAD as `silero_vad.onnx` and
uses it locally to reject silence, keyboard noise, and other non-speech audio
before generating speaker embeddings. Silero VAD is published under the MIT
License:

https://github.com/snakers4/silero-vad/blob/master/LICENSE

The exact artifact URL and SHA-256 are recorded in `models/README.md`.

## Application dependencies

Recall links open-source npm and Rust crates under their respective declared
licenses. The current Apple-silicon macOS dependency graph includes MIT,
Apache-2.0, BSD, ISC, Zlib, Unicode, CC0, Unlicense, Boost Software License,
CDLA-Permissive-2.0, and MPL-2.0 terms or compatible combinations of them.
MPL-2.0 remains a file-level copyleft license for the covered dependency files;
Recall does not modify those dependency files.

Exact package names, versions, source repositories, and declared license
expressions are pinned by `package-lock.json` and `src-tauri/Cargo.lock` and
can be checked with:

~~~sh
npm run audit:licenses
~~~

Binary distributors should regenerate and review complete dependency notices
for the exact release lockfiles rather than treating this summary as legal
advice.

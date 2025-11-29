# Recall desktop app

Desktop-first Tauri + Rust app with web UI.

## Stack
- Tauri 2.x target (Rust backend in `src-tauri`), vanilla JS frontend (`src`).
- System tray wiring (needs Tauri 2 tray API hookup).
- Rust recording manager: threads + cpal/hound/tempfile; start/stop commands exposed.

## Audio capture plan (desktop)
- Windows: WASAPI loopback + mic capture (cpal/wasapi bindings).
- macOS: capture via user-installed virtual device (e.g., BlackHole) + mic; document routing.
- Linux: PulseAudio/PipeWire monitor sources + mic.

## STT integration (Azure)
- Batch-only (no realtime). Configure `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`.
- Pipeline: record -> chunk -> send to Azure STT with diarization -> run speaker embeddings locally -> discard raw audio after processing.

## Speaker embeddings
- Use SpeechBrain ECAPA (Apache 2.0 friendly). ONNX prebuilt from `Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024` (Apache-2.0) stored at `models/spkrec-ecapa-voxceleb.onnx`.
- Optional export script remains in `scripts/export_embedding_onnx.py` but current torch/export combo is flaky on Python 3.13; prefer the prebuilt ONNX.

## Privacy / storage
- Do not persist raw audio. Store transcripts + embeddings only; encrypt at rest.
- Optional encrypted backups with user-supplied password (no recovery if lost).

## Dev scripts
```
npm install
npm run dev
```
Rust: `cd src-tauri && cargo check`

Tests: `cd src-tauri && cargo test` (unit-less, but ensures build).

## Local dev notes (macOS)
- First-run reset: `rm -rf "$HOME/Library/Application Support/com.example.recall"` to wipe config/DB and trigger first-run flow again.
- API dev (default port 8787):
  ```
  cd ../api
  . .venv/bin/activate
  export AZURE_SPEECH_REGION=germanywestcentral
  export AZURE_SPEECH_KEY=...
  export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=recall1;AccountKey=...;EndpointSuffix=core.windows.net"
  export RECALL_STORAGE_CONTAINER=recall1
  uvicorn app.main:app --port 8787
  ```
- App dev:
  ```
  cd app
  npm install   # safe: node_modules is gitignored
  npm run dev   # or cd src-tauri && cargo run
  ```
- Default API base: `http://localhost:8787` (override via UI input if needed).

Model export (one-time):
```
# already vendored ONNX in models/, no action required
# optional: rebuild with SpeechBrain if you want a fresh export (Python 3.11 required)
# python3.11 -m venv .venv && . .venv/bin/activate
# pip install -r scripts/export_requirements.txt
# python scripts/export_embedding_onnx.py
```

## Current build status
- App now compiles against Tauri 2.5.x; recording manager runs in a worker thread (cpal/hound/tempfile).
- Tray wired via Tauri 2 tray API (open/start/stop/quit).
- ONNX model vendored; Azure STT used when API is configured (stub fallback if not).
- Transcription command `transcribe_file` posts last recording to API base (`/v1/transcribe`); expects Azure-backed response with diarization segments but falls back to stub when Azure is not configured.
- SQLite-based storage added (encrypted columns if password supplied later; currently opens without password). Stores sessions/transcripts, diarized segments, speakers, and embeddings; audio file is deleted after persistence.
- Embedding/voice matching wired: diarized segments are aggregated (~10s per speaker), embeddings computed locally via ONNX, cosine-matched to stored speakers, and new speakers created when no match crosses the threshold.
- API upload now flows through `/v1/transcribe`; `/v1/transcribe-local` remains only as a dev stub.

## API usage from app
- Default API base: `http://localhost:8787`. The UI has an input to override, or set `RECALL_API_BASE`.
- Currently calls `/v1/transcribe` (with stub fallback when Azure is not configured).

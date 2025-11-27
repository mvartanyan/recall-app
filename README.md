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

Model export (one-time):
```
# already vendored ONNX in models/, no action required
# optional: rebuild with SpeechBrain if you want a fresh export (Python 3.11 recommended)
```

## Current build status
- App now compiles against Tauri 2.5.x; recording manager runs in a worker thread (cpal/hound/tempfile).
- Tray wired via Tauri 2 tray API (open/start/stop/quit).
- ONNX model vendored; Azure STT not yet integrated.
- Transcription command `transcribe_file` posts last recording to API base (default `http://localhost:8000/v1/transcribe-local`); returns transcript string.
- SQLite-based storage added (encrypted columns if password supplied later; currently opens without password). Stores sessions/transcripts; API call results persisted and audio file deleted afterward.
- Embedding/voice matching scaffold present (ONNX via `ort`), but diarization → embeddings → matching not wired yet.
- API call currently uses dev stub; switch to `/v1/transcribe` once Azure flow is connected.

## API usage from app
- Default API base: `http://localhost:8000`. The UI has an input to override, or set `RECALL_API_BASE`.
- Currently calls `/v1/transcribe-local` (stub) after stopping; switch to `/v1/transcribe` once Azure backend is ready.

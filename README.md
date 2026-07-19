# Recall desktop

Recall is a macOS-first Tauri desktop app for recording meetings, showing live captions, producing a final diarized transcript, and identifying recurring voices with a local voice database.

The active app is standalone:

- It connects directly to Soniox with the user's own API key.
- The key is stored in macOS Keychain.
- It does not require FastAPI, Uvicorn, Azure, Blob Storage, or a localhost service.
- Conversations, transcript interventions, speaker names, and voiceprints stay in local SQLite.
- Temporary raw audio is deleted after final transcription and local voiceprint extraction.

## Current capabilities

- Capture a selectable macOS audio input as mono 16-bit WAV.
- Stream provisional live captions through Soniox stt-rt-v5.
- Run higher-quality final transcription through Soniox stt-async-v5.
- Enable speaker diarization, language identification, and code-switching-friendly language hints.
- Merge adjacent interventions from the same diarized speaker.
- Store conversations locally and edit their title, speaker assignment, and transcript text.
- Build 192-dimensional ECAPA voiceprints locally through sherpa-onnx.
- Match recurring voices conservatively; unmatched voices become VOICE1, VOICE2, and so on.
- Preview the temporary excerpt for a new voice, give it a name, or assign it to an existing named person.
- Keep both voice patterns or replace an old pattern when assigning a changed voice.
- Delete profiles and conversations.
- Record again while one or more completed recordings are still being processed.
- Inspect all transcription and attribution stages in the in-app Activity drawer.
- Start or stop recording from the macOS menu-bar item.

Recall intentionally does not generate summaries yet.

## First-time setup on macOS

You need:

1. A Soniox API key.
2. Rust installed with [rustup](https://rustup.rs/).
3. A project-local Node/npm toolchain, preferably through fnm, nvm, mise, or Volta.

Example with fnm:

~~~sh
brew install fnm
fnm install --lts
fnm use --lts
cd /Users/michael/dev/recall/app
npm ci
npm run dev
~~~

npm ci installs packages into app/node_modules/; it does not install Recall packages globally. node_modules/, Rust build output, editor files, and local environment files are ignored by Git. package-lock.json is intentionally tracked.

On first launch:

1. Open **Settings**.
2. Paste a Soniox API key and choose **Save key**. Recall writes it to macOS Keychain and clears the input.
3. Choose an audio input.
4. Adjust likely languages if needed. The defaults are en, fr, de, es, ru; these are hints, not a restriction on code-switching.
5. Leave live captions enabled unless you only want final transcription.

macOS should request microphone access when recording starts. If it does not, open **System Settings → Privacy & Security → Microphone** and enable Recall (or the development binary), then restart the app.

## Capturing meeting and system audio

Recall currently records one CoreAudio input device. A microphone works directly. To capture a remote meeting as well, choose a virtual or aggregate input such as BlackHole that combines the meeting output and microphone.

Native ScreenCaptureKit system-audio capture is not implemented yet. This is the main macOS capture limitation; the app no longer pretends that microphone input alone is full desktop capture.

## Development commands

From app/:

~~~sh
npm ci
npm test
npm run lint
npm run dev
~~~

Rust checks:

~~~sh
cd src-tauri
cargo test --offline
~~~

Build the native executable without launching or bundling it:

~~~sh
npm run build -- --debug --no-bundle
~~~

The resulting development binary is:

~~~text
app/src-tauri/target/debug/recall
~~~

Speaker-model smoke test with a mono WAV:

~~~sh
cd src-tauri
cargo run --offline --example check_speaker_model -- \
  ../models/spkrec-ecapa-voxceleb.onnx /path/to/sample.wav
~~~

The model metadata patch is reproducible with scripts/add_sherpa_metadata.py. It changes only ONNX metadata so sherpa-onnx can supply the model's required 80-bin feature input; it does not alter model weights.

## Local data and privacy

- App data: ~/Library/Application Support/com.example.recall/
- Main database: recall.db
- One-time pre-migration backup: recall.pre-standalone-v1.db
- Soniox credential: macOS Keychain service com.example.recall.soniox, account api-key
- Temporary recordings: the macOS temporary directory; deleted after processing or failure
- Soniox uploads/transcriptions: deleted after final transcript retrieval on a best-effort basis
- New-speaker preview: retained locally only while the profile remains an unnamed VOICE<n>; deleted on naming or assignment to a named profile

Transcript fields and embedding vectors have application-level AES-GCM support in the schema, but safe migration to encrypted-at-rest storage is not implemented. Existing databases therefore remain unencrypted unless they were already configured with the older password mode. Recall refuses the old destructive “enable encryption” path instead of recreating the database.

To repeat first-run testing without deleting data, quit Recall and move the app-data directory aside:

~~~sh
mv "$HOME/Library/Application Support/com.example.recall" \
  "$HOME/Library/Application Support/com.example.recall.saved"
~~~

Remove the Recall Soniox item with Keychain Access, or use **Settings → Remove**. Move the saved directory back when finished. Choose a different backup name if .saved already exists.

## Voice identification design

Soniox diarization answers “who spoke when in this recording” but does not provide a persistent cross-meeting identity. Recall handles persistent identity locally:

1. Final Soniox tokens are grouped into contiguous speaker interventions.
2. Up to about 12 seconds of audio is gathered for each diarized speaker.
3. sherpa-onnx computes the ECAPA embedding with the correct feature-extraction frontend.
4. Recall compares it only with voiceprints produced by this versioned pipeline.
5. A match requires cosine similarity of at least 0.90 and a margin of at least 0.04 over the runner-up.
6. Diarized speakers from the same new meeting are never matched against profiles created earlier in that same meeting, preventing four new voices from collapsing into the first one.

These thresholds are deliberately conservative: duplicate VOICE<n> profiles are preferable to silently attributing words to the wrong person. The user can merge duplicates while choosing whether to preserve or replace the prior voice pattern.

## Known limitations

- A valid, newly rotated Soniox key is still needed for a full live network smoke test.
- Live speaker labels are provisional and may shift; final async diarization is authoritative.
- Voice-match thresholds need calibration against more real meetings, microphones, illnesses, and languages.
- Native macOS system-audio capture is pending; virtual/aggregate input is the current route.
- Local database encryption migration is pending.
- The legacy bundle identifier com.example.recall is retained to avoid silently moving existing user data and Keychain entries.
- App signing, notarization, auto-update, and release packaging are not configured.

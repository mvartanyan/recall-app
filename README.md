# Recall desktop

Recall is a macOS-first Tauri desktop app for recording meetings, showing live captions, producing a final diarized transcript, identifying recurring voices with a local voice database, and optionally generating an on-demand OpenAI recap.

The active app is standalone:

- It connects directly to Soniox with the user's own API key.
- The key is stored in a local user-only file with `0600` permissions so it can
  be reused without recurring macOS Keychain password prompts.
- It does not require FastAPI, Uvicorn, Azure, Blob Storage, or a localhost service.
- Conversations, transcript interventions, speaker names, and voiceprints stay in local SQLite.
- Temporary raw audio is deleted after final transcription and local voiceprint extraction.
- OpenAI is optional and receives transcript/agenda content only when the user explicitly chooses **Recap**.

## Current capabilities

- Capture a selectable macOS audio input as mono 16-bit WAV.
- Use the same transparent frontal-brain-with-headset identity mark in the
  macOS Dock and the in-app sidebar; the sidebar mark spans the combined height
  of the product name and subtitle.
- Stream provisional live captions through Soniox stt-rt-v5.
- Run higher-quality final transcription through Soniox stt-async-v5.
- Enable speaker diarization, language identification, and code-switching-friendly language hints.
- Merge adjacent interventions from the same diarized speaker.
- Store conversations locally and edit their title, speaker assignment, and transcript text.
- Put intervention time and a wide person selector on one metadata line above
  the intervention text, so long human names remain readable without narrowing
  the transcript. Editors expand to show the full intervention and are
  remeasured after processing becomes visible or the window width changes.
- Build 192-dimensional ECAPA voiceprints locally through sherpa-onnx.
- Match only named people, conservatively and at most once per recording;
  unmatched or ambiguous voices become VOICE1, VOICE2, and so on.
- Preview the temporary excerpt for a new voice, give it a name, or assign it to an existing named person.
- Show provisional profiles as `Not auto-matched`; **Name person** enables
  recognition, while **Rename person** changes a human-readable name later.
- Keep both voice patterns or replace an old pattern when assigning a changed voice.
- Show only voices attributed in the selected conversation in the right pane,
  while keeping every profile manageable in a separate **Voice Library**.
- Filter historical conversations by a selected named or provisional voice.
- Delete profiles and conversations. Conversation deletion transactionally
  removes orphan unnamed `VOICE<n>` profiles, samples, and voiceprints, while
  preserving named people and provisional voices referenced by another
  conversation.
- Keep profile deletion in the full Voice Library. Named people referenced by
  conversation history are marked `History protected` and cannot be deleted;
  provisional-profile deletion explicitly warns that its attributed history
  will become Unknown speaker.
- Confirm destructive actions in a visible app-owned overlay, including when
  the Voice Library modal is already open.
- Record again while one or more completed recordings are still being processed.
- Replace the previous transcript with a dedicated live-recording surface while
  capturing, then with a full animated processing surface until the new final
  conversation is ready.
- Inspect all transcription and attribution stages in the in-app Activity drawer.
- Keep live captions in the central workspace while recording, recover missed
  native events by polling an in-memory snapshot, and log connection,
  first-text, no-text, and error states in Activity.
- Follow the latest live caption by default; scrolling upward pauses auto-follow
  and shows **Jump to latest** until the user resumes following.
- Sort people by when they were last heard and mark which profiles occur in the
  selected conversation; older profiles without a current ECAPA vector are
  labelled explicitly rather than appearing newly detected.
- Start or stop recording from the macOS menu-bar item.
- Attach or paste an agenda at any time after a conversation exists. PDF,
  DOC/DOCX, RTF, ODT, text/Markdown, HTML/XML, PowerPoint, and spreadsheet files
  below 50 MB are retained in their original form.
- Review unresolved participants before recapping, with an explicit
  **Recap anyway** override for meetings that cannot be fully attributed.
- Generate an English meeting title, executive summary, sectioned full summary,
  participant commitments, actions already reported as taken, and
  point-by-point agenda coverage through one native OpenAI Responses API call.
- View generated material in the meeting's original/dominant language or in
  English. Every intervention receives one validated translation decision;
  applicable non-English interventions show a complete English rendering while
  English and configured exclusions are removed only after coverage validation.
- Let the editable meeting title use all header space before the action buttons,
  wrap without a hard line limit, and grow the header naturally. OpenAI receives
  a soft instruction to keep generated titles within two normal desktop lines.
- Keep summary/action evidence IDs as internal factuality metadata rather than
  displaying them as links in generated views or clipboard exports.
- Copy the transcript or generated material as plain text or Markdown.
- Mark a recap stale after transcript, speaker, agenda, or translation-policy
  changes; hide stale inline translations and preserve the last good recap
  until a replacement validates and saves successfully.

Recall never generates a recap automatically.

## First-time setup on macOS

You need:

1. A Soniox API key.
2. Optionally, an OpenAI API key for recaps.
3. Rust installed with [rustup](https://rustup.rs/).
4. A project-local Node/npm toolchain, preferably through fnm, nvm, mise, or Volta.

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
2. Paste a Soniox API key and choose **Save key**. Recall writes it to its local
   app-data key file with user-only permissions, clears the input, and reuses it
   on later launches without a Keychain prompt.
3. Choose an audio input.
4. Adjust likely languages if needed. The defaults are en, fr, de, es, ru; these are hints, not a restriction on code-switching.
5. Leave live captions enabled unless you only want final transcription.
6. To use recaps, save an OpenAI key, choose the model, and optionally list
   source-language codes that should not receive inline English translations.

Configured Soniox/OpenAI status badges stay hidden. A missing key is shown as
an actionable warning instead of consuming permanent topbar space.

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
cargo clippy --offline -- -D warnings
~~~

Probe the exact Rust realtime stack with a mono signed 16-bit PCM WAV. This
reads the configured local key file and never prints it:

~~~sh
cd src-tauri
cargo run --offline --example probe_realtime -- /path/to/sample.wav
~~~

The Tauri main-window capability in `src-tauri/capabilities/main.json` grants the
bundled view permission to subscribe to native recording and transcription
events. Changes to native capabilities require stopping and restarting
`npm run dev`; reloading only the view is not sufficient.

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

The vendored model is the official WeSpeaker ECAPA-TDNN-512 large-margin ONNX
model. To refetch it, verify the pinned upstream SHA-256, and apply the metadata
patch:

~~~sh
uv run --with-requirements scripts/requirements-model-packaging.txt \
  python scripts/fetch_embedding_model.py
~~~

Model provenance and both checksums are recorded in models/README.md.

## Local data and privacy

- App data: ~/Library/Application Support/com.example.recall/
- Main database: recall.db
- One-time pre-migration backup: recall.pre-standalone-v1.db
- Pre-reference-policy backup: recall.pre-voice-reference-v1.db
- Soniox credential: `soniox-api-key` in the app-data directory, plaintext with
  macOS user-only `0600` permissions; never returned to the desktop view or
  stored in SQLite
- OpenAI credential: `openai-api-key` in the same directory, with the same
  local-only `0600` contract; never returned to JavaScript or stored in SQLite
- Pre-recap migration backup: recall.pre-recap-v1.db
- Agenda originals and structured recap payloads: stored with the conversation
  in encrypted-capable SQLite fields
- Temporary recordings: the macOS temporary directory; deleted after processing or failure
- Soniox uploads/transcriptions: deleted after final transcript retrieval on a best-effort basis
- New-speaker preview: retained locally only while the profile remains an unnamed VOICE<n>; deleted on naming or assignment to a named profile

Transcript fields and embedding vectors have application-level AES-GCM support in the schema, but safe migration to encrypted-at-rest storage is not implemented. Existing databases therefore remain unencrypted unless they were already configured with the older password mode. Recall refuses the old destructive “enable encryption” path instead of recreating the database.

OpenAI privacy boundary: no meeting content is sent during recording, Soniox
processing, voice matching, agenda attachment, or ordinary browsing. Choosing
**Recap** sends the final speaker-attributed transcript and optional agenda in a
single Responses API request with `store: false`. This request-level setting is
not itself a Zero Data Retention guarantee; the OpenAI account's applicable
data controls and policy still govern provider handling.

To repeat first-run testing without deleting data, quit Recall and move the app-data directory aside:

~~~sh
mv "$HOME/Library/Application Support/com.example.recall" \
  "$HOME/Library/Application Support/com.example.recall.saved"
~~~

Use **Settings → Remove** before moving the directory, or remove
`soniox-api-key` from the moved directory. A legacy Keychain entry from older
builds is ignored and may be deleted separately in Keychain Access. Move the
saved directory back when finished. Choose a different backup name if `.saved`
already exists.

## Voice identification design

Soniox diarization answers “who spoke when in this recording” but does not provide a persistent cross-meeting identity. Recall handles persistent identity locally:

1. Final Soniox tokens are grouped into contiguous speaker interventions.
2. Up to about 12 seconds of audio is gathered for each diarized speaker.
3. sherpa-onnx computes a 192-dimensional official WeSpeaker ECAPA-TDNN-512
   embedding with the correct feature-extraction frontend.
4. Recall compares it only with reference voiceprints produced by this
   versioned pipeline and belonging to people the user has named.
5. At least three seconds of usable speech is required; shorter or failed
   samples remain unattributed and do not create a VOICE profile.
6. A match requires cosine similarity of at least 0.90 and a margin of at least
   0.06 over the runner-up named person.
7. A named person can claim at most one diarized voice in a recording. If two
   voices compete for that person within 0.05, neither claim is trusted.
8. Automatic matches do not add their vectors back to the person's references.
   Only naming or explicitly assigning a provisional profile expands the
   reference library.

These rules deliberately optimize for precision: duplicate `VOICE<n>` profiles
are preferable to silently attributing words to the wrong person. The user can
preview and name a voice or assign a duplicate while choosing whether to
preserve or replace the prior reference pattern.

The current pipeline version is `wespeaker-ecapa512-lm-v2`. Embeddings from the
replaced v1 model are preserved but ignored. The database now distinguishes
reference vectors from unconfirmed observations. During the additive migration,
the oldest vector for each existing profile/model is retained as its reference;
later accumulated vectors are preserved but quarantined from matching. This
repairs the earlier self-training contamination without deleting history. A
current database copy is created as `recall.pre-voice-reference-v1.db` before
that migration and is never overwritten.

## Known limitations

- The user confirmed visible microphone-to-native live captions after the
  Rustls and event-plus-snapshot fixes. The new manual-scroll/follow control,
  Voice Library modal, history filter, stacked intervention layout, and
  conversation cleanup have automated coverage and still need one native visual
  smoke test.
- Live speaker labels are provisional and may shift; final async diarization is authoritative.
- The stricter named-only/unique-claim voice policy has unit and migration
  coverage but still needs a real one-person repeat test and a multi-person
  false-accept test.
- Native macOS system-audio capture is pending; virtual/aggregate input is the current route.
- Local database encryption migration is pending.
- The legacy bundle identifier com.example.recall is retained to avoid silently moving existing user data.
- App signing, notarization, auto-update, and release packaging are not configured.
- The native recap client, schema validation, persistence, stale-result guard,
  and interface contracts have automated coverage. The configured model has
  also returned valid structured results for the current multilingual meeting
  and bounded schema probes. The v2 schema now constrains every evidence and
  translation reference to the exact segment IDs in the request, and a live
  two-segment probe passed that contract. Full native save/tab acceptance plus
  PDF/DOCX agenda testing still need confirmation; model output quality and cost
  have not yet been calibrated.

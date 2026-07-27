# Recall

**A local-first, open-source meeting recorder for macOS.**

Recall records meetings, shows live captions, creates a speaker-separated final transcript, and uses a local voice database to recognize people across meetings. OpenAI can prepare a recap when requested.

Recall is an open-source alternative to subscription meeting tools. The app has no subscription fee or hosted account. Users supply their own API keys and pay each provider according to use. Conversations, participant names, agendas, recaps, and voice profiles are stored in a local SQLite archive.

## Supported providers

The maintained Recall app supports two cloud providers:

- [Soniox](https://soniox.com/) for live and final speech-to-text, language identification, and within-meeting diarization.
- [OpenAI](https://platform.openai.com/) for optional recaps requested by the user.

Support for another provider can be added in a fork. Provider-specific transport and response handling live in [`src-tauri/src/soniox.rs`](src-tauri/src/soniox.rs) and [`src-tauri/src/openai.rs`](src-tauri/src/openai.rs). The local database, recording flow, voice library, and desktop interface can remain unchanged. A replacement needs to map its events and output into Recall's transcript or recap contracts and update the related tests.

Recall-owned source is available under your choice of the [MIT License](LICENSE-MIT) or [Apache License 2.0](LICENSE-APACHE). Third-party dependencies and the bundled voice model retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Data and provider charges

The active app is standalone:

- It connects directly to Soniox with the user's API key.
- The key is stored in a local user-only file with `0600` permissions so it can
  be reused without recurring macOS Keychain password prompts.
- The desktop app runs without FastAPI, Uvicorn, Azure, Blob Storage, or a localhost service.
- Conversations, transcript interventions, speaker names, and voiceprints stay in local SQLite.
- Raw audio moves into private recovery storage before final transcription. It
  is deleted after the final conversation commits; failed or interrupted jobs
  retain it for an explicit retry.
- OpenAI is optional. Recall sends transcript and agenda content to it when the user chooses **Recap**.

Recall stores its archive locally. During transcription, it sends audio directly to Soniox. When the user chooses **Recap**, it sends the finalized meeting material directly to OpenAI. Recall has no hosted service between the app and these providers. Recall adds no fee; provider charges depend on usage and current pricing.

## Current capabilities

- Capture a selectable macOS audio input as mono 16-bit WAV.
- Use the same transparent frontal-brain-with-headset identity mark in the
  macOS Dock and the in-app sidebar; the sidebar mark spans the combined height
  of the product name and subtitle.
- Stream provisional original-language captions through Soniox stt-rt-v5 and,
  when useful, show a separate live translation in the user's preferred
  language. Speech already in the preferred language or an explicitly excluded
  language is not translated on screen.
- Keep original live captions working when a previously saved target is no
  longer supported by the active STT adapter. Settings preserves the unavailable
  value for correction and Activity reports the omitted translation; Recall
  does not silently choose another target.
- Run higher-quality final transcription through Soniox stt-async-v5.
- Enable speaker diarization, language identification, and code-switching-friendly language hints.
- Merge adjacent interventions from the same diarized speaker.
- Store conversations locally and edit their title, speaker assignment, and transcript text.
- Put intervention time and a wide person button on one metadata line above
  the intervention text, so long human names remain readable without narrowing
  the transcript. The button uses the available row width and shortens its label
  only when the full name genuinely does not fit. One shared searchable picker
  handles attribution instead of duplicating the whole voice library in every
  row. Interventions render as text, create an expanding editor only when
  requested, and load in 100-intervention batches for very long meetings.
- Keep the conversation list metadata-only. Opening a conversation loads one
  native payload with that meeting, its interventions, and recap state; a
  bounded five-conversation cache makes recent revisits immediate and is
  invalidated by relevant mutations. Transcript-text search remains available
  through a debounced native search.
- Build 192-dimensional ECAPA voiceprints locally through sherpa-onnx.
- Match only named people, conservatively and at most once per recording.
  Strong, unambiguous evidence assigns automatically; plausible evidence is
  shown as a one-click likely-person suggestion; unmatched voices become
  VOICE1, VOICE2, and so on.
- Build references only from clean central windows in longer interventions;
  reject overlapping, silent, short, or acoustically inconsistent candidates,
  and combine obvious split provider labels only at very high agreement.
- Keep Unicode-normalized human names unique. Legacy duplicate-name profiles
  are shown for merge or rename and excluded from automatic matching.
- Preview the temporary excerpt for a new voice, give it a name, or assign it to an existing named person.
- Show provisional profiles as `Not auto-matched`; **Name person** enables
  recognition, while **Rename person** changes a human-readable name later.
- Keep both voice patterns or replace an old pattern when assigning a changed voice.
- Show only voices attributed in the selected conversation in the right pane.
  Keep the complete identity database in the sidebar's paginated
  **People & Voices** manager, with separate searchable Profiles and exact
  conversation-scoped Unassigned views. Long person names wrap inside their
  cards instead of widening the Voices pane.
- Preview the full impact before merging profiles or assigning unassigned
  groups. Revalidate the affected conversations, create an integrity-checked
  database backup, and apply transcript, recap, voiceprint, and sample changes
  atomically.
- Filter historical conversations by a named person. Duplicate profiles with
  the same display name appear once, alphabetically, and the filter includes
  conversations attached to any of those profile IDs. Provisional `VOICE<n>`
  and Unknown labels stay out of this people-oriented filter.
- Delete profiles and conversations. Conversation deletion transactionally
  removes orphan unnamed `VOICE<n>` profiles, samples, and voiceprints, while
  preserving named people and provisional voices referenced by another
  conversation.
- Keep profile deletion in **People & Voices**. Named people referenced by
  conversation history are marked `History protected` and cannot be deleted;
  provisional-profile deletion explicitly warns that its attributed history
  will become Unknown speaker.
- Confirm destructive actions in a visible app-owned overlay, including when
  the People & Voices modal is already open.
- Record again while one or more completed recordings are still being processed.
- Navigate and use unrelated features while a final transcript or recap runs.
  Status stays with the affected conversation; only conflicting changes to
  that conversation are disabled, and different conversations may be recapped
  concurrently.
- Add a synthetic **Current recording** row while capture is active. It opens
  the live-caption surface, but the user can navigate to any historical
  conversation and work there without stopping the recording. Stopping or
  finishing background processing does not pull focus away from the view the
  user selected. A durable draft conversation is created from the live captions
  before final STT starts; active and failed jobs remain in the list with
  persistent status, exact errors, and retry controls.
- Keep every stopped WAV in a private app-owned recovery directory until the
  final transcript transaction commits. Interrupted jobs become retryable after
  restart, and an orphan WAV from a crash between file persistence and database
  insertion is surfaced as a recovered conversation.
- Inspect all transcription and attribution stages in the in-app Activity drawer.
- Keep live captions at the full remaining viewport height while recording,
  use the sidebar **Stop recording** control as the only in-window stop action,
  recover missed
  native events by polling an in-memory snapshot, and log connection,
  first-text, no-text, and error states in Activity.
- Follow the latest live caption by default; scrolling upward pauses auto-follow
  and shows **Jump to latest** until the user resumes following.
- Sort people by when they were last heard and mark which profiles occur in the
  selected conversation; older profiles without a current ECAPA vector are
  labelled as historical profiles.
- Start or stop recording from the macOS menu-bar item.
- Attach or paste an agenda at any time after a conversation exists. PDF,
  DOC/DOCX, RTF, ODT, text/Markdown, HTML/XML, PowerPoint, and spreadsheet files
  below 50 MB are retained in their original form.
- Review unresolved participants before recapping, with an explicit
  **Recap anyway** override for meetings that cannot be fully attributed.
- Generate a meeting title, executive summary, sectioned full summary,
  participant commitments, actions already reported as taken, and
  point-by-point agenda coverage through an explicit native OpenAI Responses
  API recap run in the user's preferred language. The holistic meeting analysis
  is separate from bounded per-intervention translation batches, so long
  meetings do not require one ever-growing structured response.
- View generated material in the meeting's original/dominant language or in the
  preferred language recorded with that recap. Every intervention receives one
  validated translation decision; applicable interventions show a complete
  preferred-language rendering while preferred-language and configured
  exclusions are removed only after coverage validation.
- Let the editable meeting title use all header space before the action buttons,
  wrap without a hard line limit, and grow the header naturally. OpenAI receives
  a soft instruction to keep generated titles within two normal desktop lines.
- Keep summary/action evidence IDs as internal factuality metadata. Generated
  views and clipboard exports omit them.
- Copy the transcript or generated material as plain text or Markdown.
- Mark a recap stale after transcript, speaker, or agenda changes; hide stale
  inline translations and preserve the last good recap until a replacement
  validates and saves successfully. A later settings change does not relabel or
  invalidate a recap already saved with its target language.
- Read existing English-specific recap payloads as English-target recaps. After
  validating the old content fingerprint, Recall upgrades only their fingerprint
  metadata; it does not rewrite their stored generated payload.

Recall never generates a recap automatically.

## First launch on macOS

Recall opens a getting-started guide the first time it launches. The guide explains the two provider roles, links only to official account pages, and can be reopened later from **Settings → Getting started**.

Soniox provides live captions and the final speaker-separated transcript. OpenAI is optional. Recall contacts OpenAI for summaries and recaps when the user chooses **Recap**.

To configure transcription:

1. Create an account or sign in at the [Soniox Console](https://console.soniox.com/).
2. Open **My First Project → API Keys** and generate a key.
3. Add prepaid balance or automatic top-up if your Soniox account requires it.
4. In Recall Settings, paste the key and choose **Save key**.

To enable optional recaps:

1. Sign in to the [OpenAI API Platform](https://platform.openai.com/). ChatGPT subscriptions and API billing are separate.
2. Configure [API billing](https://platform.openai.com/settings/organization/billing/overview).
3. Create a secret on the [API Keys page](https://platform.openai.com/api-keys).
4. Paste it into Recall Settings and choose a model available to your API project.

Recall stores both keys in local macOS user-only files. It does not put them in SQLite or return them to the desktop JavaScript view.

## Run from source on macOS

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

After launching from source:

1. Open **Settings**.
2. Paste a Soniox API key and choose **Save key**. Recall writes it to its local
   app-data key file with user-only permissions, clears the input, and reuses it
   on later launches without a Keychain prompt.
3. Choose an audio input.
4. Adjust likely languages if needed. The defaults are en, fr, de, es, ru; these are hints, not a restriction on code-switching.
5. Choose whether to enable live captions. They may increase STT charges because
   final transcription still runs after recording.
6. Choose your preferred language. Recall uses it for live translation and for
   translations in future on-demand recaps. It is automatically excluded from
   the no-translation list.
7. To use recaps, save an OpenAI key, choose the model, and optionally list
   other source-language codes that should not receive translations.

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
npm run test:e2e
npm run lint
npm run audit:licenses
npm run audit:secrets
npm run dev
~~~

The end-to-end suite serves the bundled desktop assets with a mocked Tauri
bridge and uses the installed Google Chrome. It verifies navigation and UI
state without launching Recall, opening the microphone, reading the local
archive, or requiring provider keys.

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

Build and verify an optimized Apple-silicon DMG for local/internal testing:

~~~sh
npm run package:mac:local
~~~

This recipe uses ad-hoc signing and is not a distributable public release.
Developer ID signing and Apple notarization are required before ordinary users
can open a downloaded build without a Gatekeeper warning. See
[PACKAGING.md](PACKAGING.md) for artifact paths, verification, release
credentials, architecture choices, and the external-release checklist.

The latest explicitly unsigned Apple-silicon preview is
[Recall v0.2.2](https://github.com/mvartanyan/recall-app/releases/tag/v0.2.2).

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
- Pre-processing-job migration backup: recall.pre-processing-v1.db
- SQLite archive and migration-backup permissions: user-only `0600`, enforced
  whenever Recall opens the archive
- Agenda originals and structured recap payloads: stored with the conversation
  in encrypted-capable SQLite fields
- Active/failed processing recordings: the app-data `processing/` directory,
  with directory mode `0700` and WAV mode `0600`; deleted after successful
  final commit, successful retry, or explicit conversation deletion
- Soniox uploads/transcriptions: deleted after final transcript retrieval on a best-effort basis
- New-speaker preview: retained locally only while the profile remains an unnamed VOICE<n>; deleted on naming or assignment to a named profile

Transcript fields and embedding vectors have application-level AES-GCM support in the schema, but safe migration to encrypted-at-rest storage is not implemented. Existing databases therefore remain unencrypted unless they were already configured with the older password mode. Recall refuses the old destructive “enable encryption” path instead of recreating the database.

OpenAI privacy boundary: no meeting content is sent during recording, Soniox
processing, voice matching, agenda attachment, or ordinary browsing. Choosing
**Recap** sends the final speaker-attributed transcript and optional agenda in
an on-demand Responses API run with `store: false`. The meeting analysis uses
one request; translations use fixed-size batches and may therefore use several
additional requests for a long transcript. Nothing is sent automatically. The
request-level `store: false` setting is not itself a Zero Data Retention
guarantee; the OpenAI account's applicable data controls and policy still
govern provider handling.

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

Soniox assigns speaker labels within a recording. Recall handles persistent identity across meetings locally:

1. Final Soniox tokens are grouped into contiguous speaker interventions.
2. Recall examines the longest interventions first, rejects intervals that
   overlap another provider speaker, trims turn boundaries, and extracts
   centered four-second windows.
3. sherpa-onnx computes a 192-dimensional official WeSpeaker ECAPA-TDNN-512
   embedding with the correct feature-extraction frontend.
4. Candidate windows must contain audible speech and form a consistent
   acoustic majority. Recall retains at most about 12 seconds from that
   majority. Mixed or tied candidates are excluded from matching, while the
   provider speaker remains available as a reviewable VOICE profile without a
   trusted reference.
5. Recall compares the result only with reference voiceprints produced by this
   versioned pipeline and belonging to people the user has named.
6. At least three seconds of usable speech is required for a trusted
   voiceprint. Shorter or failed samples produce a manual-review VOICE profile
   rather than a matchable reference.
7. Below 0.94 creates a new provisional voice. At or above 0.94 creates a
   likely-person suggestion unless automatic evidence is stronger.
8. Automatic assignment requires either one score of at least 0.97 with a lead
   of at least 0.03 over every different identity, or two references for the
   same person scoring at least 0.94 while every different identity remains
   below 0.94.
9. A named person can claim at most one diarized voice in a recording. When
   several voices claim that person, only a claim leading the next by at least
   0.06 can remain automatic.
10. Duplicate normalized human names are excluded from automatic matching until
   the user merges or renames them.
11. Separate provider labels are treated as one voice only when every clean
   label voiceprint agrees with every other at 0.97 or better.
12. Automatic matches do not add their vectors back to the person's references.
   Only naming or explicitly assigning a provisional profile expands the
   reference library.

These rules favor precision. A duplicate `VOICE<n>` profile is safer than an
incorrect automatic identity. A likely-person card shows its score, runner-up,
reference support, and a one-click assignment. Accepting it activates the
incoming voiceprint only if it agrees with an existing reference at 0.94 or
better; incompatible observations stay quarantined. The user can preview and
name a voice or assign a duplicate while choosing whether to preserve
compatible prior patterns or replace them. A no-voiceprint profile labels the
current transcript but does not participate in later automatic recognition
until a clean profile is assigned to it. Existing generic Unknown turns can be
grouped into one provisional profile or assigned intervention by intervention.

The current pipeline version is
`wespeaker-ecapa512-lm-v3-clean-window`. Embeddings from the replaced v1 and v2
pipelines are preserved but ignored. This intentionally quarantines references
created under the old sampling and 0.90 matching policy; measured current-data
cross-person similarity reached 0.904. Existing names and history remain, but a
clean v3 provisional occurrence must be assigned once to establish a reviewed
current reference. The database distinguishes reference vectors from
unconfirmed observations. During the earlier additive migration, the oldest
vector for each existing profile/model was retained as its reference; later
accumulated vectors were preserved but quarantined from matching. A current
database copy was created as `recall.pre-voice-reference-v1.db` before that
migration and is never overwritten.

The current matcher stores its decision evidence in `voice_match_decisions`.
Before that schema is added or upgraded in an existing archive, Recall creates
`recall.pre-voice-match-v1.db` and never overwrites it.

## Known limitations

- The user confirmed visible microphone-to-native live captions after the
  Rustls and event-plus-snapshot fixes. The manual-scroll/follow control, Voice
  Library modal, named-person history filter, selectable **Current recording**
  view, preferred-language settings, translated-caption stream, stacked
  intervention layout, and conversation cleanup have automated coverage and
  still need one native visual smoke test.
- Live speaker labels are provisional and may shift; final async diarization is authoritative.
- The v3 clean-window, identity-level consensus, likely-person review, and
  unique-claim voice policy has model, unit, migration, and mocked UI coverage
  but still needs a real repeated one-person test and a labelled multi-person
  false-accept test. Provider diarization has no exposed sensitivity or
  expected-speaker-count setting. Recall can repair only very high-confidence
  split labels; it cannot recover a provider intervention that already mixes
  or mislabels speakers.
- Native macOS system-audio capture is pending; virtual/aggregate input is the current route.
- Local database encryption migration is pending.
- The legacy bundle identifier com.example.recall is retained to avoid silently moving existing user data.
- A repeatable ad-hoc Apple-silicon app/DMG build, microphone entitlement, model
  attribution, and package verifier are configured. Developer ID signing,
  notarization, Intel/universal builds, and auto-update are not configured.
- The native recap client, schema validation, persistence, stale-result guard,
  and interface contracts have automated coverage. The configured model has
  also returned valid structured results for the current multilingual meeting
  and bounded schema probes. The v2 schema now constrains every evidence and
  translation reference to the exact segment IDs in the request, and a live
  two-segment probe passed that contract. Full native save/tab acceptance plus
  PDF/DOCX agenda testing still need confirmation; model output quality and cost
  have not yet been calibrated.

## License and publication

Recall-owned source is dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. This lets forks use, modify, redistribute, and commercialize the code under widely used permissive terms. Contributions are accepted under the same dual license unless explicitly agreed otherwise.

Dependencies are not relicensed by Recall. Their declared licenses are checked by `npm run audit:licenses`; the current macOS dependency graph has no GPL, AGPL, LGPL, SSPL, Business Source License, or non-commercial dependency. Several transitive Rust crates use MPL-2.0 file-level terms, which remain their own terms. The included WeSpeaker ECAPA model is CC BY 4.0 and requires attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`models/README.md`](models/README.md).

Before a public binary release, regenerate and review complete third-party notices, complete Apple Developer ID signing/notarization, and run the acceptance checklist in [PACKAGING.md](PACKAGING.md). The license audit is an engineering safeguard, not legal advice.

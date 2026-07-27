import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const html = fs.readFileSync(new URL("../src/index.html", import.meta.url), "utf8");
const javascript = fs.readFileSync(new URL("../src/main.js", import.meta.url), "utf8");
const rustMain = fs.readFileSync(new URL("../src-tauri/src/main.rs", import.meta.url), "utf8");
const rustDb = fs.readFileSync(new URL("../src-tauri/src/db.rs", import.meta.url), "utf8");
const rustSoniox = fs.readFileSync(new URL("../src-tauri/src/soniox.rs", import.meta.url), "utf8");
const rustOpenAI = fs.readFileSync(new URL("../src-tauri/src/openai.rs", import.meta.url), "utf8");
const rustRecap = fs.readFileSync(new URL("../src-tauri/src/recap.rs", import.meta.url), "utf8");
const rustJamieImport = fs.readFileSync(
  new URL("../src-tauri/src/jamie_import.rs", import.meta.url),
  "utf8",
);
const rustState = fs.readFileSync(new URL("../src-tauri/src/state.rs", import.meta.url), "utf8");
const rustConfig = fs.readFileSync(new URL("../src-tauri/src/config.rs", import.meta.url), "utf8");
const rustCredentials = fs.readFileSync(
  new URL("../src-tauri/src/credentials.rs", import.meta.url),
  "utf8",
);
const cargoToml = fs.readFileSync(new URL("../src-tauri/Cargo.toml", import.meta.url), "utf8");
const tauriConfig = fs.readFileSync(
  new URL("../src-tauri/tauri.conf.json", import.meta.url),
  "utf8",
);
const tauriSettings = JSON.parse(tauriConfig);
const macEntitlements = fs.readFileSync(
  new URL("../src-tauri/Entitlements.plist", import.meta.url),
  "utf8",
);
const thirdPartyNotices = fs.readFileSync(
  new URL("../THIRD_PARTY_NOTICES.md", import.meta.url),
  "utf8",
);
const licenseMit = fs.readFileSync(new URL("../LICENSE-MIT", import.meta.url), "utf8");
const licenseApache = fs.readFileSync(
  new URL("../LICENSE-APACHE", import.meta.url),
  "utf8",
);
const packageJson = fs.readFileSync(new URL("../package.json", import.meta.url), "utf8");
const appReadme = fs.readFileSync(new URL("../README.md", import.meta.url), "utf8");
const packageMacScript = fs.readFileSync(
  new URL("../scripts/package_macos_local.sh", import.meta.url),
  "utf8",
);
const stylesheet = fs.readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
const mainCapability = JSON.parse(
  fs.readFileSync(
    new URL("../src-tauri/capabilities/main.json", import.meta.url),
    "utf8",
  ),
);

function matches(source, expression) {
  return Array.from(source.matchAll(expression), (match) => match[1]);
}

test("every element requested by the UI controller exists exactly once", () => {
  const htmlIds = matches(html, /\bid="([^"]+)"/g);
  const requestedIds = matches(javascript, /getElementById\("([^"]+)"\)/g);
  assert.equal(new Set(htmlIds).size, htmlIds.length, "HTML contains duplicate element IDs");
  const missing = requestedIds.filter((id) => !htmlIds.includes(id));
  assert.deepEqual(missing, []);
});

test("every native command invoked by the UI is registered by Tauri", () => {
  const invoked = new Set(matches(javascript, /invoke\("([^"]+)"/g));
  const handlerBlock = rustMain.match(/generate_handler!\[([\s\S]*?)\]\)/);
  assert(handlerBlock, "Tauri command registration block not found");
  const registered = new Set(
    handlerBlock[1]
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean),
  );
  assert.deepEqual(
    Array.from(invoked).filter((command) => !registered.has(command)),
    [],
  );
});

test("every native event listened for by the UI is emitted by the app", () => {
  const listened = new Set(matches(javascript, /listen\("([^"]+)"/g));
  const emitted = new Set([
    ...matches(rustMain, /\.emit\(\s*"([^"]+)"/g),
    ...matches(rustSoniox, /\.emit\(\s*"([^"]+)"/g),
  ]);
  assert.deepEqual(
    Array.from(listened).filter((event) => !emitted.has(event)),
    [],
  );
});

test("the main window may subscribe to native events", () => {
  assert(mainCapability.windows.includes("main"));
  assert(mainCapability.permissions.includes("core:event:default"));
});

test("the desktop type scale remains readable", () => {
  const pixelSizes = matches(stylesheet, /font-size:\s*(\d+)px/g).map(Number);
  assert(pixelSizes.length > 0);
  assert(Math.min(...pixelSizes) >= 12, "UI contains text smaller than 12px");
});

test("the Recall mark is used by both the desktop view and native bundle", () => {
  assert.match(html, /<img class="brand-mark" src="recall-icon\.png"/);
  assert.match(stylesheet, /\.brand-mark[\s\S]*?width:\s*52px[\s\S]*?height:\s*52px/);
  assert.match(tauriConfig, /"icons\/icon\.icns"/);
  assert.match(tauriConfig, /"icons\/icon\.png"/);
  assert.match(tauriConfig, /"infoPlist":\s*"Info\.plist"/);
  for (const asset of [
    "../src/recall-icon.png",
    "../src-tauri/icons/icon.png",
    "../src-tauri/icons/icon.icns",
  ]) {
    assert(fs.existsSync(new URL(asset, import.meta.url)), `Missing identity asset: ${asset}`);
  }
});

test("the macOS package declares its runtime, microphone, model, and notice contract", () => {
  assert.equal(tauriSettings.bundle.macOS.minimumSystemVersion, "11.0");
  assert.equal(tauriSettings.bundle.macOS.entitlements, "Entitlements.plist");
  assert.match(macEntitlements, /com\.apple\.security\.device\.audio-input/);
  assert.match(macEntitlements, /<true\s*\/>/);
  assert.equal(
    tauriSettings.bundle.resources["../THIRD_PARTY_NOTICES.md"],
    "THIRD_PARTY_NOTICES.md",
  );
  assert.equal(tauriSettings.bundle.resources["../LICENSE-MIT"], "LICENSE-MIT");
  assert.equal(
    tauriSettings.bundle.resources["../LICENSE-APACHE"],
    "LICENSE-APACHE",
  );
  assert.match(thirdPartyNotices, /WeSpeaker ECAPA-TDNN-512/);
  assert.match(thirdPartyNotices, /Creative Commons Attribution 4\.0/);
  assert.match(packageJson, /"package:mac:local"/);
  assert.match(packageJson, /"verify:mac:package"/);
  assert.match(packageMacScript, /npm run audit:licenses/);
  assert.match(packageMacScript, /npm run audit:secrets/);
});

test("the project publishes permissive source licenses without relicensing the model", () => {
  assert.equal(JSON.parse(packageJson).license, "MIT OR Apache-2.0");
  assert.match(cargoToml, /^license\s*=\s*"MIT OR Apache-2\.0"/m);
  assert.match(licenseMit, /Permission is hereby granted, free of charge/);
  assert.match(licenseApache, /Apache License[\s\S]*Version 2\.0/);
  assert.match(thirdPartyNotices, /model is made available under/);
  assert.match(thirdPartyNotices, /CC BY 4\.0/);
});

test("first launch explains provider setup and remains reopenable", () => {
  assert.match(html, /id="onboardingDialog"/);
  assert.match(html, /Meeting transcripts, stored on your Mac/);
  assert.match(html, /Provider support/);
  assert.match(html, /Soniox provides live captions and the final speaker-separated transcript/);
  assert.match(html, /OpenAI is optional/);
  assert.match(html, /Recall contacts OpenAI when you choose/);
  assert.match(html, /OpenAI API key \(optional\)/);
  assert.match(html, /OpenAI account-level data controls and retention terms still apply/);
  assert.match(html, /id="gettingStartedButton"/);
  assert.match(javascript, /shouldShowOnboarding/);
  assert.match(javascript, /scheduleInitialSetupPrompt/);
  assert.match(javascript, /invoke\("complete_onboarding"/);
  assert.match(rustConfig, /onboarding_version/);
  assert.match(javascript, /open_external_url/);
  assert.match(rustMain, /const ALLOWED_EXTERNAL_URLS/);
  assert.match(rustMain, /fn is_allowed_external_url/);
});

test("settings remain scrollable and explain the live-caption charge", () => {
  assert.match(html, /class="settings-scroll"/);
  assert.match(
    stylesheet,
    /\.settings-scroll\s*\{[\s\S]*?overflow-y:\s*auto;[\s\S]*?\}/,
  );
  assert.match(
    stylesheet,
    /\.settings-modal form\s*\{[\s\S]*?display:\s*flex;[\s\S]*?max-height:/,
  );
  assert.match(
    html,
    /This may increase STT charges/,
  );
  assert.match(
    stylesheet,
    /\.recap-settings-grid \.toggle-row\s*\{[\s\S]*?width:\s*100%;[\s\S]*?\}/,
  );
  assert.doesNotMatch(stylesheet, /\.recap-settings-grid \.toggle-row\s*\{[^}]*max-width:/);
});

test("runtime copy uses provider-neutral STT and LLM terms", () => {
  assert.match(javascript, /LLM recap/);
  assert.doesNotMatch(html, /OpenAI recap/);
  assert.match(javascript, /"stt:upload:start": "Uploading recording to the STT provider"/);
  assert.match(javascript, /llm: "Waiting for the LLM provider"/);
  assert.doesNotMatch(javascript, /live Soniox captions|Uploading the recording to Soniox/);
  assert.doesNotMatch(javascript, /Starting on-demand OpenAI recap|OpenAI recap saved locally/);
  assert.match(rustSoniox, /progress\("stt:upload:start"/);
  assert.doesNotMatch(rustSoniox, /progress\("soniox:/);
  assert.match(rustMain, /emit_recap_progress\(\s*app_handle,\s*session_id,\s*"llm:start"/);
  assert.doesNotMatch(rustSoniox, /Soniox/);
  assert.doesNotMatch(rustOpenAI, /OpenAI/);
  assert.doesNotMatch(rustRecap, /OpenAI/);
});

test("public copy avoids discarded slogans and staged contrasts", () => {
  const publicCopy = `${html}\n${appReadme}`;
  for (const expression of [
    /This is deliberate/i,
    /not a lock-in/i,
    /Opinionated by default/i,
    /forkable by design/i,
    /provider-picker maze/i,
    /least-common-denominator/i,
    /Own your meeting memory/i,
  ]) {
    assert.doesNotMatch(publicCopy, expression);
  }
});

test("live captions have event delivery plus a native polling fallback", () => {
  assert.match(javascript, /listen\("live-transcription"/);
  assert.match(javascript, /invoke\("get_live_transcription"\)/);
  assert.match(rustMain, /fn get_live_transcription/);
  assert.match(javascript, /Live captions are receiving speech/);
  assert.match(javascript, /Live captions finished without receiving speech/);
});

test("live captions carry separate original and preferred-language streams", () => {
  assert.match(html, /id="liveTranslatedTranscript"/);
  assert.match(html, /id="liveTranslationWarning"/);
  assert.match(rustSoniox, /"type": "one_way"/);
  assert.match(rustSoniox, /"target_language": target_language/);
  assert.match(rustSoniox, /translation_status\.as_deref\(\) == Some\("translation"\)/);
  assert.match(javascript, /payload\.translated_text/);
  assert.match(rustSoniox, /Original live captions will continue/);
  assert.match(javascript, /Live translation: /);
});

test("live captions can pause auto-follow and jump back to the latest text", () => {
  assert.match(html, /id="jumpToLiveButton"/);
  assert.match(javascript, /isNearScrollBottom\(event\?\.currentTarget \|\| elements\.liveTranscript\)/);
  assert.match(javascript, /if \(state\.liveFollow\) scrollLiveToLatest\(\)/);
  assert.match(javascript, /jumpToLiveButton\.addEventListener\("click"/);
});

test("recording and processing replace rather than overlay the old transcript", () => {
  assert.match(javascript, /elements\.livePanel\.hidden = mode !== "recording"/);
  assert.match(javascript, /elements\.processingState\.hidden = mode !== "processing"/);
  assert.match(javascript, /elements\.transcriptContent\.hidden = mode !== "conversation"/);
});

test("final transcription is recovery-safe and retryable", () => {
  assert.match(rustDb, /CREATE TABLE IF NOT EXISTS processing_jobs/);
  assert.match(rustDb, /fn mark_interrupted_processing_jobs/);
  assert.match(rustDb, /status='failed'[\s\S]*?interrupted when Recall closed/);
  assert.match(rustMain, /fn persist_recording_audio/);
  assert.match(rustMain, /create_processing_session\(/);
  assert.match(rustMain, /"audio:retained"/);
  assert.match(rustMain, /fn retry_processing/);
  assert.match(javascript, /invoke\("retry_processing"/);
  assert.match(html, /id="processingRecoveryBanner"/);
  assert.match(html, /id="retryProcessingButton"/);
  assert.match(javascript, /The recording and live-caption draft are still saved locally/);
  assert.match(javascript, /Final transcript needs retry/);
});

test("long final uploads stream from disk and do not use the old 90-second limit", () => {
  assert.match(rustSoniox, /multipart::Part::file\(path\)/);
  assert.match(rustSoniox, /REST_REQUEST_TIMEOUT.*2 \* 60 \* 60/);
  assert.match(rustSoniox, /TRANSCRIPTION_DEADLINE.*2 \* 60 \* 60/);
  assert.doesNotMatch(rustSoniox, /std::fs::read\(path\)/);
  assert.doesNotMatch(rustSoniox, /timeout\(Duration::from_secs\(90\)\)/);
});

test("people distinguish selected, historical, and matchable profiles", () => {
  assert.match(javascript, /In selected conversation/);
  assert.match(javascript, /Last heard/);
  assert.match(javascript, /No current voiceprint/);
  assert.match(javascript, /Name person/);
  assert.match(javascript, /Rename person/);
  assert.match(javascript, /Not auto-matched/);
  assert.match(javascript, /Automatic recognition on/);
  assert.match(html, /Only named people are eligible for automatic recognition/);
});

test("current voices and the full Voice Library are separate surfaces", () => {
  assert.match(html, /id="voiceLibraryDialog"/);
  assert.match(html, /Only voices attributed in this conversation appear here/);
  assert.match(javascript, /const currentSpeakers = state\.speakers\.filter/);
  assert.match(javascript, /function renderVoiceLibrary\(\)/);
});

test("completed conversations are not masked by unrelated processing jobs", () => {
  assert.match(javascript, /isSessionProcessing\(selectedSession\)/);
  assert.match(javascript, /reconcileTrackedRuns\(state\.sessions\)/);
  const speakerRenderer = javascript.slice(
    javascript.indexOf("function renderSpeakers"),
    javascript.indexOf("function renderVoiceLibrary"),
  );
  assert.doesNotMatch(speakerRenderer, /state\.activeRuns\.size/);
});

test("unknown interventions remain reviewable without inventing a voiceprint", () => {
  assert.match(javascript, /function buildUnknownSpeakerCard/);
  assert.match(javascript, /Group as one voice/);
  assert.match(javascript, /invoke\("create_profile_for_unknown_segments"/);
  assert.match(rustMain, /fn create_profile_for_unknown_segments/);
  assert.match(rustDb, /create_speaker_for_unattributed_segments/);
  assert.match(rustMain, /no safe automatic match was available/);
  assert.match(rustMain, /no trusted voiceprint was created/);
});

test("conversation history can be filtered by a voice profile", () => {
  assert.match(html, /id="conversationSpeakerFilter"/);
  assert.match(javascript, /groupVoiceFilters\(state\.speakers\)/);
  assert.match(javascript, /invoke\("list_session_ids_for_speakers"/);
  assert.match(rustMain, /fn list_session_ids_for_speakers/);
});

test("the conversation voice filter excludes provisional and unknown profiles", () => {
  assert.match(javascript, /groupVoiceFilters\(state\.speakers\)/);
  assert.match(javascript, /selectedStillExists/);
  assert.match(javascript, /state\.voiceFilteredSessionIds = null/);
});

test("an active recording is a selectable sidebar workspace", () => {
  assert.match(javascript, /dataset\.currentRecording = "true"/);
  assert.match(javascript, /function selectCurrentRecording\(\)/);
  assert.match(javascript, /recordingViewSelected: state\.liveWorkspaceSelected/);
  assert.match(javascript, /state\.openQueuedDraftRevision === state\.navigationRevision/);
  assert.match(javascript, /Voice preview is unavailable during recording/);
});

test("interventions put time and one shared speaker picker above the text", () => {
  assert.match(javascript, /speakerColumn\.append\(time, speakerButton\)/);
  assert.match(stylesheet, /\.segment\s*\{\s*display:\s*block;/);
  assert.match(
    stylesheet,
    /\.segment-speaker-button[\s\S]*?flex:\s*1 1 auto;[\s\S]*?min-width:\s*0;[\s\S]*?max-width:\s*100%/,
  );
  assert.match(html, /id="speakerPickerDialog"/);
  assert.match(javascript, /function renderSpeakerPicker\(\)/);
});

test("long participant names use available transcript width and wrap inside voice cards", () => {
  assert.match(javascript, /copy\.className = "speaker-copy"/);
  assert.match(
    stylesheet,
    /\.speaker-card[\s\S]*?width:\s*100%;[\s\S]*?min-width:\s*0;[\s\S]*?max-width:\s*100%/,
  );
  assert.match(stylesheet, /\.speaker-copy\s*\{\s*min-width:\s*0;/);
  assert.match(
    stylesheet,
    /\.speaker-name[\s\S]*?line-height:\s*1\.2;[\s\S]*?overflow-wrap:\s*anywhere;/,
  );
});

test("interventions render progressively as text and create editors only on demand", () => {
  assert.match(javascript, /const SEGMENT_RENDER_BATCH = 100/);
  assert.match(javascript, /state\.selectedSegments\.slice\(0, state\.renderedSegmentCount\)/);
  assert.match(javascript, /function beginSegmentEdit\(/);
  assert.match(javascript, /document\.createElement\("textarea"\)/);
  assert.match(javascript, /function showMoreSegments\(\)/);
  assert.match(stylesheet, /\.segment-text-display[\s\S]*?margin-left:\s*12px/);
});

test("conversation navigation loads one scoped payload while the archive stays metadata-only", () => {
  assert.match(rustMain, /fn list_sessions[\s\S]*?Result<Vec<SessionSummary>, String>/);
  assert.match(rustMain, /fn load_conversation[\s\S]*?Result<ConversationPayload, String>/);
  assert.match(rustMain, /recap_state_view_from\(app_state\.inner\(\), &db, &session, &segments\)/);
  assert.match(javascript, /invoke\("load_conversation", \{ sessionId \}\)/);
  assert.match(javascript, /const CONVERSATION_CACHE_LIMIT = 5/);
  assert.match(javascript, /invoke\("search_session_ids", \{ query \}\)/);
  assert.match(rustDb, /pub struct SessionSummary/);
  assert.match(rustDb, /CREATE INDEX IF NOT EXISTS sessions_created_at_idx/);
});

test("conversation deletion reports orphan unnamed voice cleanup", () => {
  assert.match(javascript, /orphan unnamed voice/);
  assert.match(rustMain, /fn delete_session[\s\S]*?Result<usize, String>/);
});

test("destructive actions use the in-app confirmation flow", () => {
  assert.match(html, /id="confirmationDialog"/);
  assert.match(javascript, /requestConfirmation\(\{[\s\S]*?Delete this conversation/);
  assert.match(javascript, /requestConfirmation\(\{[\s\S]*?Delete this voice profile/);
  assert.doesNotMatch(javascript, /window\.confirm/);
});

test("named people used by history cannot be deleted", () => {
  assert.match(html, /named people used by conversation history are protected/);
  assert.match(javascript, /protectedTag\.textContent = "History protected"/);
  assert.match(
    javascript,
    /if \(!provisional && speaker\.conversation_count > 0\) \{\s*deleteButton\.disabled = true/,
  );
  assert.match(rustDb, /Reassign or delete those conversations before deleting/);
});

test("People & Voices is paginated, conversation-scoped, previewed, and atomic", () => {
  for (const id of [
    "peopleVoicesButton",
    "voiceLibraryDialog",
    "identityProfilesTab",
    "identityUnassignedTab",
    "identitySearch",
    "identityStatusFilter",
    "identityPreviousPage",
    "identityNextPage",
    "identityMergeButton",
    "identityMergeDialog",
    "identityPreviewButton",
    "identityConfirmButton",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
  assert.match(javascript, /new Map\(\)[\s\S]*selectedUnassignedGroups/);
  assert.match(
    javascript,
    /JSON\.stringify\(\[\s*String\(key\?\.session_id[\s\S]*key\?\.speaker_label/,
  );
  assert.match(javascript, /"list_identity_profiles"/);
  assert.match(javascript, /"list_unassigned_identities"/);
  assert.match(javascript, /invoke\("preview_identity_consolidation"/);
  assert.match(javascript, /invoke\("consolidate_identities"/);
  const consolidationUi = javascript.slice(
    javascript.indexOf("async function confirmIdentityConsolidation"),
    javascript.indexOf("function actionButton"),
  );
  assert.doesNotMatch(consolidationUi, /loadSessions\(/);
  assert.match(consolidationUi, /identityOperationBadge\.hidden = false/);
  assert.match(rustMain, /tokio::task::spawn_blocking/);
  assert.match(rustMain, /claim_identity_sessions/);
  assert.match(rustDb, /verified_runtime_backup\("pre-identity-merge"\)/);
  assert.match(rustDb, /The affected conversations changed after the impact preview/);
  assert.match(rustDb, /rebuild_session_transcripts_in_transaction/);
  assert.match(rustDb, /segments_speaker_session_idx/);
  assert.match(rustDb, /embeddings_speaker_model_reference_idx/);
});

test("speaker matching requires trusted names and unique meeting claims", () => {
  assert.match(rustMain, /if is_provisional_label\(label\)/);
  assert.match(rustMain, /resolve_unique_profile_matches/);
  assert.match(rustMain, /reference left unchanged/);
  assert.match(rustMain, /if is_new \{\s*db\.insert_embedding/);
});

test("ambiguous voice matches remain reviewable and survive navigation", () => {
  assert.match(javascript, /speaker\.likely_match/);
  assert.match(javascript, /Likely " \+ likelyMatch\.label/);
  assert.match(javascript, /Assign to " \+ likelyMatch\.label/);
  assert.match(javascript, /Choose another person/);
  assert.match(javascript, /invoke\("accept_voice_match_suggestion"/);
  assert.match(rustDb, /CREATE TABLE IF NOT EXISTS voice_match_decisions/);
  assert.match(rustDb, /pub fn insert_voice_match_decision/);
  assert.match(rustDb, /pub fn accept_voice_match_suggestion/);
  assert.match(rustMain, /VoiceMatchKind::Suggested/);
});

test("duplicate person names are blocked and legacy conflicts are visible", () => {
  assert.match(javascript, /duplicate_name_conflict/);
  assert.match(javascript, /Duplicate name/);
  assert.match(javascript, /Automatic matching ignores all of them/);
  assert.match(javascript, /result\?\.status === "conflict"/);
  assert.match(rustDb, /normalized_person_name/);
  assert.match(rustDb, /status: "conflict"/);
  assert.match(rustMain, /conflicted_profiles/);
});

test("voice observations use centroids and suggestion reference hygiene", () => {
  assert.match(rustMain, /average_embeddings\(selected_vectors\.iter\(\)\.cloned\(\)\)/);
  assert.doesNotMatch(
    rustMain,
    /selected_windows == 1[\s\S]{0,500}embedder\.embed\(&pcm/,
  );
  assert.match(rustDb, /SUGGESTION_REFERENCE_COMPATIBILITY_THRESHOLD/);
  assert.match(rustDb, /is_reference=0 WHERE speaker_id/);
  assert.match(javascript, /incompatible voiceprint/);
});

test("voice sampling uses clean central windows and quarantines the previous pipeline", () => {
  assert.match(rustMain, /SAMPLE_EDGE_TRIM_MS/);
  assert.match(rustMain, /fn clean_sample_windows/);
  assert.match(rustMain, /overlaps_other_speaker/);
  assert.match(rustMain, /dominant_consistent_indices/);
  assert.match(rustMain, /SAME_VOICE_SPLIT_THRESHOLD/);
  assert.match(rustMain, /voiceprint:labels:coalesced/);
  assert.match(
    fs.readFileSync(new URL("../src-tauri/src/embedding.rs", import.meta.url), "utf8"),
    /wespeaker-ecapa512-lm-v3-clean-window/,
  );
});

test("compact modal fields remain inside their horizontal inset", () => {
  assert.match(
    stylesheet,
    /\.compact-modal form > input\[type="text"\][\s\S]*?width:\s*calc\(100% - 48px\)/,
  );
});

test("the Soniox key stays in a local user-only file", () => {
  assert.match(rustCredentials, /SONIOX_KEY_FILENAME.*soniox-api-key/);
  assert.match(rustCredentials, /from_mode\(0o600\)/);
  assert.doesNotMatch(cargoToml, /^keyring\s*=/m);
  assert.match(html, /file readable only by your macOS user account/);
  assert.match(javascript, /Recall will reuse it without a Keychain prompt/);
});

test("OpenAI recaps are explicit native Responses API calls with strict stateless output", () => {
  assert.match(rustOpenAI, /https:\/\/api\.openai\.com\/v1\/responses/);
  assert.match(rustOpenAI, /"store": false/);
  assert.match(rustOpenAI, /"truncation": "disabled"/);
  assert.match(rustOpenAI, /"tools": \[\]/);
  assert.match(rustOpenAI, /"type": "json_schema"/);
  assert.match(rustOpenAI, /"strict": true/);
  assert.match(javascript, /async function requestRecap\(\)/);
  assert.match(javascript, /invoke\("generate_recap"/);
  assert.match(rustOpenAI, /translation_chunks\(request\.segments\)/);
  assert.match(rustOpenAI, /TRANSLATION_CHUNK_MAX_CHARACTERS/);
  assert.match(html, /id="recapStatusBanner"/);
  assert.match(javascript, /recapJobs: new Map\(\)/);
  const initializeBlock = javascript.match(/async function initialize\(\) \{[\s\S]*?\n\}/);
  assert(initializeBlock);
  assert.doesNotMatch(initializeBlock[0], /generate_recap/);
});

test("OpenAI credentials are local user-only data and never returned to JavaScript", () => {
  assert.match(rustCredentials, /OPENAI_KEY_FILENAME.*openai-api-key/);
  assert.match(rustState, /openai_key_path/);
  assert.match(rustCredentials, /save_openai_api_key/);
  assert.match(rustCredentials, /from_mode\(0o600\)/);
  assert.doesNotMatch(javascript, /load_openai_key|OPENAI_API_KEY/);
  assert.match(html, /OpenAI account-level data controls and retention terms still apply/);
});

test("recaps and original agenda files use encrypted-capable local persistence", () => {
  assert.match(rustDb, /CREATE TABLE IF NOT EXISTS session_agendas/);
  assert.match(rustDb, /CREATE TABLE IF NOT EXISTS session_recaps/);
  assert.match(rustDb, /pre-recap-v1\.db/);
  assert.match(rustDb, /self\.crypto\.encrypt\(content\)/);
  assert.match(rustDb, /self\.crypto\.encrypt\(&payload_bytes\)/);
  assert.match(rustDb, /DELETE FROM session_recaps WHERE session_id/);
  assert.match(rustDb, /DELETE FROM session_agendas WHERE session_id/);
  assert.match(rustOpenAI, /"type": "input_file"/);
  assert.match(rustOpenAI, /"file_data"/);
});

test("recap content changes are fingerprinted while language preferences are regeneration-only", () => {
  assert.match(rustRecap, /struct FingerprintInput<'a>[\s\S]*?agenda: Option<.*?>,[\s\S]*?\n}/);
  assert.match(rustRecap, /legacy_source_fingerprint/);
  assert.match(rustMain, /update_recap_source_fingerprint/);
  assert.match(rustRecap, /content_sha256/);
  assert.match(javascript, /state\.recapState\.stale/);
  assert.match(javascript, /if \(!state\.recapState\?\.recap \|\| state\.recapState\.stale\) return \[\]/);
  assert.match(html, /This recap is out of date/);
});

test("recap UI exposes participant review, agenda, result tabs, translations, and copy formats", () => {
  for (const label of [
    "Executive summary",
    "Full summary",
    "Actions",
    "Agenda coverage",
    "Copy text",
    "Copy Markdown",
    "Recap anyway",
  ]) {
    assert.match(html, new RegExp(label));
  }
  assert.match(javascript, /buildTranslationPlan\(segment\.text, translations\)/);
  assert.match(javascript, /TRANSLATION:/);
  assert.match(javascript, /navigator\.clipboard\.writeText/);
  assert.match(javascript, /invoke\("choose_agenda_file"/);
  assert.match(javascript, /const persistedState = await invoke\("get_recap_state"/);
  assert.match(javascript, /Recap interface ready with/);
  assert.match(javascript, /state\.recapJobs\.set\(sessionId/);
  assert.match(html, /id="recapStatusDismiss"[^>]*hidden/);
  assert.doesNotMatch(html, /id="recapProgressDialog"/);
});

test("recap and final transcription status are scoped without taking over unrelated work", () => {
  assert.match(javascript, /sessionRecapJob\?\.status === "running"/);
  assert.match(javascript, /recapIsRunning\(session\.id\)/);
  assert.match(javascript, /await Promise\.all\(\[loadSpeakers\(\), loadSessions\(\)\]\)/);
  assert.match(javascript, /selectedBeforeRefresh === sessionId/);
  assert.doesNotMatch(javascript, /selectedBeforeRefresh === sessionId \|\| !selectedBeforeRefresh/);
  assert.match(rustState, /recap_in_flight: Arc<Mutex<HashSet<String>>>/);
  assert.match(rustMain, /ensure_sessions_not_recapping/);
  const runRecapBlock = javascript.slice(
    javascript.indexOf("async function runRecap"),
    javascript.indexOf("async function loadSettingsData"),
  );
  assert.doesNotMatch(runRecapBlock, /showModal\(/);
});

test("recording uses the sidebar stop control and live captions fill the remaining viewport", () => {
  assert.doesNotMatch(html, /id="stopButton"|Stop &amp; process|Stop and process/);
  assert.match(javascript, /recordButtonLabel\.textContent = recording \? "Stop recording"/);
  assert.match(stylesheet, /body\.recording-active \.content-grid[\s\S]*?flex:\s*1 1 0/);
  assert.match(stylesheet, /body\.recording-active \.live-transcript[\s\S]*?max-height:\s*none/);
});

test("summaries and actions keep evidence internal instead of displaying links", () => {
  assert.doesNotMatch(javascript, /section\.evidence_segment_ids/);
  const actionRenderer = javascript.slice(
    javascript.indexOf("function renderActionGroup"),
    javascript.indexOf("function renderTranscript"),
  );
  assert.doesNotMatch(actionRenderer, /appendGeneratedEvidence/);
  assert.doesNotMatch(javascript, /evidenceLabel\(item\.evidence_segment_ids\)/);
  assert.match(rustRecap, /validate_required_evidence/);
});

test("topbar actions stay on one line while the meeting title uses available width and wraps", () => {
  assert.match(html, /<textarea id="conversationTitle"[^>]*rows="1"/);
  assert.match(stylesheet, /\.topbar-copy[\s\S]*?flex:\s*1 1 auto/);
  assert.match(stylesheet, /\.conversation-title[\s\S]*?width:\s*100%/);
  assert.match(stylesheet, /\.conversation-title[\s\S]*?white-space:\s*pre-wrap/);
  assert.doesNotMatch(stylesheet, /\.conversation-title\s*\{[^}]*max-width/);
  assert.match(stylesheet, /\.topbar-actions[\s\S]*?flex:\s*0 0 auto/);
  assert.match(stylesheet, /\.compact-action[\s\S]*?white-space:\s*nowrap/);
  assert.match(javascript, /function scheduleConversationTitleResize\(\)/);
  assert.match(javascript, /conversationTitle\.scrollHeight/);
});

test("configured services stay quiet while missing keys remain visible", () => {
  assert.match(html, /id="serviceBadge"[^>]*hidden/);
  assert.match(html, /id="openaiServiceBadge"[^>]*hidden/);
  assert.match(html, /id="keyStatus"[^>]*hidden/);
  assert.match(html, /id="openaiKeyStatus"[^>]*hidden/);
  assert.match(javascript, /elements\.serviceBadge\.hidden = configured/);
  assert.match(javascript, /elements\.keyStatus\.hidden = configured/);
  assert.match(javascript, /elements\.openaiKeyStatus\.hidden = configured/);
  assert.match(javascript, /elements\.openaiServiceBadge\.hidden = configured/);
  assert.match(javascript, /Soniox key needed/);
  assert.match(javascript, /Key needed/);
});

test("recap preferences separate Soniox hints from translation exclusions", () => {
  assert.match(rustConfig, /openai_model/);
  assert.match(rustConfig, /preferred_language/);
  assert.match(rustConfig, /no_translation_languages/);
  assert.match(html, /id="preferredLanguage"/);
  assert.match(html, /id="languageHints"/);
  assert.match(html, /id="noTranslationLanguages"/);
  assert.match(javascript, /parseNoTranslationLanguages/);
  assert.match(javascript, /invoke\("list_translation_languages"\)/);
  assert.match(rustRecap, /target_language/);
  assert.match(rustRecap, /translated_text/);
  assert.match(rustOpenAI, /preferred_language/);
});

test("active desktop code has no localhost API or Azure dependency", () => {
  for (const source of [html, javascript, rustMain, rustSoniox, tauriConfig, packageJson]) {
    assert.doesNotMatch(source, /localhost:\d+|api base url|azure/i);
  }
  assert.doesNotMatch(packageJson, /http-server|npm-run-all/i);
});

test("Jamie archives use a reviewable, recoverable native import workflow", () => {
  for (const id of [
    "chooseJamieExportButton",
    "resumeJamieImportButton",
    "jamieImportDialog",
    "jamieImportError",
    "jamieImportErrorMessage",
    "jamieIdentityList",
    "jamieMeetingList",
    "jamieImportButton",
    "importedExecutiveTab",
    "importedFullSummaryTab",
    "importedTasksTab",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`));
  }
  assert.match(
    html,
    /id="jamieImportSettingsSection"[^>]*hidden[^>]*aria-hidden="true"/,
  );
  assert.match(
    html,
    /id="jamieImportDialog"[^>]*hidden[^>]*aria-hidden="true"/,
  );
  assert.match(
    javascript,
    /const JAMIE_IMPORT_UI_ENABLED = window\.__RECALL_ENABLE_JAMIE_IMPORT__ === true/,
  );
  assert.match(
    javascript,
    /JAMIE_IMPORT_UI_ENABLED \? invoke\("list_import_batches"\) : Promise\.resolve\(\[\]\)/,
  );
  assert.match(javascript, /openJamieImport\("choose_jamie_export"\)/);
  assert.match(javascript, /const sourcePath = await invoke\("choose_jamie_export"\)/);
  assert.match(javascript, /invoke\("inspect_jamie_export", \{ sourcePath \}\)/);
  assert.match(javascript, /jamieIdentityValidationIssue\(identity, preview\)/);
  assert.match(html, /Needs attention only/);
  assert.match(
    stylesheet,
    /\.jamie-import-modal\s*\{[^}]*height:\s*min\(900px,\s*calc\(100vh - 40px\)\)/s,
  );
  assert.match(javascript, /invoke\("save_jamie_import_draft"/);
  assert.match(javascript, /invoke\("run_jamie_import"/);
  assert.match(javascript, /invoke\("rollback_jamie_import"/);
  assert.match(javascript, /It was not generated by Recall/);
  assert.match(rustJamieImport, /BufReader/);
  assert.match(rustJamieImport, /JAMIE_IMPORTER_VERSION/);
  assert.match(rustJamieImport, /proposed_map/);
  assert.match(rustJamieImport, /alias == "Mv"/);
  assert.match(rustDb, /CREATE TABLE IF NOT EXISTS import_batches/);
  assert.match(rustDb, /CREATE TABLE IF NOT EXISTS imported_sessions/);
  assert.match(rustDb, /CREATE TABLE IF NOT EXISTS session_import_artifacts/);
  assert.match(rustDb, /verified_runtime_backup\("pre-jamie-import"\)/);
  assert.match(rustDb, /verified_runtime_backup\("pre-jamie-rollback"\)/);
  assert.match(rustDb, /PRAGMA integrity_check/);
});

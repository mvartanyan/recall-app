import {
  formatDuration,
  formatTimestamp,
  isProvisionalLabel,
  parseLanguageHints,
  transcriptFromSegments,
} from "./ui-helpers.mjs";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const elements = {
  recordButton: document.getElementById("recordButton"),
  recordButtonLabel: document.getElementById("recordButtonLabel"),
  emptyRecordButton: document.getElementById("emptyRecordButton"),
  stopButton: document.getElementById("stopButton"),
  recordingBanner: document.getElementById("recordingBanner"),
  recordingTimer: document.getElementById("recordingTimer"),
  levelBar: document.getElementById("levelBar"),
  livePanel: document.getElementById("livePanel"),
  liveStatus: document.getElementById("liveStatus"),
  liveTranscript: document.getElementById("liveTranscript"),
  sessionsList: document.getElementById("sessionsList"),
  conversationSearch: document.getElementById("conversationSearch"),
  refreshSessions: document.getElementById("refreshSessions"),
  conversationTitle: document.getElementById("conversationTitle"),
  conversationMeta: document.getElementById("conversationMeta"),
  deleteSessionButton: document.getElementById("deleteSessionButton"),
  serviceBadge: document.getElementById("serviceBadge"),
  emptyState: document.getElementById("emptyState"),
  processingState: document.getElementById("processingState"),
  processingTitle: document.getElementById("processingTitle"),
  processingDetail: document.getElementById("processingDetail"),
  transcriptContent: document.getElementById("transcriptContent"),
  segmentsList: document.getElementById("segmentsList"),
  legacyTranscript: document.getElementById("legacyTranscript"),
  saveState: document.getElementById("saveState"),
  speakersList: document.getElementById("speakersList"),
  refreshSpeakers: document.getElementById("refreshSpeakers"),
  settingsButton: document.getElementById("settingsButton"),
  settingsDialog: document.getElementById("settingsDialog"),
  settingsForm: document.getElementById("settingsForm"),
  sonioxKey: document.getElementById("sonioxKey"),
  keyStatus: document.getElementById("keyStatus"),
  saveKeyButton: document.getElementById("saveKeyButton"),
  deleteKeyButton: document.getElementById("deleteKeyButton"),
  inputDevice: document.getElementById("inputDevice"),
  languageHints: document.getElementById("languageHints"),
  liveTranscription: document.getElementById("liveTranscription"),
  settingsFeedback: document.getElementById("settingsFeedback"),
  activityButton: document.getElementById("activityButton"),
  activityBadge: document.getElementById("activityBadge"),
  activityDrawer: document.getElementById("activityDrawer"),
  activityLog: document.getElementById("activityLog"),
  closeActivity: document.getElementById("closeActivity"),
  clearActivity: document.getElementById("clearActivity"),
  nameDialog: document.getElementById("nameDialog"),
  nameForm: document.getElementById("nameForm"),
  nameSpeakerId: document.getElementById("nameSpeakerId"),
  speakerName: document.getElementById("speakerName"),
  assignDialog: document.getElementById("assignDialog"),
  assignForm: document.getElementById("assignForm"),
  assignSourceId: document.getElementById("assignSourceId"),
  assignTarget: document.getElementById("assignTarget"),
  unlockDialog: document.getElementById("unlockDialog"),
  unlockForm: document.getElementById("unlockForm"),
  databasePassword: document.getElementById("databasePassword"),
  unlockFeedback: document.getElementById("unlockFeedback"),
  toastRegion: document.getElementById("toastRegion"),
};

const state = {
  status: null,
  preferences: null,
  sessions: [],
  speakers: [],
  selectedSessionId: null,
  selectedSegments: [],
  recording: false,
  recordingStartedAt: null,
  recordingTimer: null,
  activeRuns: new Set(),
  progressCounts: new Map(),
  progressEventIds: new Set(),
  pollTimer: null,
  activityOpen: false,
  unseenActivity: 0,
  previewAudio: null,
  sessionLoadSequence: 0,
};

function errorText(error) {
  if (typeof error === "string") return error;
  if (error && typeof error.message === "string") return error.message;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

function addActivity(message, kind) {
  const entry = document.createElement("div");
  entry.className = "activity-entry" + (kind ? " " + kind : "");
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const copy = document.createElement("span");
  copy.textContent = message;
  entry.append(time, copy);
  elements.activityLog.append(entry);
  elements.activityLog.scrollTop = elements.activityLog.scrollHeight;
  console.log("[Recall]", message);
  if (!state.activityOpen) {
    state.unseenActivity += 1;
    elements.activityBadge.textContent = String(Math.min(99, state.unseenActivity));
    elements.activityBadge.hidden = false;
  }
}

function showToast(message, kind) {
  const toast = document.createElement("div");
  toast.className = "toast" + (kind === "error" ? " error" : "");
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 5200);
}

function setServiceStatus(configured) {
  state.status = Object.assign({}, state.status || {}, {
    soniox_key_configured: configured,
  });
  elements.serviceBadge.textContent = configured ? "Soniox configured" : "Soniox key needed";
  elements.serviceBadge.classList.toggle("warning", !configured);
  elements.keyStatus.textContent = configured ? "Configured" : "Not configured";
  elements.keyStatus.classList.toggle("warning", !configured);
  elements.deleteKeyButton.disabled = !configured;
}

function updateContentVisibility() {
  const hasSelection = Boolean(state.selectedSessionId);
  const isProcessing = state.activeRuns.size > 0;
  elements.processingState.hidden = !isProcessing;
  elements.transcriptContent.hidden = !hasSelection;
  elements.emptyState.hidden = hasSelection || isProcessing || state.recording;
  if (isProcessing) {
    const count = state.activeRuns.size;
    elements.processingTitle.textContent =
      count === 1 ? "Processing recording" : "Processing " + count + " recordings";
  }
}

function setProcessingDetail(detail) {
  elements.processingDetail.textContent = detail || "Working…";
  updateContentVisibility();
}

function setRecordingUi(recording, started) {
  state.recording = recording;
  elements.recordingBanner.hidden = !recording;
  elements.recordButton.classList.toggle("recording", recording);
  elements.recordButtonLabel.textContent = recording ? "Stop recording" : "New recording";
  elements.emptyRecordButton.disabled = recording;
  if (recording) {
    if (!state.recordingStartedAt) state.recordingStartedAt = Date.now();
    if (!state.recordingTimer) {
      state.recordingTimer = window.setInterval(updateRecordingTimer, 250);
    }
    const liveEnabled = Boolean(started && started.live_started);
    elements.livePanel.hidden = !liveEnabled;
    elements.liveStatus.textContent = liveEnabled ? "Connecting…" : "Live captions disabled";
    elements.liveTranscript.textContent = liveEnabled ? "Listening…" : "";
    if (started && started.device_name) {
      elements.conversationMeta.textContent =
        "Recording from " + started.device_name + " at " + started.sample_rate + " Hz";
    }
  } else {
    if (state.recordingTimer) window.clearInterval(state.recordingTimer);
    state.recordingTimer = null;
    state.recordingStartedAt = null;
    elements.recordingTimer.textContent = "00:00";
    elements.levelBar.style.width = "2%";
  }
  updateContentVisibility();
}

function updateRecordingTimer() {
  if (!state.recordingStartedAt) return;
  const elapsed = Date.now() - state.recordingStartedAt;
  const total = Math.floor(elapsed / 1000);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  elements.recordingTimer.textContent =
    String(minutes).padStart(2, "0") + ":" + String(seconds).padStart(2, "0");
}

async function startRecording() {
  if (state.recording) {
    await stopRecording();
    return;
  }
  if (!state.status || !state.status.soniox_key_configured) {
    showToast("Add your Soniox API key before recording.", "error");
    await openSettings();
    return;
  }
  elements.recordButton.disabled = true;
  elements.emptyRecordButton.disabled = true;
  addActivity("Starting a new recording");
  try {
    const inputDevice =
      state.preferences && state.preferences.selected_input_device
        ? state.preferences.selected_input_device
        : null;
    const started = await invoke("start_recording", { inputDevice });
    setRecordingUi(true, started);
    addActivity(
      "Recording started from " +
        started.device_name +
        (started.live_started ? "; live Soniox captions enabled" : ""),
      "success",
    );
  } catch (error) {
    const message = errorText(error);
    addActivity("Recording failed to start: " + message, "error");
    showToast(message, "error");
    setRecordingUi(false);
  } finally {
    elements.recordButton.disabled = false;
    elements.emptyRecordButton.disabled = false;
  }
}

async function stopRecording() {
  if (!state.recording) return;
  elements.recordButton.disabled = true;
  elements.stopButton.disabled = true;
  addActivity("Stopping recording");
  try {
    const path = await invoke("stop_recording");
    setRecordingUi(false);
    elements.livePanel.hidden = true;
    addActivity("Recording stopped; queueing final transcription");
    const runId = await invoke("transcribe_file_async", { path });
    trackRun(runId);
    addActivity("[" + runId.slice(0, 8) + "] Final transcription queued");
    setProcessingDetail("Uploading the recording to Soniox…");
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not stop or queue recording: " + message, "error");
    showToast(message, "error");
  } finally {
    elements.recordButton.disabled = false;
    elements.stopButton.disabled = false;
  }
}

function trackRun(runId) {
  if (!runId) return;
  state.activeRuns.add(runId);
  if (!state.progressCounts.has(runId)) state.progressCounts.set(runId, 0);
  ensurePolling();
  updateContentVisibility();
}

function finishRun(runId) {
  if (!runId) return;
  state.activeRuns.delete(runId);
  state.progressCounts.delete(runId);
  if (state.activeRuns.size === 0 && state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  updateContentVisibility();
}

function stageDescription(stage, detail) {
  const descriptions = {
    queued: "Queued for final transcription",
    "transcription:start": "Preparing final transcription",
    "soniox:upload:start": "Uploading recording to Soniox",
    "soniox:upload:done": "Upload finished",
    "soniox:transcription:start": "Starting final Soniox transcription",
    "soniox:transcription:waiting": "Waiting for Soniox",
    "soniox:transcription:status": "Soniox processing",
    "soniox:transcript:download:start": "Downloading diarized transcript",
    "soniox:transcript:download:done": "Final transcript received",
    "soniox:cleanup:start": "Removing temporary Soniox artifacts",
    "soniox:cleanup:done": "Soniox artifacts removed",
    "audio:read:start": "Preparing audio for local speaker identification",
    "audio:read:done": "Local audio prepared",
    "voiceprints:start": "Extracting and matching voiceprints locally",
    "voiceprint:new": "New voice profile created",
    "voiceprint:matched": "Known voice identified",
    "voiceprint:sample:stored": "Temporary voice preview saved",
    "voiceprints:done": "Speaker attribution finished",
    "transcription:done": "Conversation saved locally",
    "audio:cleanup:done": "Temporary local recording deleted",
    complete: "Processing complete",
    error: "Processing failed",
  };
  return detail || descriptions[stage] || stage;
}

async function handleProgressEvent(event) {
  if (!event || !event.stage) return;
  if (event.event_id) {
    if (state.progressEventIds.has(event.event_id)) return;
    state.progressEventIds.add(event.event_id);
    if (state.progressEventIds.size > 2000) {
      state.progressEventIds = new Set(Array.from(state.progressEventIds).slice(-1000));
    }
  }
  const runId = event.run_id;
  const prefix = runId ? "[" + runId.slice(0, 8) + "] " : "";
  const detail = stageDescription(event.stage, event.detail);
  const kind = event.stage === "error" || event.stage.includes("warning") ? "error" :
    event.stage === "complete" ? "success" : "";
  addActivity(prefix + event.stage + ": " + detail, kind);

  if (runId && event.stage !== "complete" && event.stage !== "error") trackRun(runId);
  if (event.stage !== "complete" && event.stage !== "error") setProcessingDetail(detail);

  if (event.stage === "complete") {
    finishRun(runId);
    const sessionId = event.detail;
    await Promise.all([loadSpeakers(), loadSessions(sessionId)]);
    if (sessionId) await selectSession(sessionId);
    showToast("Conversation transcribed and attributed.");
  } else if (event.stage === "error") {
    finishRun(runId);
    showToast(detail, "error");
  } else if (event.stage === "voiceprint:new") {
    showToast("A new voice needs a name.");
  }
}

async function pollProgress() {
  for (const runId of Array.from(state.activeRuns)) {
    try {
      const events = await invoke("get_progress", { runId });
      state.progressCounts.set(runId, events.length);
      for (const event of events) await handleProgressEvent(event);
    } catch (error) {
      addActivity("[" + runId.slice(0, 8) + "] Progress check failed: " + errorText(error), "error");
    }
  }
}

function ensurePolling() {
  if (state.pollTimer) return;
  state.pollTimer = window.setInterval(pollProgress, 1400);
}

function sessionTitle(session) {
  return (session.title || "").trim() || "Untitled conversation";
}

function sessionDate(session) {
  const date = new Date(session.created_at);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function loadSessions(preferredId) {
  try {
    state.sessions = await invoke("list_sessions");
    renderSessions();
    if (preferredId && state.sessions.some((session) => session.id === preferredId)) {
      state.selectedSessionId = preferredId;
      renderSessions();
    } else if (
      state.selectedSessionId &&
      !state.sessions.some((session) => session.id === state.selectedSessionId)
    ) {
      state.selectedSessionId = null;
      state.selectedSegments = [];
    }
    updateContentVisibility();
  } catch (error) {
    addActivity("Could not load conversations: " + errorText(error), "error");
  }
}

function renderSessions() {
  const query = elements.conversationSearch.value.trim().toLowerCase();
  const filtered = state.sessions.filter((session) => {
    const searchable = (sessionTitle(session) + " " + (session.transcript || "")).toLowerCase();
    return !query || searchable.includes(query);
  });
  elements.sessionsList.replaceChildren();
  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "sidebar-empty";
    empty.textContent = query ? "No matching conversations." : "No conversations yet.";
    elements.sessionsList.append(empty);
    return;
  }
  for (const session of filtered) {
    const button = document.createElement("button");
    button.type = "button";
    button.className =
      "session-item" + (session.id === state.selectedSessionId ? " selected" : "");
    const title = document.createElement("strong");
    title.textContent = sessionTitle(session);
    const meta = document.createElement("span");
    const parts = [sessionDate(session)];
    if (session.duration_ms > 0) parts.push(formatDuration(session.duration_ms));
    meta.textContent = parts.filter(Boolean).join(" · ");
    button.append(title, meta);
    button.addEventListener("click", () => selectSession(session.id));
    elements.sessionsList.append(button);
  }
}

async function selectSession(sessionId) {
  const session = state.sessions.find((candidate) => candidate.id === sessionId);
  if (!session) return;
  const sequence = ++state.sessionLoadSequence;
  state.selectedSessionId = sessionId;
  renderSessions();
  elements.conversationTitle.disabled = false;
  elements.conversationTitle.value = sessionTitle(session);
  const parts = [sessionDate(session)];
  if (session.duration_ms > 0) parts.push(formatDuration(session.duration_ms));
  elements.conversationMeta.textContent = parts.join(" · ");
  elements.deleteSessionButton.hidden = false;
  elements.saveState.textContent = "Loading…";
  elements.transcriptContent.hidden = false;
  elements.emptyState.hidden = true;
  try {
    const segments = await invoke("list_segments", { sessionId });
    if (sequence !== state.sessionLoadSequence) return;
    state.selectedSegments = segments;
    renderTranscript(session);
    elements.saveState.textContent = "Saved locally";
  } catch (error) {
    addActivity("Could not load conversation: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
  updateContentVisibility();
}

function renderTranscript(session) {
  elements.segmentsList.replaceChildren();
  if (!state.selectedSegments.length) {
    elements.legacyTranscript.hidden = false;
    elements.legacyTranscript.textContent =
      (session && session.transcript) || "This conversation has no transcript.";
    return;
  }
  elements.legacyTranscript.hidden = true;
  for (const segment of state.selectedSegments) {
    const row = document.createElement("article");
    row.className = "segment";
    const speakerColumn = document.createElement("div");
    speakerColumn.className = "segment-speaker";
    const select = buildSpeakerSelect(segment.speaker_id, segment.speaker_label);
    select.setAttribute("aria-label", "Speaker for this intervention");
    select.addEventListener("change", async () => {
      await assignSegmentSpeaker(segment, select.value || null);
    });
    const time = document.createElement("time");
    time.textContent = formatTimestamp(segment.start_ms);
    speakerColumn.append(select, time);

    const text = document.createElement("textarea");
    text.className = "segment-text";
    text.value = segment.text || "";
    text.setAttribute("aria-label", "Transcript intervention");
    autoResize(text);
    text.addEventListener("input", () => {
      autoResize(text);
      elements.saveState.textContent = "Unsaved changes";
    });
    text.addEventListener("blur", async () => {
      const value = text.value.trim();
      if (value === (segment.text || "").trim()) {
        elements.saveState.textContent = "Saved locally";
        return;
      }
      await saveSegmentText(segment, value);
    });
    text.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") text.blur();
    });
    row.append(speakerColumn, text);
    elements.segmentsList.append(row);
  }
}

function buildSpeakerSelect(selectedId, fallbackLabel) {
  const select = document.createElement("select");
  const unknown = document.createElement("option");
  unknown.value = "";
  unknown.textContent = fallbackLabel && !selectedId ? fallbackLabel : "Unknown speaker";
  select.append(unknown);
  for (const speaker of state.speakers) {
    const option = document.createElement("option");
    option.value = speaker.id;
    option.textContent = speaker.label || "Unnamed voice";
    option.selected = speaker.id === selectedId;
    select.append(option);
  }
  return select;
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.max(70, textarea.scrollHeight) + "px";
}

async function saveSegmentText(segment, text) {
  elements.saveState.textContent = "Saving…";
  try {
    await invoke("update_segment_text", {
      segmentId: segment.id,
      sessionId: segment.session_id,
      text,
    });
    segment.text = text;
    syncSelectedSessionTranscript();
    elements.saveState.textContent = "Saved locally";
    addActivity("Transcript intervention updated", "success");
  } catch (error) {
    elements.saveState.textContent = "Save failed";
    addActivity("Could not save transcript edit: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

async function assignSegmentSpeaker(segment, speakerId) {
  elements.saveState.textContent = "Saving…";
  try {
    await invoke("assign_segment_speaker", {
      segmentId: segment.id,
      sessionId: segment.session_id,
      speakerId,
    });
    const speaker = state.speakers.find((candidate) => candidate.id === speakerId);
    segment.speaker_id = speakerId;
    segment.speaker_label = speaker ? speaker.label : null;
    syncSelectedSessionTranscript();
    elements.saveState.textContent = "Saved locally";
    addActivity(
      "Intervention assigned to " + (speaker ? speaker.label : "Unknown speaker"),
      "success",
    );
  } catch (error) {
    elements.saveState.textContent = "Save failed";
    addActivity("Could not assign speaker: " + errorText(error), "error");
    showToast(errorText(error), "error");
    await selectSession(segment.session_id);
  }
}

function syncSelectedSessionTranscript() {
  const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
  if (session) session.transcript = transcriptFromSegments(state.selectedSegments);
}

async function saveConversationTitle() {
  if (!state.selectedSessionId) return;
  const title = elements.conversationTitle.value.trim();
  if (!title) {
    const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
    elements.conversationTitle.value = sessionTitle(session);
    return;
  }
  try {
    await invoke("update_session_title", {
      sessionId: state.selectedSessionId,
      title,
    });
    const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
    if (session) session.title = title;
    renderSessions();
    addActivity("Conversation title saved", "success");
  } catch (error) {
    addActivity("Could not save conversation title: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

async function deleteSelectedSession() {
  if (!state.selectedSessionId) return;
  const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
  if (
    !window.confirm(
      "Delete “" +
        sessionTitle(session) +
        "” and its transcript? Saved people and voiceprints are kept.",
    )
  ) {
    return;
  }
  try {
    await invoke("delete_session", { sessionId: state.selectedSessionId });
    addActivity("Conversation deleted", "success");
    state.selectedSessionId = null;
    state.selectedSegments = [];
    elements.conversationTitle.value = "New conversation";
    elements.conversationTitle.disabled = true;
    elements.conversationMeta.textContent = "Record a conversation to begin";
    elements.deleteSessionButton.hidden = true;
    await Promise.all([loadSessions(), loadSpeakers()]);
  } catch (error) {
    addActivity("Could not delete conversation: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

async function loadSpeakers() {
  try {
    state.speakers = await invoke("list_speakers_with_stats");
    renderSpeakers();
  } catch (error) {
    addActivity("Could not load voice profiles: " + errorText(error), "error");
  }
}

function speakerInitial(label) {
  const value = (label || "?").trim();
  return value ? value[0].toUpperCase() : "?";
}

function renderSpeakers() {
  elements.speakersList.replaceChildren();
  if (!state.speakers.length) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent =
      "No voice profiles yet. Recall creates them after the first diarized recording.";
    elements.speakersList.append(empty);
    return;
  }
  for (const speaker of state.speakers) {
    const label = speaker.label || "Unnamed voice";
    const provisional = isProvisionalLabel(label);
    const card = document.createElement("article");
    card.className = "speaker-card" + (provisional ? " provisional" : "");
    const header = document.createElement("div");
    header.className = "speaker-header";
    const identity = document.createElement("div");
    identity.className = "speaker-identity";
    const avatar = document.createElement("div");
    avatar.className = "speaker-avatar";
    avatar.textContent = speakerInitial(label);
    const copy = document.createElement("div");
    const name = document.createElement("div");
    name.className = "speaker-name";
    name.textContent = label;
    const meta = document.createElement("div");
    meta.className = "speaker-meta";
    meta.textContent =
      speaker.conversation_count +
      " conversation" +
      (speaker.conversation_count === 1 ? "" : "s") +
      " · " +
      speaker.embedding_count +
      " voiceprint" +
      (speaker.embedding_count === 1 ? "" : "s");
    copy.append(name, meta);
    identity.append(avatar, copy);
    header.append(identity);
    card.append(header);
    if (provisional) {
      const tag = document.createElement("span");
      tag.className = "new-voice-tag";
      tag.textContent = "Needs identification";
      card.append(tag);
    }

    const actions = document.createElement("div");
    actions.className = "speaker-actions";
    const preview = actionButton("Preview", () => previewSpeaker(speaker));
    preview.disabled = speaker.sample_count === 0;
    preview.title =
      speaker.sample_count === 0
        ? "No sample is retained after a person is named"
        : "Play the excerpt used for this voiceprint";
    actions.append(preview);
    if (provisional) {
      const nameButton = actionButton("Name", () => openNameDialog(speaker), "primary-mini");
      const assignButton = actionButton("Assign…", () => openAssignDialog(speaker));
      assignButton.disabled = !state.speakers.some(
        (candidate) => candidate.id !== speaker.id && !isProvisionalLabel(candidate.label),
      );
      assignButton.title = assignButton.disabled
        ? "Name another profile before assigning this voice to it"
        : "Assign this voice to a known person";
      actions.append(nameButton, assignButton);
    } else {
      actions.append(
        actionButton("Rename", () => openNameDialog(speaker)),
        actionButton("Merge…", () => openAssignDialog(speaker)),
      );
    }
    actions.append(actionButton("Delete", () => deleteSpeaker(speaker), "danger-mini"));
    card.append(actions);
    elements.speakersList.append(card);
  }
}

function actionButton(label, handler, className) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  if (className) button.className = className;
  button.addEventListener("click", handler);
  return button;
}

async function previewSpeaker(speaker) {
  addActivity("Loading preview for " + speaker.label);
  try {
    const samples = await invoke("get_speaker_samples", { speakerId: speaker.id });
    if (!samples.length) {
      showToast("No sample is retained for this profile.", "error");
      addActivity("No preview is retained for " + speaker.label, "error");
      await loadSpeakers();
      return;
    }
    if (state.previewAudio) state.previewAudio.pause();
    state.previewAudio = new Audio("data:audio/wav;base64," + samples[0].sample_b64);
    await state.previewAudio.play();
    addActivity("Playing voice preview for " + speaker.label, "success");
  } catch (error) {
    addActivity("Could not play voice preview: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

function openNameDialog(speaker) {
  elements.nameSpeakerId.value = speaker.id;
  elements.speakerName.value = isProvisionalLabel(speaker.label) ? "" : speaker.label || "";
  elements.nameDialog.showModal();
  window.setTimeout(() => elements.speakerName.focus(), 0);
}

async function saveSpeakerName(event) {
  event.preventDefault();
  const speakerId = elements.nameSpeakerId.value;
  const name = elements.speakerName.value.trim();
  if (!speakerId || !name) return;
  try {
    await invoke("rename_speaker", { speakerId, newLabel: name });
    elements.nameDialog.close();
    addActivity("Voice profile named " + name + "; temporary sample deleted", "success");
    showToast("Voice profile saved as " + name + ".");
    await Promise.all([loadSpeakers(), loadSessions()]);
    if (state.selectedSessionId) await selectSession(state.selectedSessionId);
  } catch (error) {
    addActivity("Could not name voice profile: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

function openAssignDialog(source) {
  const candidates = state.speakers.filter(
    (candidate) => candidate.id !== source.id && !isProvisionalLabel(candidate.label),
  );
  if (!candidates.length) {
    showToast("There are no other named people to assign this voice to.", "error");
    return;
  }
  elements.assignSourceId.value = source.id;
  elements.assignTarget.replaceChildren();
  for (const candidate of candidates) {
    const option = document.createElement("option");
    option.value = candidate.id;
    option.textContent = candidate.label || "Unnamed voice";
    elements.assignTarget.append(option);
  }
  const keep = elements.assignForm.querySelector('input[name="voiceprintMode"][value="keep"]');
  keep.checked = true;
  elements.assignDialog.showModal();
}

async function assignVoiceProfile(event) {
  event.preventDefault();
  const sourceId = elements.assignSourceId.value;
  const targetId = elements.assignTarget.value;
  const mode = elements.assignForm.querySelector('input[name="voiceprintMode"]:checked');
  const replaceEmbeddings = mode && mode.value === "replace";
  if (!sourceId || !targetId) return;
  try {
    await invoke("merge_speakers", {
      sourceId,
      targetId,
      replaceEmbeddings,
    });
    const target = state.speakers.find((speaker) => speaker.id === targetId);
    elements.assignDialog.close();
    addActivity(
      "Voice profile assigned to " +
        (target ? target.label : "existing person") +
        (replaceEmbeddings ? "; prior voiceprints replaced" : "; voiceprints combined"),
      "success",
    );
    showToast("Voice profile assignment saved.");
    await Promise.all([loadSpeakers(), loadSessions()]);
    if (state.selectedSessionId) await selectSession(state.selectedSessionId);
  } catch (error) {
    addActivity("Could not assign voice profile: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

async function deleteSpeaker(speaker) {
  const label = speaker.label || "this profile";
  if (
    !window.confirm(
      "Delete " +
        label +
        " and its local voiceprints? Existing transcript turns will become unattributed.",
    )
  ) {
    return;
  }
  try {
    await invoke("delete_speaker", { speakerId: speaker.id });
    addActivity("Voice profile " + label + " deleted", "success");
    await Promise.all([loadSpeakers(), loadSessions()]);
    if (state.selectedSessionId) await selectSession(state.selectedSessionId);
  } catch (error) {
    addActivity("Could not delete voice profile: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

async function loadSettingsData() {
  const [status, preferences, devices] = await Promise.all([
    invoke("app_status"),
    invoke("get_preferences"),
    invoke("list_input_devices"),
  ]);
  state.status = status;
  state.preferences = preferences;
  setServiceStatus(status.soniox_key_configured);
  if (status.recording && !state.recording) {
    setRecordingUi(true, {
      device_name: preferences.selected_input_device || "selected input",
      sample_rate: 0,
      live_started: preferences.live_transcription,
    });
  }
  elements.languageHints.value = (preferences.language_hints || []).join(", ");
  elements.liveTranscription.checked = preferences.live_transcription;
  elements.inputDevice.replaceChildren();
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "System default microphone";
  elements.inputDevice.append(defaultOption);
  let selectedAvailable = !preferences.selected_input_device;
  for (const device of devices) {
    const option = document.createElement("option");
    option.value = device.name;
    option.textContent = device.name + (device.is_default ? " (default)" : "");
    option.selected = preferences.selected_input_device === device.name;
    if (option.selected) selectedAvailable = true;
    elements.inputDevice.append(option);
  }
  if (!selectedAvailable && preferences.selected_input_device) {
    const unavailable = document.createElement("option");
    unavailable.value = preferences.selected_input_device;
    unavailable.textContent = preferences.selected_input_device + " (unavailable)";
    unavailable.selected = true;
    elements.inputDevice.append(unavailable);
  }
  if (!status.speaker_model_available) {
    addActivity("Local speaker model is missing; automatic voice matching is unavailable", "error");
  }
}

async function openSettings() {
  elements.settingsFeedback.textContent = "Loading…";
  try {
    await loadSettingsData();
    elements.settingsFeedback.textContent = "";
    if (!elements.settingsDialog.open) elements.settingsDialog.showModal();
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not load settings: " + message, "error");
    if (!elements.settingsDialog.open) elements.settingsDialog.showModal();
  }
}

async function saveSonioxKey() {
  const apiKey = elements.sonioxKey.value.trim();
  if (!apiKey) {
    elements.settingsFeedback.textContent = "Paste a key first.";
    return;
  }
  elements.saveKeyButton.disabled = true;
  elements.settingsFeedback.textContent = "Saving to macOS Keychain…";
  try {
    await invoke("save_soniox_key", { apiKey });
    elements.sonioxKey.value = "";
    setServiceStatus(true);
    elements.settingsFeedback.textContent = "Key saved securely.";
    addActivity("Soniox API key saved in macOS Keychain", "success");
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not save Soniox API key: " + message, "error");
  } finally {
    elements.saveKeyButton.disabled = false;
  }
}

async function deleteSonioxKey() {
  if (!window.confirm("Remove the Soniox API key from macOS Keychain?")) return;
  try {
    await invoke("delete_soniox_key");
    setServiceStatus(false);
    elements.settingsFeedback.textContent = "Key removed.";
    addActivity("Soniox API key removed from macOS Keychain", "success");
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not remove Soniox API key: " + message, "error");
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const selectedInputDevice = elements.inputDevice.value || null;
  const languageHints = parseLanguageHints(elements.languageHints.value);
  const liveTranscription = elements.liveTranscription.checked;
  elements.settingsFeedback.textContent = "Saving…";
  try {
    await invoke("save_preferences", {
      selectedInputDevice,
      languageHints,
      liveTranscription,
    });
    state.preferences = {
      encryption_enabled: state.preferences ? state.preferences.encryption_enabled : false,
      selected_input_device: selectedInputDevice,
      language_hints: languageHints,
      live_transcription: liveTranscription,
    };
    elements.settingsFeedback.textContent = "Saved.";
    elements.settingsDialog.close();
    addActivity("Recording preferences saved", "success");
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not save settings: " + message, "error");
  }
}

async function unlockDatabase(event) {
  event.preventDefault();
  const password = elements.databasePassword.value;
  if (!password) return;
  elements.unlockFeedback.textContent = "Unlocking…";
  try {
    await invoke("unlock_db", { password });
    elements.databasePassword.value = "";
    await loadSettingsData();
    await Promise.all([loadSpeakers(), loadSessions()]);
    if (state.sessions.length) await selectSession(state.sessions[0].id);
    elements.unlockDialog.close();
    elements.unlockFeedback.textContent = "";
    addActivity("Existing encrypted database unlocked", "success");
    updateContentVisibility();
    if (!state.status.soniox_key_configured) await openSettings();
  } catch (error) {
    const message = errorText(error);
    elements.databasePassword.value = "";
    elements.unlockFeedback.textContent = message;
    addActivity("Could not unlock existing database: " + message, "error");
  }
}

function setActivityOpen(open) {
  state.activityOpen = open;
  elements.activityDrawer.classList.toggle("open", open);
  elements.activityDrawer.setAttribute("aria-hidden", String(!open));
  if (open) {
    state.unseenActivity = 0;
    elements.activityBadge.hidden = true;
  }
}

async function registerListeners() {
  await listen("transcription:progress", (event) => {
    void handleProgressEvent(event.payload);
  });
  await listen("transcription:queued", (event) => {
    const runId = event.payload;
    trackRun(runId);
    addActivity("[" + runId.slice(0, 8) + "] Queued from the menu bar");
  });
  await listen("recording:started", (event) => {
    setRecordingUi(true, event.payload);
  });
  await listen("recording:stopped", () => {
    setRecordingUi(false);
    elements.livePanel.hidden = true;
  });
  await listen("recording:error", (event) => {
    const message = errorText(event.payload);
    addActivity("Recording error: " + message, "error");
    showToast(message, "error");
  });
  await listen("recording:level", (event) => {
    const level = Math.max(0, Math.min(1, Number(event.payload.level) || 0));
    elements.levelBar.style.width = Math.max(2, level * 100) + "%";
  });
  await listen("live-transcription", (event) => {
    const payload = event.payload || {};
    elements.liveStatus.textContent = payload.status || "Live";
    if (payload.text) elements.liveTranscript.textContent = payload.text;
    if (payload.error) {
      addActivity("Live transcription error: " + payload.error, "error");
      showToast(payload.error, "error");
    }
    if (payload.finished && !state.recording) elements.livePanel.hidden = true;
  });
}

function bindInterface() {
  elements.recordButton.addEventListener("click", startRecording);
  elements.emptyRecordButton.addEventListener("click", startRecording);
  elements.stopButton.addEventListener("click", stopRecording);
  elements.refreshSessions.addEventListener("click", () => loadSessions());
  elements.refreshSpeakers.addEventListener("click", loadSpeakers);
  elements.conversationSearch.addEventListener("input", renderSessions);
  elements.conversationTitle.addEventListener("change", saveConversationTitle);
  elements.conversationTitle.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      elements.conversationTitle.blur();
    }
  });
  elements.deleteSessionButton.addEventListener("click", deleteSelectedSession);
  elements.settingsButton.addEventListener("click", openSettings);
  elements.saveKeyButton.addEventListener("click", saveSonioxKey);
  elements.deleteKeyButton.addEventListener("click", deleteSonioxKey);
  elements.settingsForm.addEventListener("submit", saveSettings);
  elements.nameForm.addEventListener("submit", saveSpeakerName);
  elements.assignForm.addEventListener("submit", assignVoiceProfile);
  elements.unlockForm.addEventListener("submit", unlockDatabase);
  elements.activityButton.addEventListener("click", () => setActivityOpen(!state.activityOpen));
  elements.closeActivity.addEventListener("click", () => setActivityOpen(false));
  elements.clearActivity.addEventListener("click", () => {
    elements.activityLog.replaceChildren();
    state.unseenActivity = 0;
    elements.activityBadge.hidden = true;
  });
  for (const button of document.querySelectorAll("[data-close-dialog]")) {
    button.addEventListener("click", () => {
      const dialog = document.getElementById(button.dataset.closeDialog);
      if (dialog && dialog.open) dialog.close();
    });
  }
}

async function initialize() {
  bindInterface();
  addActivity("Starting Recall desktop");
  try {
    await registerListeners();
    addActivity("Desktop event listeners ready", "success");
    await loadSettingsData();
    if (state.status.needs_password) {
      addActivity("Existing encrypted database needs its password");
      elements.unlockDialog.showModal();
      updateContentVisibility();
      return;
    }
    await Promise.all([loadSpeakers(), loadSessions()]);
    if (state.sessions.length) await selectSession(state.sessions[0].id);
    updateContentVisibility();
    addActivity("Recall is ready", "success");
    if (!state.status.soniox_key_configured) {
      window.setTimeout(() => {
        if (!elements.settingsDialog.open) elements.settingsDialog.showModal();
      }, 250);
    }
  } catch (error) {
    const message = errorText(error);
    addActivity("Recall could not initialize: " + message, "error");
    showToast(message, "error");
    setActivityOpen(true);
  }
}

void initialize();

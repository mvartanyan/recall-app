import {
  ONBOARDING_VERSION,
  buildTranslationPlan,
  contentMode,
  filterSessions,
  formatDuration,
  formatTimestamp,
  isNearScrollBottom,
  isProvisionalLabel,
  parseLanguageHints,
  parseNoTranslationLanguages,
  recapTabAvailability,
  shouldShowOnboarding,
  transcriptFromSegments,
  translatedSegmentText,
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
  jumpToLiveButton: document.getElementById("jumpToLiveButton"),
  sessionsList: document.getElementById("sessionsList"),
  conversationSearch: document.getElementById("conversationSearch"),
  conversationSpeakerFilter: document.getElementById("conversationSpeakerFilter"),
  refreshSessions: document.getElementById("refreshSessions"),
  conversationTitle: document.getElementById("conversationTitle"),
  conversationMeta: document.getElementById("conversationMeta"),
  agendaButton: document.getElementById("agendaButton"),
  recapButton: document.getElementById("recapButton"),
  deleteSessionButton: document.getElementById("deleteSessionButton"),
  serviceBadge: document.getElementById("serviceBadge"),
  openaiServiceBadge: document.getElementById("openaiServiceBadge"),
  emptyState: document.getElementById("emptyState"),
  processingState: document.getElementById("processingState"),
  processingTitle: document.getElementById("processingTitle"),
  processingDetail: document.getElementById("processingDetail"),
  transcriptContent: document.getElementById("transcriptContent"),
  recapStaleBanner: document.getElementById("recapStaleBanner"),
  staleRegenerateButton: document.getElementById("staleRegenerateButton"),
  recapTabs: document.getElementById("recapTabs"),
  transcriptTab: document.getElementById("transcriptTab"),
  executiveTab: document.getElementById("executiveTab"),
  fullSummaryTab: document.getElementById("fullSummaryTab"),
  actionsTab: document.getElementById("actionsTab"),
  agendaCoverageTab: document.getElementById("agendaCoverageTab"),
  transcriptTabPanel: document.getElementById("transcriptTabPanel"),
  generatedTabPanel: document.getElementById("generatedTabPanel"),
  generatedEyebrow: document.getElementById("generatedEyebrow"),
  generatedTitle: document.getElementById("generatedTitle"),
  generatedContent: document.getElementById("generatedContent"),
  generatedLanguageToggle: document.getElementById("generatedLanguageToggle"),
  showOriginalButton: document.getElementById("showOriginalButton"),
  showEnglishButton: document.getElementById("showEnglishButton"),
  copyTranscriptText: document.getElementById("copyTranscriptText"),
  copyTranscriptMarkdown: document.getElementById("copyTranscriptMarkdown"),
  copyGeneratedText: document.getElementById("copyGeneratedText"),
  copyGeneratedMarkdown: document.getElementById("copyGeneratedMarkdown"),
  segmentsList: document.getElementById("segmentsList"),
  legacyTranscript: document.getElementById("legacyTranscript"),
  saveState: document.getElementById("saveState"),
  speakersList: document.getElementById("speakersList"),
  refreshSpeakers: document.getElementById("refreshSpeakers"),
  voiceLibraryButton: document.getElementById("voiceLibraryButton"),
  voiceLibraryDialog: document.getElementById("voiceLibraryDialog"),
  voiceLibraryList: document.getElementById("voiceLibraryList"),
  confirmationDialog: document.getElementById("confirmationDialog"),
  confirmationForm: document.getElementById("confirmationForm"),
  confirmationTitle: document.getElementById("confirmationTitle"),
  confirmationMessage: document.getElementById("confirmationMessage"),
  confirmationCancel: document.getElementById("confirmationCancel"),
  confirmationAccept: document.getElementById("confirmationAccept"),
  settingsButton: document.getElementById("settingsButton"),
  settingsDialog: document.getElementById("settingsDialog"),
  settingsForm: document.getElementById("settingsForm"),
  sonioxKey: document.getElementById("sonioxKey"),
  keyStatus: document.getElementById("keyStatus"),
  saveKeyButton: document.getElementById("saveKeyButton"),
  deleteKeyButton: document.getElementById("deleteKeyButton"),
  openaiKey: document.getElementById("openaiKey"),
  openaiKeyStatus: document.getElementById("openaiKeyStatus"),
  saveOpenAIKeyButton: document.getElementById("saveOpenAIKeyButton"),
  deleteOpenAIKeyButton: document.getElementById("deleteOpenAIKeyButton"),
  openaiModel: document.getElementById("openaiModel"),
  inputDevice: document.getElementById("inputDevice"),
  languageHints: document.getElementById("languageHints"),
  noTranslationLanguages: document.getElementById("noTranslationLanguages"),
  liveTranscription: document.getElementById("liveTranscription"),
  settingsFeedback: document.getElementById("settingsFeedback"),
  agendaDialog: document.getElementById("agendaDialog"),
  agendaForm: document.getElementById("agendaForm"),
  agendaCurrent: document.getElementById("agendaCurrent"),
  agendaText: document.getElementById("agendaText"),
  attachAgendaButton: document.getElementById("attachAgendaButton"),
  removeAgendaButton: document.getElementById("removeAgendaButton"),
  agendaFeedback: document.getElementById("agendaFeedback"),
  saveAgendaTextButton: document.getElementById("saveAgendaTextButton"),
  unresolvedDialog: document.getElementById("unresolvedDialog"),
  unresolvedList: document.getElementById("unresolvedList"),
  cancelUnresolvedButton: document.getElementById("cancelUnresolvedButton"),
  reviewUnresolvedButton: document.getElementById("reviewUnresolvedButton"),
  recapAnywayButton: document.getElementById("recapAnywayButton"),
  recapProgressDialog: document.getElementById("recapProgressDialog"),
  recapProgressTitle: document.getElementById("recapProgressTitle"),
  recapProgressDetail: document.getElementById("recapProgressDetail"),
  recapProgressSpinner: document.getElementById("recapProgressSpinner"),
  recapProgressClose: document.getElementById("recapProgressClose"),
  activityButton: document.getElementById("activityButton"),
  activityBadge: document.getElementById("activityBadge"),
  activityDrawer: document.getElementById("activityDrawer"),
  activityLog: document.getElementById("activityLog"),
  closeActivity: document.getElementById("closeActivity"),
  clearActivity: document.getElementById("clearActivity"),
  onboardingDialog: document.getElementById("onboardingDialog"),
  onboardingExploreButton: document.getElementById("onboardingExploreButton"),
  onboardingSettingsButton: document.getElementById("onboardingSettingsButton"),
  gettingStartedButton: document.getElementById("gettingStartedButton"),
  nameDialog: document.getElementById("nameDialog"),
  nameForm: document.getElementById("nameForm"),
  nameSpeakerId: document.getElementById("nameSpeakerId"),
  nameDialogTitle: document.getElementById("nameDialogTitle"),
  speakerName: document.getElementById("speakerName"),
  saveSpeakerNameButton: document.getElementById("saveSpeakerNameButton"),
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
  recordingSource: null,
  activeRuns: new Set(),
  queueingProcessing: false,
  processingDetail: "Preparing final transcription…",
  progressCounts: new Map(),
  progressEventIds: new Set(),
  pollTimer: null,
  livePollTimer: null,
  livePollErrorLogged: false,
  activityOpen: false,
  unseenActivity: 0,
  previewAudio: null,
  sessionLoadSequence: 0,
  lastLiveStatus: null,
  lastLiveSignature: null,
  liveHasText: false,
  liveEnabledForRecording: false,
  liveFollow: true,
  voiceFilteredSessionIds: null,
  voiceFilterSequence: 0,
  confirmationResolve: null,
  confirmationPreviousFocus: null,
  titleResizeFrame: null,
  transcriptResizeFrame: null,
  recapState: null,
  activeRecapTab: "transcript",
  generatedLanguage: "original",
  recapRequestSessionId: null,
  translationWarnings: new Set(),
  onboardingAcknowledgedThisLaunch: false,
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

function onboardingIsDue() {
  if (state.onboardingAcknowledgedThisLaunch) return false;
  return shouldShowOnboarding(state.preferences?.onboarding_version);
}

async function acknowledgeOnboarding() {
  await invoke("complete_onboarding", { version: ONBOARDING_VERSION });
  state.onboardingAcknowledgedThisLaunch = true;
  state.preferences = {
    ...state.preferences,
    onboarding_version: ONBOARDING_VERSION,
  };
}

function openOnboarding() {
  if (elements.settingsDialog.open) elements.settingsDialog.close();
  if (!elements.onboardingDialog.open) {
    elements.onboardingDialog.showModal();
    addActivity("Getting started guide opened");
  }
}

async function finishOnboarding(openSettingsAfter) {
  try {
    await acknowledgeOnboarding();
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not save the getting started state: " + message, "error");
    showToast(message, "error");
    return;
  }
  if (elements.onboardingDialog.open) elements.onboardingDialog.close();
  addActivity("Getting started guide completed", "success");
  if (openSettingsAfter) await openSettings();
}

async function openExternalUrl(url, button) {
  button.disabled = true;
  try {
    await invoke("open_external_url", { url });
    addActivity("Opened an official setup page in the default browser", "success");
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not open the setup page: " + message, "error");
    showToast(message, "error");
  } finally {
    button.disabled = false;
  }
}

function scheduleInitialSetupPrompt() {
  window.setTimeout(() => {
    if (onboardingIsDue()) {
      openOnboarding();
    } else if (!state.status.soniox_key_configured) {
      void openSettings();
    }
  }, 250);
}

function requestConfirmation({ title, message, acceptLabel = "Delete" }) {
  if (state.confirmationResolve) return Promise.resolve(false);
  elements.confirmationTitle.textContent = title;
  elements.confirmationMessage.textContent = message;
  elements.confirmationAccept.textContent = acceptLabel;
  state.confirmationPreviousFocus = document.activeElement;
  elements.confirmationDialog.hidden = false;
  window.setTimeout(() => elements.confirmationAccept.focus(), 0);
  return new Promise((resolve) => {
    state.confirmationResolve = resolve;
  });
}

function settleConfirmation(confirmed) {
  const resolve = state.confirmationResolve;
  state.confirmationResolve = null;
  elements.confirmationDialog.hidden = true;
  if (state.confirmationPreviousFocus && state.confirmationPreviousFocus.isConnected) {
    state.confirmationPreviousFocus.focus();
  }
  state.confirmationPreviousFocus = null;
  if (resolve) resolve(Boolean(confirmed));
}

function setServiceStatus(configured) {
  state.status = Object.assign({}, state.status || {}, {
    soniox_key_configured: configured,
  });
  elements.serviceBadge.textContent = "Soniox key needed";
  elements.serviceBadge.hidden = configured;
  elements.serviceBadge.classList.add("warning");
  elements.keyStatus.textContent = "Key needed";
  elements.keyStatus.hidden = configured;
  elements.keyStatus.classList.add("warning");
  elements.deleteKeyButton.disabled = !configured;
}

function setOpenAIStatus(configured) {
  state.status = Object.assign({}, state.status || {}, {
    openai_key_configured: configured,
  });
  elements.openaiKeyStatus.textContent = "Key needed";
  elements.openaiKeyStatus.hidden = configured;
  elements.openaiKeyStatus.classList.add("warning");
  elements.openaiServiceBadge.textContent = "OpenAI key needed";
  elements.openaiServiceBadge.hidden = configured;
  elements.openaiServiceBadge.classList.add("warning");
  elements.deleteOpenAIKeyButton.disabled = !configured;
}

function updateContentVisibility() {
  const mode = contentMode({
    recording: state.recording,
    queueing: state.queueingProcessing,
    processingCount: state.activeRuns.size,
    selectedSessionId: state.selectedSessionId,
  });
  elements.livePanel.hidden = mode !== "recording";
  elements.processingState.hidden = mode !== "processing";
  elements.transcriptContent.hidden = mode !== "conversation";
  elements.emptyState.hidden = mode !== "empty";
  if (mode === "conversation") scheduleTranscriptResize();

  if (mode === "processing") {
    const count = Math.max(1, state.activeRuns.size);
    elements.processingTitle.textContent =
      count === 1 ? "Processing recording" : "Processing " + count + " recordings";
    elements.processingDetail.textContent = state.processingDetail;
  }

  if (mode === "recording") {
    elements.conversationTitle.disabled = true;
    elements.conversationTitle.value = "Recording in progress";
    elements.conversationMeta.textContent =
      state.recordingSource || "Listening to the selected audio input";
    elements.deleteSessionButton.hidden = true;
    elements.agendaButton.hidden = true;
    elements.recapButton.hidden = true;
  } else if (mode === "processing") {
    elements.conversationTitle.disabled = true;
    elements.conversationTitle.value =
      state.activeRuns.size > 1 ? "Processing recordings" : "Processing recording";
    elements.conversationMeta.textContent = state.processingDetail;
    elements.deleteSessionButton.hidden = true;
    elements.agendaButton.hidden = true;
    elements.recapButton.hidden = true;
  } else if (mode === "conversation") {
    const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
    if (session) {
      elements.conversationTitle.disabled = false;
      elements.conversationTitle.value = sessionTitle(session);
      const parts = [sessionDate(session)];
      if (session.duration_ms > 0) parts.push(formatDuration(session.duration_ms));
      elements.conversationMeta.textContent = parts.filter(Boolean).join(" · ");
      elements.deleteSessionButton.hidden = false;
      elements.agendaButton.hidden = false;
      elements.recapButton.hidden = false;
      const hasRecap = Boolean(state.recapState && state.recapState.recap);
      elements.agendaButton.textContent =
        state.recapState && state.recapState.agenda ? "Edit agenda" : "Add agenda";
      elements.recapButton.textContent = hasRecap ? "Regenerate recap" : "Recap";
      elements.recapButton.disabled = Boolean(
        state.recapRequestSessionId === state.selectedSessionId ||
          (state.recapState && state.recapState.in_flight),
      );
    }
  } else {
    elements.conversationTitle.disabled = true;
    elements.conversationTitle.value = "New conversation";
    elements.conversationMeta.textContent = "Record a conversation to begin";
    elements.deleteSessionButton.hidden = true;
    elements.agendaButton.hidden = true;
    elements.recapButton.hidden = true;
  }
  scheduleConversationTitleResize();
}

function setProcessingDetail(detail) {
  state.processingDetail = detail || "Working…";
  updateContentVisibility();
}

function scrollLiveToLatest() {
  window.requestAnimationFrame(() => {
    elements.liveTranscript.scrollTop = elements.liveTranscript.scrollHeight;
  });
}

function setLiveFollow(following, scroll = true) {
  state.liveFollow = Boolean(following);
  elements.jumpToLiveButton.hidden = state.liveFollow;
  if (state.liveFollow && scroll) scrollLiveToLatest();
}

function handleLiveScroll() {
  const following = isNearScrollBottom(elements.liveTranscript);
  if (following !== state.liveFollow) setLiveFollow(following, false);
}

function setRecordingUi(recording, started) {
  const wasRecording = state.recording;
  state.recording = recording;
  elements.recordingBanner.hidden = !recording;
  elements.recordButton.classList.toggle("recording", recording);
  elements.recordButtonLabel.textContent = recording ? "Stop recording" : "New recording";
  elements.emptyRecordButton.disabled = recording;
  if (recording) {
    if (!wasRecording) {
      state.lastLiveStatus = null;
      state.lastLiveSignature = null;
      state.liveHasText = false;
      state.livePollErrorLogged = false;
      setLiveFollow(true);
    }
    if (!state.recordingStartedAt) state.recordingStartedAt = Date.now();
    if (!state.recordingTimer) {
      state.recordingTimer = window.setInterval(updateRecordingTimer, 250);
    }
    const liveEnabled = Boolean(started && started.live_started);
    state.liveEnabledForRecording = liveEnabled;
    elements.liveStatus.textContent = liveEnabled ? "Connecting…" : "Live captions disabled";
    elements.liveTranscript.textContent = liveEnabled
      ? "Listening for speech…"
      : "Live captions are disabled for this recording. Audio is still being captured for the final transcript.";
    if (started && started.device_name) {
      state.recordingSource =
        "Recording from " + started.device_name + " at " + started.sample_rate + " Hz";
    }
    if (liveEnabled) startLivePolling();
  } else {
    if (state.recordingTimer) window.clearInterval(state.recordingTimer);
    state.recordingTimer = null;
    state.recordingStartedAt = null;
    state.recordingSource = null;
    state.liveEnabledForRecording = false;
    stopLivePolling();
    elements.recordingTimer.textContent = "00:00";
    elements.levelBar.style.width = "2%";
  }
  renderSpeakers();
  renderVoiceLibrary();
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

function handleLiveTranscript(payload) {
  if (!payload) return;
  const status = String(payload.status || "Live");
  const text = String(payload.text || "").trim();
  const error = payload.error ? String(payload.error) : "";
  const signature = JSON.stringify([status, text, Boolean(payload.finished), error]);
  if (signature === state.lastLiveSignature) return;
  state.lastLiveSignature = signature;

  elements.liveStatus.textContent = status;
  if (status !== state.lastLiveStatus) {
    addActivity("Live captions: " + status, error ? "error" : "");
    state.lastLiveStatus = status;
  }
  if (text) {
    const previousScrollTop = elements.liveTranscript.scrollTop;
    elements.liveTranscript.textContent = text;
    if (state.liveFollow) {
      scrollLiveToLatest();
    } else {
      elements.liveTranscript.scrollTop = previousScrollTop;
    }
    if (!state.liveHasText) {
      state.liveHasText = true;
      addActivity("Live captions are receiving speech", "success");
    }
  } else if (status === "Live captions connected" || status === "Live") {
    elements.liveTranscript.textContent = "Listening for speech…";
    if (state.liveFollow) scrollLiveToLatest();
  }
  if (error) {
    addActivity("Live transcription error: " + error, "error");
    showToast(error, "error");
  }
  if (payload.finished && !state.liveHasText && state.liveEnabledForRecording) {
    addActivity("Live captions finished without receiving speech", "error");
  }
  updateContentVisibility();
}

async function pollLiveTranscript() {
  if (!state.recording || !state.liveEnabledForRecording) return;
  try {
    const payload = await invoke("get_live_transcription");
    state.livePollErrorLogged = false;
    handleLiveTranscript(payload);
  } catch (error) {
    if (!state.livePollErrorLogged) {
      state.livePollErrorLogged = true;
      addActivity("Live-caption status check failed: " + errorText(error), "error");
    }
  }
}

function startLivePolling() {
  if (state.livePollTimer) return;
  void pollLiveTranscript();
  state.livePollTimer = window.setInterval(pollLiveTranscript, 400);
}

function stopLivePolling() {
  if (!state.livePollTimer) return;
  window.clearInterval(state.livePollTimer);
  state.livePollTimer = null;
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
  state.queueingProcessing = false;
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
        (started.live_started ? "; live STT captions enabled" : ""),
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
  state.queueingProcessing = true;
  setProcessingDetail("Finalizing the recording…");
  addActivity("Stopping recording");
  try {
    const path = await invoke("stop_recording");
    setRecordingUi(false);
    addActivity("Recording stopped; queueing final transcription");
    const runId = await invoke("transcribe_file_async", { path });
    trackRun(runId);
    state.queueingProcessing = false;
    addActivity("[" + runId.slice(0, 8) + "] Final transcription queued");
    setProcessingDetail("Uploading the recording to the STT provider…");
  } catch (error) {
    state.queueingProcessing = false;
    const message = errorText(error);
    addActivity("Could not stop or queue recording: " + message, "error");
    showToast(message, "error");
    renderSpeakers();
    updateContentVisibility();
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
  renderSpeakers();
  renderVoiceLibrary();
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
  renderSpeakers();
  renderVoiceLibrary();
  updateContentVisibility();
}

function stageDescription(stage, detail) {
  const descriptions = {
    queued: "Queued for final transcription",
    "transcription:start": "Preparing final transcription",
    "stt:upload:start": "Uploading recording to the STT provider",
    "stt:upload:done": "Upload finished",
    "stt:transcription:start": "Starting final STT transcription",
    "stt:transcription:waiting": "Waiting for the STT provider",
    "stt:transcription:status": "STT provider processing",
    "stt:transcript:download:start": "Downloading diarized transcript",
    "stt:transcript:download:done": "Final transcript received",
    "stt:cleanup:start": "Removing temporary provider artifacts",
    "stt:cleanup:done": "Provider artifacts removed",
    "audio:read:start": "Preparing audio for local speaker identification",
    "audio:read:done": "Local audio prepared",
    "voiceprints:start": "Extracting and matching voiceprints locally",
    "voiceprint:new": "New voice profile created",
    "voiceprint:matched": "Known voice identified",
    "voiceprint:skipped": "Voiceprint skipped; transcript left unattributed",
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
    const sessionId = event.detail;
    setProcessingDetail("Opening the final transcript…");
    await Promise.all([loadSpeakers(), loadSessions(sessionId)]);
    if (sessionId) await selectSession(sessionId);
    finishRun(runId);
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
    renderSpeakers();
    updateContentVisibility();
  } catch (error) {
    addActivity("Could not load conversations: " + errorText(error), "error");
  }
}

function renderSessions() {
  const query = elements.conversationSearch.value.trim().toLowerCase();
  const filtered = filterSessions(state.sessions, query, state.voiceFilteredSessionIds);
  elements.sessionsList.replaceChildren();
  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "sidebar-empty";
    empty.textContent =
      query || state.voiceFilteredSessionIds
        ? "No matching conversations."
        : "No conversations yet.";
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
  state.selectedSegments = [];
  state.recapState = null;
  state.activeRecapTab = "transcript";
  renderSpeakers();
  elements.segmentsList.replaceChildren();
  elements.legacyTranscript.hidden = false;
  elements.legacyTranscript.textContent = "Loading conversation…";
  renderSessions();
  elements.saveState.textContent = "Loading…";
  updateContentVisibility();
  try {
    const [segmentsResult, recapResult] = await Promise.allSettled([
      invoke("list_segments", { sessionId }),
      invoke("get_recap_state", { sessionId }),
    ]);
    if (sequence !== state.sessionLoadSequence) return;
    if (segmentsResult.status === "rejected") throw segmentsResult.reason;
    state.selectedSegments = segmentsResult.value;
    if (recapResult.status === "fulfilled") {
      state.recapState = recapResult.value;
    } else {
      addActivity("Could not load recap data: " + errorText(recapResult.reason), "error");
    }
    renderRecapShell();
    renderTranscript(session);
    renderSpeakers();
    elements.saveState.textContent = "Saved locally";
  } catch (error) {
    addActivity("Could not load conversation: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
  updateContentVisibility();
}

async function refreshRecapState({ rerenderTranscript = true } = {}) {
  const sessionId = state.selectedSessionId;
  if (!sessionId) {
    state.recapState = null;
    renderRecapShell();
    return;
  }
  const recapState = await invoke("get_recap_state", { sessionId });
  if (sessionId !== state.selectedSessionId) return;
  state.recapState = recapState;
  renderRecapShell();
  if (rerenderTranscript) {
    const session = state.sessions.find((candidate) => candidate.id === sessionId);
    renderTranscript(session);
  }
}

function renderRecapShell() {
  const availableTabs = new Set(recapTabAvailability(state.recapState));
  const hasRecap = availableTabs.has("executive");
  const hasAgendaCoverage = availableTabs.has("agenda");
  elements.executiveTab.hidden = !availableTabs.has("executive");
  elements.fullSummaryTab.hidden = !availableTabs.has("full");
  elements.actionsTab.hidden = !availableTabs.has("actions");
  elements.agendaCoverageTab.hidden = !availableTabs.has("agenda");
  elements.recapStaleBanner.hidden = !(
    hasRecap && state.recapState && state.recapState.stale
  );
  if (!hasRecap && state.activeRecapTab !== "transcript") {
    state.activeRecapTab = "transcript";
  }
  if (state.activeRecapTab === "agenda" && !hasAgendaCoverage) {
    state.activeRecapTab = "transcript";
  }
  selectRecapTab(state.activeRecapTab);
  updateContentVisibility();
}

function selectRecapTab(tab) {
  const recapRecord = state.recapState && state.recapState.recap;
  if (tab !== "transcript" && !recapRecord) tab = "transcript";
  if (tab === "agenda" && !recapRecord?.payload?.agenda_present) tab = "transcript";
  state.activeRecapTab = tab;
  for (const button of elements.recapTabs.querySelectorAll("[data-recap-tab]")) {
    const selected = button.dataset.recapTab === tab;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  }
  elements.transcriptTabPanel.hidden = tab !== "transcript";
  elements.generatedTabPanel.hidden = tab === "transcript";
  if (tab === "transcript") {
    scheduleTranscriptResize();
  } else {
    renderGeneratedTab();
  }
}

function localized(value) {
  if (!value) return "";
  return String(value[state.generatedLanguage] || value.english || value.original || "");
}

function evidenceLabel(ids) {
  const labels = (ids || []).map((id) => {
    const segment = state.selectedSegments.find((candidate) => candidate.id === id);
    return segment ? formatTimestamp(segment.start_ms) : id;
  });
  return labels.length ? "Evidence: " + labels.join(", ") : "";
}

function appendGeneratedEvidence(container, ids) {
  const label = evidenceLabel(ids);
  if (!label) return;
  const evidence = document.createElement("span");
  evidence.className = "generated-evidence";
  evidence.textContent = label;
  container.append(evidence);
}

function appendEmptyGeneratedState(copy) {
  const empty = document.createElement("p");
  empty.className = "supporting-copy";
  empty.textContent = copy;
  elements.generatedContent.append(empty);
}

function renderGeneratedTab() {
  const payload = state.recapState?.recap?.payload;
  elements.generatedContent.replaceChildren();
  if (!payload) return;
  const tab = state.activeRecapTab;
  const labels = {
    executive: ["Recap", "Executive summary"],
    full: ["Recap", "Full summary"],
    actions: ["Attribution", "Actions"],
    agenda: ["Meeting agenda", "Agenda coverage"],
  };
  const [eyebrow, title] = labels[tab] || labels.executive;
  elements.generatedEyebrow.textContent = eyebrow;
  elements.generatedTitle.textContent = title;
  elements.showOriginalButton.classList.toggle(
    "selected",
    state.generatedLanguage === "original",
  );
  elements.showEnglishButton.classList.toggle(
    "selected",
    state.generatedLanguage === "english",
  );

  if (tab === "executive") {
    const paragraph = document.createElement("p");
    paragraph.textContent = localized(payload.executive_summary);
    elements.generatedContent.append(paragraph);
    return;
  }
  if (tab === "full") {
    for (const section of payload.full_summary || []) {
      const heading = document.createElement("h3");
      heading.textContent = localized(section.heading);
      const body = document.createElement("p");
      body.textContent = localized(section.body);
      elements.generatedContent.append(heading, body);
    }
    return;
  }
  if (tab === "actions") {
    renderActionGroup("Commitments", payload.commitments || []);
    renderActionGroup("Actions already taken", payload.actions_already_taken || []);
    return;
  }
  if (tab === "agenda") {
    if (!(payload.agenda_coverage || []).length) {
      appendEmptyGeneratedState("No agenda coverage was returned.");
      return;
    }
    const list = document.createElement("ol");
    list.className = "generated-list";
    for (const item of payload.agenda_coverage) {
      const row = document.createElement("li");
      row.className = "generated-item";
      const status = document.createElement("span");
      status.className = "agenda-status " + item.status;
      status.textContent = String(item.status || "").replace("-", " ");
      const heading = document.createElement("strong");
      heading.textContent = localized(item.agenda_item);
      const statement = document.createElement("p");
      statement.textContent = localized(item.statement);
      row.append(status, heading, statement);
      appendGeneratedEvidence(row, item.evidence_segment_ids);
      list.append(row);
    }
    elements.generatedContent.append(list);
  }
}

function renderActionGroup(title, items) {
  const heading = document.createElement("h3");
  heading.textContent = title;
  elements.generatedContent.append(heading);
  if (!items.length) {
    appendEmptyGeneratedState("None identified in the transcript.");
    return;
  }
  const list = document.createElement("ul");
  list.className = "generated-list";
  for (const item of items) {
    const row = document.createElement("li");
    row.className = "generated-item";
    const participant = document.createElement("strong");
    participant.textContent = item.participant;
    const statement = document.createElement("p");
    statement.textContent = localized(item.statement);
    row.append(participant, statement);
    const metaParts = [];
    const timing = localized(item.stated_timing).trim();
    const uncertainty = localized(item.uncertainty).trim();
    if (timing) metaParts.push("Timing: " + timing);
    if (uncertainty) metaParts.push("Uncertainty: " + uncertainty);
    if (metaParts.length) {
      const meta = document.createElement("span");
      meta.className = "item-meta";
      meta.textContent = metaParts.join(" · ");
      row.append(meta);
    }
    list.append(row);
  }
  elements.generatedContent.append(list);
}

function renderTranscript(session) {
  elements.segmentsList.replaceChildren();
  if (!state.selectedSegments.length) {
    elements.legacyTranscript.hidden = false;
    const legacyText = (session && session.transcript) || "This conversation has no transcript.";
    const legacyId = session ? "legacy-" + session.id : "";
    elements.legacyTranscript.textContent = translatedSegmentText(
      legacyText,
      legacyId ? segmentTranslations(legacyId) : [],
    );
    return;
  }
  elements.legacyTranscript.hidden = true;
  for (const segment of state.selectedSegments) {
    const row = document.createElement("article");
    row.className = "segment";
    row.dataset.segmentId = segment.id;
    const speakerColumn = document.createElement("div");
    speakerColumn.className = "segment-speaker";
    const select = buildSpeakerSelect(segment.speaker_id, segment.speaker_label);
    select.setAttribute("aria-label", "Speaker for this intervention");
    select.addEventListener("change", async () => {
      await assignSegmentSpeaker(segment, select.value || null);
    });
    const time = document.createElement("time");
    time.textContent = formatTimestamp(segment.start_ms);
    speakerColumn.append(time, select);

    const body = document.createElement("div");
    body.className = "segment-body";
    const text = document.createElement("textarea");
    text.className = "segment-text";
    text.value = segment.text || "";
    text.setAttribute("aria-label", "Transcript intervention");
    text.addEventListener("input", () => {
      autoResize(text);
      elements.saveState.textContent = "Unsaved changes";
    });
    text.addEventListener("blur", async () => {
      const value = text.value.trim();
      if (value === (segment.text || "").trim()) {
        elements.saveState.textContent = "Saved locally";
        if (segmentTranslations(segment.id).length) renderTranscript(session);
        return;
      }
      await saveSegmentText(segment, value);
    });
    text.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") text.blur();
    });
    const translations = segmentTranslations(segment.id);
    if (translations.length) {
      const plan = buildTranslationPlan(segment.text, translations);
      const rich = document.createElement("div");
      rich.className = "segment-rich-text";
      for (const chunk of plan.chunks) {
        rich.append(document.createTextNode(chunk.source));
        if (chunk.translation) {
          const translation = document.createElement("span");
          translation.className = "translation-inline";
          translation.textContent = "(TRANSLATION: " + chunk.translation + ")";
          rich.append(document.createTextNode(" "), translation);
        }
      }
      body.append(rich);
      if (plan.fallbacks.length) {
        const fallbacks = document.createElement("div");
        fallbacks.className = "translation-fallbacks";
        for (const fallback of plan.fallbacks) {
          const translation = document.createElement("p");
          translation.className = "translation-fallback";
          translation.textContent = "(TRANSLATION: " + fallback.english_translation + ")";
          fallbacks.append(translation);
          const warningKey =
            state.recapState.recap.generated_at + ":" + segment.id + ":" + fallback.source_excerpt;
          if (!state.translationWarnings.has(warningKey)) {
            state.translationWarnings.add(warningKey);
            addActivity(
              "A translation for the intervention at " +
                formatTimestamp(segment.start_ms) +
                " could not be anchored exactly and is shown beneath it",
            );
          }
        }
        body.append(fallbacks);
      }
      const edit = document.createElement("button");
      edit.type = "button";
      edit.className = "text-button segment-edit-button";
      edit.textContent = "Edit transcript";
      text.hidden = true;
      edit.addEventListener("click", () => {
        rich.hidden = true;
        const fallbacks = body.querySelector(".translation-fallbacks");
        if (fallbacks) fallbacks.hidden = true;
        edit.hidden = true;
        text.hidden = false;
        autoResize(text);
        text.focus();
      });
      body.append(edit);
    }
    body.append(text);
    row.append(speakerColumn, body);
    elements.segmentsList.append(row);
    autoResize(text);
  }
  scheduleTranscriptResize();
}

function segmentTranslations(segmentId) {
  if (!state.recapState?.recap || state.recapState.stale) return [];
  return (state.recapState.recap.payload.translations || []).filter(
    (translation) => translation.segment_id === segmentId,
  );
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

function scheduleConversationTitleResize() {
  if (state.titleResizeFrame) {
    window.cancelAnimationFrame(state.titleResizeFrame);
  }
  state.titleResizeFrame = window.requestAnimationFrame(() => {
    state.titleResizeFrame = null;
    elements.conversationTitle.style.height = "0px";
    elements.conversationTitle.style.height =
      Math.max(34, elements.conversationTitle.scrollHeight) + "px";
  });
}

function scheduleTranscriptResize() {
  if (state.transcriptResizeFrame) {
    window.cancelAnimationFrame(state.transcriptResizeFrame);
  }
  state.transcriptResizeFrame = window.requestAnimationFrame(() => {
    state.transcriptResizeFrame = null;
    for (const textarea of elements.segmentsList.querySelectorAll(".segment-text")) {
      autoResize(textarea);
    }
  });
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
    await refreshRecapState({ rerenderTranscript: false });
    const session = state.sessions.find((candidate) => candidate.id === segment.session_id);
    renderTranscript(session);
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
    await loadSpeakers();
    await refreshRecapState({ rerenderTranscript: false });
    const session = state.sessions.find((candidate) => candidate.id === segment.session_id);
    renderTranscript(session);
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
  const sessionId = state.selectedSessionId;
  const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
  const confirmed = await requestConfirmation({
    title: "Delete this conversation?",
    message:
      "“" +
      sessionTitle(session) +
      "” and its transcript will be deleted. Named people are kept. Any unnamed VOICE profiles used only by this conversation will also be removed.",
    acceptLabel: "Delete conversation",
  });
  if (!confirmed) return;
  elements.deleteSessionButton.disabled = true;
  addActivity("Deleting conversation “" + sessionTitle(session) + "”…");
  try {
    const removedVoices = await invoke("delete_session", {
      sessionId,
    });
    addActivity(
      "Conversation deleted" +
        (removedVoices
          ? "; removed " +
            removedVoices +
            " orphan unnamed voice" +
            (removedVoices === 1 ? "" : "s")
          : ""),
      "success",
    );
    state.selectedSessionId = null;
    state.selectedSegments = [];
    state.recapState = null;
    state.activeRecapTab = "transcript";
    elements.conversationTitle.value = "New conversation";
    elements.conversationTitle.disabled = true;
    elements.conversationMeta.textContent = "Record a conversation to begin";
    elements.deleteSessionButton.hidden = true;
    await Promise.all([loadSessions(), loadSpeakers()]);
  } catch (error) {
    addActivity("Could not delete conversation: " + errorText(error), "error");
    showToast(errorText(error), "error");
  } finally {
    elements.deleteSessionButton.disabled = false;
  }
}

async function loadSpeakers() {
  try {
    state.speakers = await invoke("list_speakers_with_stats");
    renderConversationSpeakerFilter();
    renderSpeakers();
    renderVoiceLibrary();
    renderSessions();
    if (elements.conversationSpeakerFilter.value) {
      await applyConversationVoiceFilter();
    }
  } catch (error) {
    addActivity("Could not load voice profiles: " + errorText(error), "error");
  }
}

function renderConversationSpeakerFilter() {
  const selectedId = elements.conversationSpeakerFilter.value;
  elements.conversationSpeakerFilter.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "All voices";
  elements.conversationSpeakerFilter.append(all);
  const filterableSpeakers = state.speakers.filter(
    (candidate) => candidate.conversation_count > 0,
  );
  for (const speaker of filterableSpeakers) {
    const option = document.createElement("option");
    option.value = speaker.id;
    option.textContent = speaker.label || "Unnamed voice";
    elements.conversationSpeakerFilter.append(option);
  }
  const selectedStillExists = filterableSpeakers.some((speaker) => speaker.id === selectedId);
  if (selectedId && selectedStillExists) {
    elements.conversationSpeakerFilter.value = selectedId;
  } else if (selectedId) {
    state.voiceFilterSequence += 1;
    state.voiceFilteredSessionIds = null;
  }
}

async function applyConversationVoiceFilter() {
  const speakerId = elements.conversationSpeakerFilter.value;
  const sequence = ++state.voiceFilterSequence;
  if (!speakerId) {
    state.voiceFilteredSessionIds = null;
    renderSessions();
    return;
  }
  try {
    const sessionIds = await invoke("list_session_ids_for_speaker", { speakerId });
    if (sequence !== state.voiceFilterSequence) return;
    state.voiceFilteredSessionIds = new Set(sessionIds);
    renderSessions();
  } catch (error) {
    if (sequence !== state.voiceFilterSequence) return;
    state.voiceFilteredSessionIds = null;
    elements.conversationSpeakerFilter.value = "";
    renderSessions();
    addActivity("Could not filter conversations by voice: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

function speakerInitial(label) {
  const value = (label || "?").trim();
  return value ? value[0].toUpperCase() : "?";
}

function profileDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown date";
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderSpeakers() {
  elements.speakersList.replaceChildren();
  if (state.recording) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent = "Voice profiles will appear after the recording is processed.";
    elements.speakersList.append(empty);
    return;
  }
  if (state.queueingProcessing || state.activeRuns.size > 0) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent = "Detecting and identifying voices for the current recording…";
    elements.speakersList.append(empty);
    return;
  }
  if (!state.selectedSessionId) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent = "Select a conversation to see the voices attributed in it.";
    elements.speakersList.append(empty);
    return;
  }
  const selectedSpeakerIds = new Set(
    state.selectedSegments.map((segment) => segment.speaker_id).filter(Boolean),
  );
  const currentSpeakers = state.speakers.filter((speaker) => selectedSpeakerIds.has(speaker.id));
  if (!currentSpeakers.length) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent = "No manageable voice profiles are attributed in this conversation.";
    elements.speakersList.append(empty);
    return;
  }
  for (const speaker of currentSpeakers) {
    elements.speakersList.append(buildSpeakerCard(speaker, true, false));
  }
}

function renderVoiceLibrary() {
  elements.voiceLibraryList.replaceChildren();
  if (!state.speakers.length) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent =
      "No voice profiles yet. Recall creates them after the first diarized recording.";
    elements.voiceLibraryList.append(empty);
    return;
  }
  const selectedSpeakerIds = new Set(
    state.recording || state.queueingProcessing || state.activeRuns.size > 0
      ? []
      : state.selectedSegments.map((segment) => segment.speaker_id).filter(Boolean),
  );
  for (const speaker of state.speakers) {
    elements.voiceLibraryList.append(
      buildSpeakerCard(speaker, selectedSpeakerIds.has(speaker.id), true),
    );
  }
}

function buildSpeakerCard(speaker, inSelectedConversation, inVoiceLibrary) {
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
    " reference voiceprint" +
    (speaker.embedding_count === 1 ? "" : "s");
  const recency = document.createElement("div");
  recency.className = "speaker-recency";
  recency.textContent = "Last heard " + profileDate(speaker.last_seen_at || speaker.created_at);
  copy.append(name, meta, recency);
  identity.append(avatar, copy);
  header.append(identity);
  card.append(header);
  const tags = document.createElement("div");
  tags.className = "speaker-tags";
  if (inSelectedConversation) {
    const currentTag = document.createElement("span");
    currentTag.className = "current-voice-tag";
    currentTag.textContent = "In selected conversation";
    tags.append(currentTag);
  }
  if (provisional) {
    const tag = document.createElement("span");
    tag.className = "new-voice-tag";
    tag.textContent = "Needs identification";
    tags.append(tag);
    const matchingTag = document.createElement("span");
    matchingTag.className = "legacy-voice-tag";
    matchingTag.textContent = "Not auto-matched";
    tags.append(matchingTag);
  } else if (speaker.embedding_count > 0) {
    const matchingTag = document.createElement("span");
    matchingTag.className = "recognition-voice-tag";
    matchingTag.textContent = "Automatic recognition on";
    tags.append(matchingTag);
  }
  if (!provisional && speaker.conversation_count > 0) {
    const protectedTag = document.createElement("span");
    protectedTag.className = "protected-voice-tag";
    protectedTag.textContent = "History protected";
    tags.append(protectedTag);
  }
  if (speaker.embedding_count === 0) {
    const legacyTag = document.createElement("span");
    legacyTag.className = "legacy-voice-tag";
    legacyTag.textContent = "No current voiceprint";
    tags.append(legacyTag);
  }
  if (tags.childElementCount) card.append(tags);

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
    const nameButton = actionButton(
      "Name person",
      () => openNameDialog(speaker),
      "primary-mini",
    );
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
      actionButton("Rename person", () => openNameDialog(speaker)),
      actionButton("Merge…", () => openAssignDialog(speaker)),
    );
  }
  if (inVoiceLibrary) {
    const deleteButton = actionButton("Delete", () => deleteSpeaker(speaker), "danger-mini");
    if (!provisional && speaker.conversation_count > 0) {
      deleteButton.disabled = true;
      deleteButton.title =
        "This named person is used in conversation history. Reassign or delete those conversations first.";
    }
    actions.append(deleteButton);
  }
  card.append(actions);
  return card;
}

async function openVoiceLibrary() {
  await loadSpeakers();
  if (!elements.voiceLibraryDialog.open) elements.voiceLibraryDialog.showModal();
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
  const provisional = isProvisionalLabel(speaker.label);
  elements.nameSpeakerId.value = speaker.id;
  elements.nameDialogTitle.textContent = provisional ? "Name this person" : "Rename this person";
  elements.saveSpeakerNameButton.textContent = provisional ? "Save name" : "Save new name";
  elements.speakerName.value = provisional ? "" : speaker.label || "";
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
  const provisional = isProvisionalLabel(speaker.label);
  if (!provisional && speaker.conversation_count > 0) {
    showToast(
      label +
        " is used in " +
        speaker.conversation_count +
        " conversation" +
        (speaker.conversation_count === 1 ? "" : "s") +
        " and cannot be deleted until that history is reassigned or removed.",
      "error",
    );
    return;
  }
  const usageWarning = speaker.conversation_count
    ? " It is used in " +
      speaker.conversation_count +
      " conversation" +
      (speaker.conversation_count === 1 ? "" : "s") +
      "; those turns will become Unknown speaker."
    : "";
  const confirmed = await requestConfirmation({
    title: "Delete this voice profile?",
    message:
      label +
      " and its local voiceprints will be deleted." +
      usageWarning,
    acceptLabel: "Delete voice",
  });
  if (!confirmed) return;
  addActivity("Deleting voice profile " + label + "…");
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

function transcriptExport(markdown) {
  const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
  if (!session) return "";
  const lines = [];
  lines.push(markdown ? "# " + sessionTitle(session) : sessionTitle(session));
  const date = sessionDate(session);
  if (date) lines.push(markdown ? "_" + date + "_" : date);
  lines.push("");
  if (!state.selectedSegments.length) {
    lines.push(
      translatedSegmentText(
        session.transcript || "",
        segmentTranslations("legacy-" + session.id),
      ),
    );
    return lines.join("\n").trim();
  }
  for (const segment of state.selectedSegments) {
    const speaker = segment.speaker_label || "Unknown speaker";
    const label = formatTimestamp(segment.start_ms) + " — " + speaker;
    const text = translatedSegmentText(segment.text, segmentTranslations(segment.id));
    if (markdown) {
      lines.push("**" + label + "**", "", text, "");
    } else {
      lines.push(label, text, "");
    }
  }
  return lines.join("\n").trim();
}

function generatedExport(markdown) {
  const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
  const payload = state.recapState?.recap?.payload;
  if (!session || !payload) return "";
  const lines = [markdown ? "# " + sessionTitle(session) : sessionTitle(session), ""];
  const heading = (value, level = 2) =>
    markdown ? "#".repeat(level) + " " + value : value.toUpperCase();
  const evidence = (ids) => evidenceLabel(ids);
  if (state.activeRecapTab === "executive") {
    lines.push(heading("Executive summary"), "", localized(payload.executive_summary));
  } else if (state.activeRecapTab === "full") {
    lines.push(heading("Full summary"), "");
    for (const section of payload.full_summary || []) {
      lines.push(heading(localized(section.heading), 3), "", localized(section.body));
      lines.push("");
    }
  } else if (state.activeRecapTab === "actions") {
    lines.push(heading("Actions"), "");
    appendActionExport(lines, "Commitments", payload.commitments || [], markdown);
    appendActionExport(
      lines,
      "Actions already taken",
      payload.actions_already_taken || [],
      markdown,
    );
  } else if (state.activeRecapTab === "agenda") {
    lines.push(heading("Agenda coverage"), "");
    for (const item of payload.agenda_coverage || []) {
      const prefix = markdown ? "- " : "• ";
      lines.push(
        prefix + localized(item.agenda_item) + " [" + String(item.status).replace("-", " ") + "]",
        "  " + localized(item.statement),
      );
      if (evidence(item.evidence_segment_ids)) {
        lines.push("  " + evidence(item.evidence_segment_ids));
      }
    }
  }
  return lines.join("\n").trim();
}

function appendActionExport(lines, title, items, markdown) {
  lines.push(markdown ? "### " + title : title.toUpperCase());
  if (!items.length) {
    lines.push("None identified in the transcript.", "");
    return;
  }
  for (const item of items) {
    const prefix = markdown ? "- " : "• ";
    lines.push(prefix + item.participant + ": " + localized(item.statement));
    const timing = localized(item.stated_timing).trim();
    const uncertainty = localized(item.uncertainty).trim();
    if (timing) lines.push("  Timing: " + timing);
    if (uncertainty) lines.push("  Uncertainty: " + uncertainty);
  }
  lines.push("");
}

async function writeClipboard(value, label) {
  if (!value) return;
  let fallback = null;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      fallback = document.createElement("textarea");
      fallback.value = value;
      fallback.style.position = "fixed";
      fallback.style.opacity = "0";
      document.body.append(fallback);
      fallback.select();
      if (!document.execCommand("copy")) throw new Error("Clipboard copy was rejected");
    }
    showToast(label + " copied.");
    addActivity(label + " copied to the clipboard", "success");
  } catch (error) {
    addActivity("Could not copy to the clipboard: " + errorText(error), "error");
    showToast("Could not copy to the clipboard.", "error");
  } finally {
    if (fallback) fallback.remove();
  }
}

function renderAgendaDialog() {
  const agenda = state.recapState && state.recapState.agenda;
  elements.removeAgendaButton.hidden = !agenda;
  if (!agenda) {
    elements.agendaCurrent.textContent = "No agenda attached.";
    elements.agendaText.value = "";
    return;
  }
  const size = agenda.size_bytes < 1024
    ? agenda.size_bytes + " bytes"
    : Math.round(agenda.size_bytes / 1024) + " KB";
  elements.agendaCurrent.textContent =
    "Current agenda: " + agenda.filename + " · " + size + " · stored locally";
  elements.agendaText.value = agenda.source_kind === "text" ? agenda.text_content || "" : "";
}

function openAgendaDialog() {
  if (!state.selectedSessionId) return;
  renderAgendaDialog();
  elements.agendaFeedback.textContent = "";
  if (!elements.agendaDialog.open) elements.agendaDialog.showModal();
}

async function saveAgendaText(event) {
  event.preventDefault();
  if (!state.selectedSessionId) return;
  const text = elements.agendaText.value;
  if (!text.trim()) {
    elements.agendaFeedback.textContent = "Paste agenda text, or choose a file.";
    return;
  }
  elements.saveAgendaTextButton.disabled = true;
  elements.agendaFeedback.textContent = "Saving locally…";
  try {
    await invoke("save_agenda_text", { sessionId: state.selectedSessionId, text });
    await refreshRecapState({ rerenderTranscript: false });
    elements.agendaFeedback.textContent = "Saved.";
    elements.agendaDialog.close();
    addActivity("Pasted agenda saved locally", "success");
    showToast("Agenda saved.");
  } catch (error) {
    elements.agendaFeedback.textContent = errorText(error);
    addActivity("Could not save agenda text: " + errorText(error), "error");
  } finally {
    elements.saveAgendaTextButton.disabled = false;
  }
}

async function chooseAgendaFile() {
  if (!state.selectedSessionId) return;
  elements.attachAgendaButton.disabled = true;
  elements.agendaFeedback.textContent = "Waiting for file selection…";
  try {
    const agenda = await invoke("choose_agenda_file", {
      sessionId: state.selectedSessionId,
    });
    if (!agenda) {
      elements.agendaFeedback.textContent = "No file selected.";
      return;
    }
    await refreshRecapState({ rerenderTranscript: false });
    renderAgendaDialog();
    elements.agendaFeedback.textContent = "File stored locally.";
    addActivity("Agenda file stored locally: " + agenda.filename, "success");
    showToast("Agenda file attached.");
  } catch (error) {
    elements.agendaFeedback.textContent = errorText(error);
    addActivity("Could not attach agenda file: " + errorText(error), "error");
  } finally {
    elements.attachAgendaButton.disabled = false;
  }
}

async function removeAgenda() {
  if (!state.selectedSessionId || !state.recapState?.agenda) return;
  const confirmed = await requestConfirmation({
    title: "Remove this agenda?",
    message:
      "The locally stored agenda will be removed. Any existing recap remains visible but becomes out of date until regenerated.",
    acceptLabel: "Remove agenda",
  });
  if (!confirmed) return;
  try {
    await invoke("remove_agenda", { sessionId: state.selectedSessionId });
    await refreshRecapState({ rerenderTranscript: false });
    renderAgendaDialog();
    elements.agendaFeedback.textContent = "Agenda removed.";
    addActivity("Agenda removed from the conversation", "success");
  } catch (error) {
    elements.agendaFeedback.textContent = errorText(error);
    addActivity("Could not remove agenda: " + errorText(error), "error");
  }
}

function updateRecapProgress(stage, detail) {
  const titles = {
    prepare: "Preparing conversation",
    llm: "Waiting for the LLM provider",
    validate: "Checking the recap",
    save: "Saving locally",
    complete: "Recap ready",
    warning: "Finishing the recap",
    error: "Recap failed",
  };
  elements.recapProgressTitle.textContent = titles[stage] || "Creating recap";
  elements.recapProgressDetail.textContent = detail || "Working…";
  const failed = stage === "error";
  elements.recapProgressSpinner.hidden = failed;
  elements.recapProgressClose.hidden = !failed;
}

async function requestRecap() {
  if (!state.selectedSessionId || state.recapRequestSessionId) return;
  if (!state.status?.openai_key_configured || !String(state.preferences?.openai_model || "").trim()) {
    await openSettings();
    elements.settingsFeedback.textContent =
      "Add an OpenAI API key and model before creating a recap.";
    return;
  }
  const unresolved = state.recapState?.unresolved_profiles || [];
  if (unresolved.length) {
    elements.unresolvedList.replaceChildren();
    for (const label of unresolved) {
      const item = document.createElement("li");
      item.textContent = label;
      elements.unresolvedList.append(item);
    }
    elements.unresolvedDialog.showModal();
    return;
  }
  await runRecap(false);
}

function reviewUnresolvedParticipants() {
  elements.unresolvedDialog.close();
  selectRecapTab("transcript");
  const unresolved = new Set(state.recapState?.unresolved_profiles || []);
  const segment = state.selectedSegments.find(
    (candidate) =>
      !candidate.speaker_id || unresolved.has(candidate.speaker_label || "Unknown speaker"),
  );
  const row = segment
    ? Array.from(elements.segmentsList.children).find(
        (candidate) => candidate.dataset.segmentId === segment.id,
      )
    : null;
  if (row) {
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    const select = row.querySelector("select");
    if (select) window.setTimeout(() => select.focus(), 250);
  }
  showToast("Assign or name the highlighted participant, then click Recap again.");
}

async function runRecap(allowUnresolved) {
  const sessionId = state.selectedSessionId;
  if (!sessionId || state.recapRequestSessionId) return;
  state.recapRequestSessionId = sessionId;
  let failed = false;
  if (elements.recapProgressDialog.open) elements.recapProgressDialog.close();
  updateRecapProgress("prepare", "Preparing transcript and agenda…");
  elements.recapProgressDialog.showModal();
  updateContentVisibility();
  addActivity("Starting on-demand LLM recap");
  try {
    const commandState = await invoke("generate_recap", { sessionId, allowUnresolved });
    await loadSessions(sessionId);
    if (state.selectedSessionId === sessionId) {
      const persistedState = await invoke("get_recap_state", { sessionId });
      if (!persistedState?.recap?.payload) {
        throw new Error("The LLM provider finished, but Recall could not load the saved recap.");
      }
      state.recapState = persistedState;
      state.activeRecapTab = "executive";
      const session = state.sessions.find((candidate) => candidate.id === sessionId);
      renderRecapShell();
      renderTranscript(session);
      const visibleTabs = Array.from(
        elements.recapTabs.querySelectorAll("[data-recap-tab]"),
      ).filter((button) => !button.hidden).length;
      addActivity("Recap interface ready with " + visibleTabs + " tabs", "success");
    }
    const usage = commandState.recap;
    addActivity(
      "LLM recap saved locally" +
        (usage
          ? " (" + usage.input_tokens + " input / " + usage.output_tokens + " output tokens)"
          : ""),
      "success",
    );
    showToast("Recap ready.");
  } catch (error) {
    failed = true;
    const message = errorText(error);
    updateRecapProgress("error", message);
    addActivity("LLM recap failed: " + message, "error");
    showToast(message, "error");
    if (state.selectedSessionId === sessionId) {
      try {
        await refreshRecapState({ rerenderTranscript: false });
      } catch {
        // Preserve the currently rendered recap if refreshing also fails.
      }
    }
  } finally {
    state.recapRequestSessionId = null;
    if (!failed && elements.recapProgressDialog.open) elements.recapProgressDialog.close();
    updateContentVisibility();
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
  setOpenAIStatus(status.openai_key_configured);
  if (status.recording && !state.recording) {
    setRecordingUi(true, {
      device_name: preferences.selected_input_device || "selected input",
      sample_rate: 0,
      live_started: preferences.live_transcription,
    });
  }
  elements.languageHints.value = (preferences.language_hints || []).join(", ");
  elements.noTranslationLanguages.value = (
    preferences.no_translation_languages || []
  ).join(", ");
  elements.openaiModel.value = preferences.openai_model || "gpt-5.6-terra";
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
  elements.settingsFeedback.textContent = "Saving locally…";
  try {
    await invoke("save_soniox_key", { apiKey });
    elements.sonioxKey.value = "";
    setServiceStatus(true);
    elements.settingsFeedback.textContent =
      "Soniox key saved. Recall will reuse it without a Keychain prompt.";
    addActivity("Soniox API key saved locally", "success");
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not save Soniox API key: " + message, "error");
  } finally {
    elements.saveKeyButton.disabled = false;
  }
}

async function deleteSonioxKey() {
  const confirmed = await requestConfirmation({
    title: "Remove the Soniox key?",
    message:
      "Recall's locally stored Soniox API key will be removed. A key will be required before the next recording.",
    acceptLabel: "Remove key",
  });
  if (!confirmed) return;
  try {
    await invoke("delete_soniox_key");
    setServiceStatus(false);
    elements.settingsFeedback.textContent = "Key removed.";
    addActivity("Locally stored Soniox API key removed", "success");
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not remove Soniox API key: " + message, "error");
  }
}

async function saveOpenAIKey() {
  const apiKey = elements.openaiKey.value.trim();
  if (!apiKey) {
    elements.settingsFeedback.textContent = "Paste an OpenAI key first.";
    return;
  }
  elements.saveOpenAIKeyButton.disabled = true;
  elements.settingsFeedback.textContent = "Saving locally…";
  try {
    await invoke("save_openai_key", { apiKey });
    elements.openaiKey.value = "";
    setOpenAIStatus(true);
    elements.settingsFeedback.textContent = "OpenAI key saved locally.";
    addActivity("OpenAI API key saved locally", "success");
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not save OpenAI API key: " + message, "error");
  } finally {
    elements.saveOpenAIKeyButton.disabled = false;
  }
}

async function deleteOpenAIKey() {
  const confirmed = await requestConfirmation({
    title: "Remove the OpenAI key?",
    message:
      "Recall's locally stored OpenAI API key will be removed. Existing recaps remain available, but a key is required to generate or regenerate one.",
    acceptLabel: "Remove key",
  });
  if (!confirmed) return;
  try {
    await invoke("delete_openai_key");
    setOpenAIStatus(false);
    elements.settingsFeedback.textContent = "OpenAI key removed.";
    addActivity("Locally stored OpenAI API key removed", "success");
  } catch (error) {
    const message = errorText(error);
    elements.settingsFeedback.textContent = message;
    addActivity("Could not remove OpenAI API key: " + message, "error");
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const selectedInputDevice = elements.inputDevice.value || null;
  const languageHints = parseLanguageHints(elements.languageHints.value);
  const noTranslationLanguages = parseNoTranslationLanguages(
    elements.noTranslationLanguages.value,
  );
  const liveTranscription = elements.liveTranscription.checked;
  const openaiModel = elements.openaiModel.value.trim();
  if (!openaiModel) {
    elements.settingsFeedback.textContent = "LLM model cannot be empty.";
    return;
  }
  elements.settingsFeedback.textContent = "Saving…";
  try {
    await invoke("save_preferences", {
      selectedInputDevice,
      languageHints,
      liveTranscription,
      openaiModel,
      noTranslationLanguages,
    });
    state.preferences = {
      encryption_enabled: state.preferences ? state.preferences.encryption_enabled : false,
      selected_input_device: selectedInputDevice,
      language_hints: languageHints,
      live_transcription: liveTranscription,
      openai_model: openaiModel,
      no_translation_languages: noTranslationLanguages,
      onboarding_version: state.preferences?.onboarding_version || null,
    };
    elements.settingsFeedback.textContent = "Saved.";
    elements.settingsDialog.close();
    addActivity("Recording preferences saved", "success");
    if (state.selectedSessionId) {
      try {
        await refreshRecapState();
      } catch (error) {
        addActivity("Preferences were saved, but recap status could not refresh: " + errorText(error), "error");
      }
    }
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
    scheduleInitialSetupPrompt();
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
  await listen("recap:progress", (event) => {
    const progress = event.payload || {};
    if (!progress.stage) return;
    const kind = progress.stage === "error" ? "error" : progress.stage === "complete" ? "success" : "";
    addActivity("[recap] " + progress.stage + ": " + (progress.detail || "Working…"), kind);
    if (progress.session_id === state.recapRequestSessionId) {
      updateRecapProgress(progress.stage, progress.detail);
    }
  });
  await listen("transcription:progress", (event) => {
    void handleProgressEvent(event.payload);
  });
  await listen("transcription:queued", (event) => {
    const runId = event.payload;
    trackRun(runId);
    state.queueingProcessing = false;
    addActivity("[" + runId.slice(0, 8) + "] Queued from the menu bar");
    setProcessingDetail("Uploading the recording to the STT provider…");
  });
  await listen("recording:started", (event) => {
    state.queueingProcessing = false;
    setRecordingUi(true, event.payload);
  });
  await listen("recording:stopped", () => {
    state.queueingProcessing = true;
    setRecordingUi(false);
    setProcessingDetail("Finalizing the recording…");
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
    handleLiveTranscript(event.payload || {});
  });
}

function bindInterface() {
  elements.recordButton.addEventListener("click", startRecording);
  elements.emptyRecordButton.addEventListener("click", startRecording);
  elements.stopButton.addEventListener("click", stopRecording);
  elements.refreshSessions.addEventListener("click", () => loadSessions());
  elements.refreshSpeakers.addEventListener("click", loadSpeakers);
  elements.conversationSearch.addEventListener("input", renderSessions);
  elements.conversationSpeakerFilter.addEventListener("change", applyConversationVoiceFilter);
  elements.liveTranscript.addEventListener("scroll", handleLiveScroll, { passive: true });
  elements.jumpToLiveButton.addEventListener("click", () => setLiveFollow(true));
  elements.voiceLibraryButton.addEventListener("click", openVoiceLibrary);
  elements.agendaButton.addEventListener("click", openAgendaDialog);
  elements.recapButton.addEventListener("click", requestRecap);
  elements.staleRegenerateButton.addEventListener("click", requestRecap);
  for (const button of elements.recapTabs.querySelectorAll("[data-recap-tab]")) {
    button.addEventListener("click", () => selectRecapTab(button.dataset.recapTab));
  }
  elements.showOriginalButton.addEventListener("click", () => {
    state.generatedLanguage = "original";
    renderGeneratedTab();
  });
  elements.showEnglishButton.addEventListener("click", () => {
    state.generatedLanguage = "english";
    renderGeneratedTab();
  });
  elements.copyTranscriptText.addEventListener("click", () =>
    writeClipboard(transcriptExport(false), "Transcript text"),
  );
  elements.copyTranscriptMarkdown.addEventListener("click", () =>
    writeClipboard(transcriptExport(true), "Transcript Markdown"),
  );
  elements.copyGeneratedText.addEventListener("click", () =>
    writeClipboard(generatedExport(false), "Recap text"),
  );
  elements.copyGeneratedMarkdown.addEventListener("click", () =>
    writeClipboard(generatedExport(true), "Recap Markdown"),
  );
  elements.conversationTitle.addEventListener("change", saveConversationTitle);
  elements.conversationTitle.addEventListener("input", scheduleConversationTitleResize);
  elements.conversationTitle.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      elements.conversationTitle.blur();
    }
  });
  elements.deleteSessionButton.addEventListener("click", deleteSelectedSession);
  elements.settingsButton.addEventListener("click", openSettings);
  elements.gettingStartedButton.addEventListener("click", openOnboarding);
  elements.onboardingExploreButton.addEventListener("click", () => {
    void finishOnboarding(false);
  });
  elements.onboardingSettingsButton.addEventListener("click", () => {
    void finishOnboarding(true);
  });
  elements.onboardingDialog.addEventListener("cancel", (event) => event.preventDefault());
  for (const button of document.querySelectorAll("[data-external-url]")) {
    button.addEventListener("click", () => {
      void openExternalUrl(button.dataset.externalUrl, button);
    });
  }
  elements.saveKeyButton.addEventListener("click", saveSonioxKey);
  elements.deleteKeyButton.addEventListener("click", deleteSonioxKey);
  elements.saveOpenAIKeyButton.addEventListener("click", saveOpenAIKey);
  elements.deleteOpenAIKeyButton.addEventListener("click", deleteOpenAIKey);
  elements.settingsForm.addEventListener("submit", saveSettings);
  elements.agendaForm.addEventListener("submit", saveAgendaText);
  elements.attachAgendaButton.addEventListener("click", chooseAgendaFile);
  elements.removeAgendaButton.addEventListener("click", removeAgenda);
  elements.cancelUnresolvedButton.addEventListener("click", () =>
    elements.unresolvedDialog.close(),
  );
  elements.reviewUnresolvedButton.addEventListener("click", reviewUnresolvedParticipants);
  elements.recapAnywayButton.addEventListener("click", () => {
    elements.unresolvedDialog.close();
    void runRecap(true);
  });
  elements.recapProgressDialog.addEventListener("cancel", (event) => event.preventDefault());
  elements.nameForm.addEventListener("submit", saveSpeakerName);
  elements.assignForm.addEventListener("submit", assignVoiceProfile);
  elements.confirmationForm.addEventListener("submit", (event) => {
    event.preventDefault();
    settleConfirmation(true);
  });
  elements.confirmationCancel.addEventListener("click", () => settleConfirmation(false));
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
  window.addEventListener("resize", scheduleTranscriptResize);
  window.addEventListener("resize", scheduleConversationTitleResize);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !elements.confirmationDialog.hidden) {
      event.preventDefault();
      settleConfirmation(false);
    }
  });
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
    scheduleInitialSetupPrompt();
  } catch (error) {
    const message = errorText(error);
    addActivity("Recall could not initialize: " + message, "error");
    showToast(message, "error");
    setActivityOpen(true);
  }
}

void initialize();

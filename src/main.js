import {
  ONBOARDING_VERSION,
  buildTranslationPlan,
  contentMode,
  filterSessions,
  formatDuration,
  formatTimestamp,
  getCachedConversation,
  groupVoiceFilters,
  indexTranslations,
  invalidateConversationCache,
  isNearScrollBottom,
  isProvisionalLabel,
  isSessionProcessing,
  nextRenderedSegmentCount,
  normalizePreferredLanguage,
  parseLanguageHints,
  parseNoTranslationLanguages,
  processingRunIds,
  recapTabAvailability,
  setCachedConversation,
  shouldShowOnboarding,
  transcriptFromSegments,
  translatedSegmentText,
} from "./ui-helpers.mjs";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const JAMIE_IMPORT_UI_ENABLED = window.__RECALL_ENABLE_JAMIE_IMPORT__ === true;
const CONVERSATION_CACHE_LIMIT = 5;
const SEGMENT_RENDER_BATCH = 100;

const elements = {
  recordButton: document.getElementById("recordButton"),
  recordButtonLabel: document.getElementById("recordButtonLabel"),
  emptyRecordButton: document.getElementById("emptyRecordButton"),
  recordingBanner: document.getElementById("recordingBanner"),
  recordingTimer: document.getElementById("recordingTimer"),
  levelBar: document.getElementById("levelBar"),
  livePanel: document.getElementById("livePanel"),
  liveStatus: document.getElementById("liveStatus"),
  liveTranscript: document.getElementById("liveTranscript"),
  liveTranslationSection: document.getElementById("liveTranslationSection"),
  liveTranslationLabel: document.getElementById("liveTranslationLabel"),
  liveTranslatedTranscript: document.getElementById("liveTranslatedTranscript"),
  liveTranslationWarning: document.getElementById("liveTranslationWarning"),
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
  processingRecoveryBanner: document.getElementById("processingRecoveryBanner"),
  processingRecoveryTitle: document.getElementById("processingRecoveryTitle"),
  processingRecoveryDetail: document.getElementById("processingRecoveryDetail"),
  retryProcessingButton: document.getElementById("retryProcessingButton"),
  discardRetainedAudioButton: document.getElementById("discardRetainedAudioButton"),
  recapStaleBanner: document.getElementById("recapStaleBanner"),
  staleRegenerateButton: document.getElementById("staleRegenerateButton"),
  recapStatusBanner: document.getElementById("recapStatusBanner"),
  recapStatusSpinner: document.getElementById("recapStatusSpinner"),
  recapStatusTitle: document.getElementById("recapStatusTitle"),
  recapStatusDetail: document.getElementById("recapStatusDetail"),
  recapStatusDismiss: document.getElementById("recapStatusDismiss"),
  recapTabs: document.getElementById("recapTabs"),
  transcriptTab: document.getElementById("transcriptTab"),
  executiveTab: document.getElementById("executiveTab"),
  fullSummaryTab: document.getElementById("fullSummaryTab"),
  actionsTab: document.getElementById("actionsTab"),
  agendaCoverageTab: document.getElementById("agendaCoverageTab"),
  importedExecutiveTab: document.getElementById("importedExecutiveTab"),
  importedFullSummaryTab: document.getElementById("importedFullSummaryTab"),
  importedTasksTab: document.getElementById("importedTasksTab"),
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
  loadMoreSegments: document.getElementById("loadMoreSegments"),
  legacyTranscript: document.getElementById("legacyTranscript"),
  saveState: document.getElementById("saveState"),
  speakersList: document.getElementById("speakersList"),
  refreshSpeakers: document.getElementById("refreshSpeakers"),
  voiceLibraryButton: document.getElementById("voiceLibraryButton"),
  peopleVoicesButton: document.getElementById("peopleVoicesButton"),
  identityOperationBadge: document.getElementById("identityOperationBadge"),
  voiceLibraryDialog: document.getElementById("voiceLibraryDialog"),
  voiceLibraryList: document.getElementById("voiceLibraryList"),
  identityProfilesTab: document.getElementById("identityProfilesTab"),
  identityUnassignedTab: document.getElementById("identityUnassignedTab"),
  identitySearch: document.getElementById("identitySearch"),
  identityStatusFilter: document.getElementById("identityStatusFilter"),
  identityRefreshButton: document.getElementById("identityRefreshButton"),
  identityManagerStatus: document.getElementById("identityManagerStatus"),
  identityPreviousPage: document.getElementById("identityPreviousPage"),
  identityPageStatus: document.getElementById("identityPageStatus"),
  identityNextPage: document.getElementById("identityNextPage"),
  identitySelectionSummary: document.getElementById("identitySelectionSummary"),
  identityClearSelection: document.getElementById("identityClearSelection"),
  identityMergeButton: document.getElementById("identityMergeButton"),
  identityMergeDialog: document.getElementById("identityMergeDialog"),
  identityMergeSelection: document.getElementById("identityMergeSelection"),
  identityTarget: document.getElementById("identityTarget"),
  identityFinalLabel: document.getElementById("identityFinalLabel"),
  identityPreviewButton: document.getElementById("identityPreviewButton"),
  identityImpact: document.getElementById("identityImpact"),
  identityImpactStats: document.getElementById("identityImpactStats"),
  identityImpactWarnings: document.getElementById("identityImpactWarnings"),
  identityMergeFeedback: document.getElementById("identityMergeFeedback"),
  identityConfirmButton: document.getElementById("identityConfirmButton"),
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
  preferredLanguage: document.getElementById("preferredLanguage"),
  noTranslationLanguages: document.getElementById("noTranslationLanguages"),
  liveTranscription: document.getElementById("liveTranscription"),
  settingsFeedback: document.getElementById("settingsFeedback"),
  chooseJamieExportButton: document.getElementById("chooseJamieExportButton"),
  resumeJamieImportButton: document.getElementById("resumeJamieImportButton"),
  jamieImportSettingsSection: document.getElementById("jamieImportSettingsSection"),
  jamieImportHistory: document.getElementById("jamieImportHistory"),
  jamieImportDialog: document.getElementById("jamieImportDialog"),
  jamieImportLoading: document.getElementById("jamieImportLoading"),
  jamieImportError: document.getElementById("jamieImportError"),
  jamieImportErrorMessage: document.getElementById("jamieImportErrorMessage"),
  jamieImportReview: document.getElementById("jamieImportReview"),
  jamieImportSource: document.getElementById("jamieImportSource"),
  jamieImportSummary: document.getElementById("jamieImportSummary"),
  jamieImportStats: document.getElementById("jamieImportStats"),
  jamieValidationPanel: document.getElementById("jamieValidationPanel"),
  jamieUseSourceNamesButton: document.getElementById("jamieUseSourceNamesButton"),
  jamieIdentitySearch: document.getElementById("jamieIdentitySearch"),
  jamieIdentityNeedsReview: document.getElementById("jamieIdentityNeedsReview"),
  jamieGenericIdentityNote: document.getElementById("jamieGenericIdentityNote"),
  jamieIdentityList: document.getElementById("jamieIdentityList"),
  jamieExcludeInvalidButton: document.getElementById("jamieExcludeInvalidButton"),
  jamieMeetingSearch: document.getElementById("jamieMeetingSearch"),
  jamieMeetingIssuesOnly: document.getElementById("jamieMeetingIssuesOnly"),
  jamieMeetingList: document.getElementById("jamieMeetingList"),
  jamieImportFeedback: document.getElementById("jamieImportFeedback"),
  jamieImportButton: document.getElementById("jamieImportButton"),
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
  nameDialogHelp: document.getElementById("nameDialogHelp"),
  speakerName: document.getElementById("speakerName"),
  saveSpeakerNameButton: document.getElementById("saveSpeakerNameButton"),
  assignDialog: document.getElementById("assignDialog"),
  assignForm: document.getElementById("assignForm"),
  assignSourceId: document.getElementById("assignSourceId"),
  assignTarget: document.getElementById("assignTarget"),
  speakerPickerDialog: document.getElementById("speakerPickerDialog"),
  speakerPickerSearch: document.getElementById("speakerPickerSearch"),
  speakerPickerResults: document.getElementById("speakerPickerResults"),
  speakerPickerUnknown: document.getElementById("speakerPickerUnknown"),
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
  liveWorkspaceSelected: false,
  navigationRevision: 0,
  openQueuedDraft: false,
  openQueuedDraftRevision: null,
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
  lastLiveTranslationWarning: null,
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
  importedArtifact: null,
  conversationCache: new Map(),
  translationIndex: new Map(),
  renderedSegmentCount: 0,
  speakerPickerSegmentId: null,
  conversationSearchIds: null,
  conversationSearchSequence: 0,
  conversationSearchTimer: null,
  activeRecapTab: "transcript",
  generatedLanguage: "original",
  recapJobs: new Map(),
  translationWarnings: new Set(),
  onboardingAcknowledgedThisLaunch: false,
  translationLanguages: [],
  jamieImportPreview: null,
  jamieImportSaveTimer: null,
  jamieImportRevision: 0,
  jamieImportRunning: false,
  importBatches: [],
  identityManagerView: "profiles",
  identitySearch: "",
  identityStatus: "all",
  identityPage: 1,
  identityPageSize: 100,
  identityPageData: null,
  identityLoadSequence: 0,
  identitySearchTimer: null,
  selectedIdentityProfiles: new Map(),
  selectedUnassignedGroups: new Map(),
  identityPreview: null,
  identityPreviewSignature: null,
  identityOperationRunning: false,
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

function serializableCopy(value) {
  return JSON.parse(JSON.stringify(value));
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

function recapProgressTitle(stage) {
  const titles = {
    prepare: "Preparing conversation",
    llm: "Waiting for the LLM provider",
    "llm:start": "Starting the recap",
    "llm:done": "LLM work complete",
    "analysis:start": "Analysing the meeting",
    "analysis:done": "Meeting analysis complete",
    "translations:start": "Preparing translations",
    "translations:batch:start": "Translating the transcript",
    "translations:batch:done": "Translation batch complete",
    "translations:done": "Translations complete",
    validate: "Checking the recap",
    save: "Saving locally",
    complete: "Recap ready",
    warning: "Finishing the recap",
    error: "Recap failed",
  };
  return titles[stage] || "Creating recap";
}

function recapJob(sessionId) {
  return sessionId ? state.recapJobs.get(sessionId) || null : null;
}

function recapIsRunning(sessionId) {
  const job = recapJob(sessionId);
  if (job?.status === "running") return true;
  return Boolean(
    sessionId === state.selectedSessionId &&
      state.recapState &&
      state.recapState.in_flight,
  );
}

function rememberNativeRecapState(sessionId, recapState) {
  if (!sessionId || !recapState?.in_flight || recapJob(sessionId)) return;
  state.recapJobs.set(sessionId, {
    status: "running",
    stage: "prepare",
    detail: "This conversation is being processed in the background. You can use the rest of Recall.",
  });
}

function renderSelectedRecapStatus() {
  const sessionId = state.selectedSessionId;
  const job = recapJob(sessionId);
  const nativeInFlight = Boolean(state.recapState && state.recapState.in_flight);
  const visible = Boolean(job || nativeInFlight);
  elements.recapStatusBanner.hidden = !visible;
  elements.recapStatusBanner.classList.remove("failed");
  if (!visible) return;

  const failed = job?.status === "error";
  const stage = job?.stage || "prepare";
  elements.recapStatusBanner.classList.toggle("failed", failed);
  elements.recapStatusSpinner.hidden = failed;
  elements.recapStatusDismiss.hidden = !failed;
  elements.recapStatusTitle.textContent = recapProgressTitle(failed ? "error" : stage);
  elements.recapStatusDetail.textContent =
    job?.detail ||
    (nativeInFlight
      ? "This conversation is being processed in the background. You can use the rest of Recall."
      : "Working…");
}

function renderProcessingRecovery(session) {
  const status = session && session.processing_status;
  elements.processingRecoveryBanner.hidden = !status;
  elements.processingRecoveryBanner.classList.remove("failed", "cleanup");
  elements.retryProcessingButton.hidden = true;
  elements.discardRetainedAudioButton.hidden = true;
  if (!status) return;

  if (status === "queued" || status === "processing") {
    elements.processingRecoveryTitle.textContent = "Final transcript is processing";
    elements.processingRecoveryDetail.textContent =
      "The recording and live-caption draft are saved locally. You can record another meeting while this continues.";
    return;
  }

  if (status === "failed") {
    elements.processingRecoveryBanner.classList.add("failed");
    elements.processingRecoveryTitle.textContent = "Final transcription needs a retry";
    const error = String(session.processing_error || "Final STT processing did not finish.");
    elements.processingRecoveryDetail.textContent = session.recoverable_audio
      ? error + " The recording and live-caption draft are still saved locally."
      : error + " The live-caption draft is saved, but the retained recording could not be found.";
    elements.retryProcessingButton.hidden = !session.recoverable_audio;
    return;
  }

  if (status === "finalized" || status === "cleanup_failed") {
    elements.processingRecoveryBanner.classList.add("cleanup");
    elements.processingRecoveryTitle.textContent = "Final transcript saved";
    elements.processingRecoveryDetail.textContent = session.recoverable_audio
      ? String(session.processing_error || "Recall could not remove the retained recording after processing.")
      : "The final transcript is complete. Finish cleanup to clear the stale recovery record.";
    elements.discardRetainedAudioButton.textContent = session.recoverable_audio
      ? "Remove retained audio"
      : "Finish cleanup";
    elements.discardRetainedAudioButton.hidden = false;
  }
}

function updateContentVisibility() {
  const mode = contentMode({
    recording: state.recording,
    recordingViewSelected: state.liveWorkspaceSelected,
    queueing: state.queueingProcessing,
    processingCount: state.activeRuns.size,
    selectedSessionId: state.selectedSessionId,
  });
  elements.livePanel.hidden = mode !== "recording";
  elements.processingState.hidden = mode !== "processing";
  elements.transcriptContent.hidden = mode !== "conversation";
  elements.emptyState.hidden = mode !== "empty";
  document.body.classList.toggle("recording-active", mode === "recording");
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
    renderSelectedRecapStatus();
  } else if (mode === "processing") {
    elements.conversationTitle.disabled = true;
    elements.conversationTitle.value =
      state.activeRuns.size > 1 ? "Processing recordings" : "Processing recording";
    elements.conversationMeta.textContent = state.processingDetail;
    elements.deleteSessionButton.hidden = true;
    elements.agendaButton.hidden = true;
    elements.recapButton.hidden = true;
    renderSelectedRecapStatus();
  } else if (mode === "conversation") {
    const session = state.sessions.find((candidate) => candidate.id === state.selectedSessionId);
    if (session) {
      const processingStatus = session.processing_status;
      const finalTranscriptPending = isSessionProcessing(session) || processingStatus === "failed";
      const recapInFlight = recapIsRunning(session.id);
      const conversationLocked = isSessionProcessing(session) || recapInFlight;
      elements.conversationTitle.disabled = conversationLocked;
      elements.conversationTitle.value = sessionTitle(session);
      const parts = [sessionDate(session)];
      if (session.duration_ms > 0) parts.push(formatDuration(session.duration_ms));
      elements.conversationMeta.textContent = parts.filter(Boolean).join(" · ");
      elements.deleteSessionButton.hidden = isSessionProcessing(session);
      elements.deleteSessionButton.disabled = recapInFlight;
      elements.agendaButton.hidden = finalTranscriptPending;
      elements.recapButton.hidden = finalTranscriptPending;
      const hasRecap = Boolean(state.recapState && state.recapState.recap);
      elements.agendaButton.textContent =
        state.recapState && state.recapState.agenda ? "Edit agenda" : "Add agenda";
      elements.recapButton.textContent = recapInFlight
        ? "Recapping…"
        : hasRecap
          ? "Regenerate recap"
          : "Recap";
      elements.recapButton.disabled = recapInFlight;
      elements.agendaButton.disabled = recapInFlight;
      elements.staleRegenerateButton.disabled = recapInFlight;
      renderProcessingRecovery(session);
      renderSelectedRecapStatus();
    }
  } else {
    elements.conversationTitle.disabled = true;
    elements.conversationTitle.value = "New conversation";
    elements.conversationMeta.textContent = "Record a conversation to begin";
    elements.deleteSessionButton.hidden = true;
    elements.agendaButton.hidden = true;
    elements.recapButton.hidden = true;
    renderProcessingRecovery(null);
    renderSelectedRecapStatus();
  }
  scheduleConversationTitleResize();
}

function setProcessingDetail(detail) {
  state.processingDetail = detail || "Working…";
  updateContentVisibility();
}

function translationLanguageName(code) {
  const normalized = normalizePreferredLanguage(code);
  const language = state.translationLanguages.find(
    (candidate) => candidate.code === normalized,
  );
  return language ? language.name : normalized.toUpperCase();
}

function scrollLiveToLatest() {
  window.requestAnimationFrame(() => {
    elements.liveTranscript.scrollTop = elements.liveTranscript.scrollHeight;
    elements.liveTranslatedTranscript.scrollTop = elements.liveTranslatedTranscript.scrollHeight;
  });
}

function setLiveFollow(following, scroll = true) {
  state.liveFollow = Boolean(following);
  elements.jumpToLiveButton.hidden = state.liveFollow;
  if (state.liveFollow && scroll) scrollLiveToLatest();
}

function handleLiveScroll(event) {
  const following = isNearScrollBottom(event?.currentTarget || elements.liveTranscript);
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
      state.navigationRevision += 1;
      state.liveWorkspaceSelected = true;
      state.openQueuedDraft = false;
      state.openQueuedDraftRevision = null;
      state.lastLiveStatus = null;
      state.lastLiveSignature = null;
      state.lastLiveTranslationWarning = null;
      state.liveHasText = false;
      state.livePollErrorLogged = false;
      setLiveFollow(true);
      elements.liveTranslationSection.hidden = true;
      elements.liveTranslatedTranscript.textContent = "";
      elements.liveTranslationWarning.hidden = true;
      elements.liveTranslationWarning.textContent = "";
      stopVoicePreview();
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
    if (wasRecording && state.liveWorkspaceSelected) {
      state.openQueuedDraft = true;
      state.openQueuedDraftRevision = state.navigationRevision;
      state.liveWorkspaceSelected = false;
      state.selectedSessionId = null;
      state.selectedSegments = [];
      state.recapState = null;
    }
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
  renderSessions();
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
  const sidebarTimer = elements.sessionsList.querySelector("[data-current-recording-meta]");
  if (sidebarTimer) sidebarTimer.textContent = "Recording · " + elements.recordingTimer.textContent;
}

function handleLiveTranscript(payload) {
  if (!payload) return;
  const status = String(payload.status || "Live");
  const text = String(payload.text || "").trim();
  const translatedText = String(payload.translated_text || "").trim();
  const targetLanguage = String(
    payload.target_language || state.preferences?.preferred_language || "en",
  );
  const translationWarning = String(payload.translation_warning || "").trim();
  const error = payload.error ? String(payload.error) : "";
  const signature = JSON.stringify([
    status,
    text,
    translatedText,
    targetLanguage,
    translationWarning,
    Boolean(payload.finished),
    error,
  ]);
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
  if (translatedText) {
    const previousTranslationScrollTop = elements.liveTranslatedTranscript.scrollTop;
    elements.liveTranslationLabel.textContent =
      "Translation · " + translationLanguageName(targetLanguage);
    elements.liveTranslatedTranscript.textContent = translatedText;
    elements.liveTranslationSection.hidden = false;
    if (state.liveFollow) {
      scrollLiveToLatest();
    } else {
      elements.liveTranslatedTranscript.scrollTop = previousTranslationScrollTop;
    }
  }
  elements.liveTranslationWarning.hidden = !translationWarning;
  elements.liveTranslationWarning.textContent = translationWarning;
  if (
    translationWarning &&
    translationWarning !== state.lastLiveTranslationWarning
  ) {
    state.lastLiveTranslationWarning = translationWarning;
    addActivity("Live translation: " + translationWarning, "error");
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
  stopVoicePreview();
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
  state.queueingProcessing = true;
  setProcessingDetail("Finalizing the recording…");
  addActivity("Stopping recording");
  try {
    const path = await invoke("stop_recording");
    setRecordingUi(false);
    elements.recordButton.disabled = false;
    addActivity("Recording stopped; queueing final transcription");
    const queued = await invoke("transcribe_file_async", { path });
    const runId = queued.run_id;
    const sessionId = queued.session_id;
    state.queueingProcessing = false;
    addActivity("[" + runId.slice(0, 8) + "] Final transcription queued");
    setProcessingDetail("Uploading the retained recording to the STT provider…");
    const shouldOpenDraft =
      state.openQueuedDraft &&
      state.openQueuedDraftRevision === state.navigationRevision &&
      !state.recording;
    await loadSessions();
    const stored = state.sessions.find((session) => session.id === sessionId);
    if (stored && ["queued", "processing"].includes(stored.processing_status)) {
      trackRun(runId);
    } else {
      finishRun(runId);
    }
    if (
      sessionId &&
      shouldOpenDraft &&
      state.openQueuedDraft &&
      state.openQueuedDraftRevision === state.navigationRevision &&
      !state.recording
    ) {
      state.openQueuedDraft = false;
      state.openQueuedDraftRevision = null;
      await selectSession(sessionId, { userInitiated: false });
    }
  } catch (error) {
    state.queueingProcessing = false;
    const message = errorText(error);
    addActivity("Could not stop or queue recording: " + message, "error");
    showToast(message, "error");
    renderSpeakers();
    updateContentVisibility();
  } finally {
    elements.recordButton.disabled = false;
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

function reconcileTrackedRuns(sessions) {
  const persistedRunIds = processingRunIds(sessions);
  for (const runId of Array.from(state.activeRuns)) {
    if (persistedRunIds.has(runId)) continue;
    state.activeRuns.delete(runId);
    state.progressCounts.delete(runId);
  }
  for (const runId of persistedRunIds) {
    state.activeRuns.add(runId);
    if (!state.progressCounts.has(runId)) state.progressCounts.set(runId, 0);
  }
  if (state.activeRuns.size > 0) {
    ensurePolling();
  } else if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function stageDescription(stage, detail) {
  const descriptions = {
    queued: "Queued for final transcription",
    "retry:queued": "Retry queued from the retained recording",
    "audio:persist:start": "Saving a recovery copy locally",
    "audio:persisted": "Recording and live-caption draft saved locally",
    "audio:retained": "Recording retained for retry",
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
    "voiceprint:sample:selected": "Clean voice excerpts selected",
    "voiceprint:labels:coalesced": "Split provider voice labels combined",
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
    const selectedBeforeRefresh = state.selectedSessionId;
    setProcessingDetail("Final transcript saved locally");
    await Promise.all([loadSpeakers(), loadSessions()]);
    if (
      sessionId &&
      !state.recording &&
      !state.liveWorkspaceSelected &&
      state.selectedSessionId === selectedBeforeRefresh &&
      selectedBeforeRefresh === sessionId
    ) {
      await selectSession(sessionId);
    }
    finishRun(runId);
    showToast("Conversation transcribed and attributed.");
  } else if (event.stage === "error") {
    finishRun(runId);
    const selectedBeforeRefresh = state.selectedSessionId;
    await loadSessions();
    const failedSession = state.sessions.find(
      (session) => session.processing_run_id === runId && session.processing_status === "failed",
    );
    if (
      failedSession &&
      !state.recording &&
      !state.liveWorkspaceSelected &&
      state.selectedSessionId === selectedBeforeRefresh &&
      selectedBeforeRefresh === failedSession.id
    ) {
      await selectSession(failedSession.id);
    }
    showToast(
      failedSession && failedSession.recoverable_audio
        ? "Final transcription failed. The recording is safe and ready to retry."
        : detail,
      "error",
    );
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

async function loadSessions({ invalidateCache = false } = {}) {
  try {
    const selectedTranscript = state.sessions.find(
      (session) => session.id === state.selectedSessionId,
    )?.transcript;
    const previousSessions = new Map(
      state.sessions.map((session) => [
        session.id,
        [
          session.processing_status,
          session.processing_error,
          session.processing_run_id,
        ].join("\u0000"),
      ]),
    );
    const sessions = await invoke("list_sessions");
    if (invalidateCache) {
      invalidateConversationCache(state.conversationCache);
    } else {
      for (const session of sessions) {
        const previous = previousSessions.get(session.id);
        const current = [
          session.processing_status,
          session.processing_error,
          session.processing_run_id,
        ].join("\u0000");
        if (previous !== undefined && previous !== current) {
          invalidateConversationCache(state.conversationCache, session.id);
        }
        previousSessions.delete(session.id);
      }
      for (const removedId of previousSessions.keys()) {
        invalidateConversationCache(state.conversationCache, removedId);
      }
    }
    if (selectedTranscript !== undefined) {
      const selected = sessions.find(
        (session) => session.id === state.selectedSessionId,
      );
      if (selected) selected.transcript = selectedTranscript;
    }
    state.sessions = sessions;
    reconcileTrackedRuns(state.sessions);
    renderSessions();
    if (
      state.selectedSessionId &&
      !state.sessions.some((session) => session.id === state.selectedSessionId)
    ) {
      state.selectedSessionId = null;
      state.selectedSegments = [];
      state.recapState = null;
      state.importedArtifact = null;
      state.translationIndex = new Map();
      state.renderedSegmentCount = 0;
    }
    renderSpeakers();
    updateContentVisibility();
  } catch (error) {
    addActivity("Could not load conversations: " + errorText(error), "error");
  }
}

function renderSessions() {
  const query = elements.conversationSearch.value.trim().toLowerCase();
  const filtered = filterSessions(
    state.sessions,
    query,
    state.voiceFilteredSessionIds,
    query ? state.conversationSearchIds : null,
  );
  elements.sessionsList.replaceChildren();
  if (state.recording) {
    const current = document.createElement("button");
    current.type = "button";
    current.className =
      "session-item current-recording" + (state.liveWorkspaceSelected ? " selected" : "");
    current.dataset.currentRecording = "true";
    const title = document.createElement("strong");
    title.textContent = "Current recording";
    const meta = document.createElement("span");
    meta.dataset.currentRecordingMeta = "true";
    meta.textContent = "Recording · " + elements.recordingTimer.textContent;
    current.append(title, meta);
    current.addEventListener("click", selectCurrentRecording);
    elements.sessionsList.append(current);
  }
  if (!filtered.length) {
    if (state.recording) return;
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
      "session-item" +
      (!state.liveWorkspaceSelected && session.id === state.selectedSessionId ? " selected" : "");
    if (isSessionProcessing(session)) {
      button.classList.add("processing");
    } else if (session.processing_status === "failed") {
      button.classList.add("failed");
    }
    const sessionRecapJob = recapJob(session.id);
    if (sessionRecapJob?.status === "running") button.classList.add("recapping");
    if (sessionRecapJob?.status === "error") button.classList.add("recap-failed");
    const title = document.createElement("strong");
    title.textContent = sessionTitle(session);
    const meta = document.createElement("span");
    const parts = [sessionDate(session)];
    if (session.duration_ms > 0) parts.push(formatDuration(session.duration_ms));
    if (session.processing_status === "failed") parts.push("Final transcript needs retry");
    if (session.processing_status === "cleanup_failed") parts.push("Audio cleanup needed");
    if (sessionRecapJob?.status === "running") parts.push("Recap in progress");
    if (sessionRecapJob?.status === "error") parts.push("Recap failed");
    meta.textContent = parts.filter(Boolean).join(" · ");
    button.append(title, meta);
    button.addEventListener("click", () => selectSession(session.id));
    elements.sessionsList.append(button);
  }
}

function scheduleConversationSearch() {
  if (state.conversationSearchTimer) {
    window.clearTimeout(state.conversationSearchTimer);
    state.conversationSearchTimer = null;
  }
  const query = elements.conversationSearch.value.trim();
  const sequence = ++state.conversationSearchSequence;
  state.conversationSearchIds = null;
  renderSessions();
  if (!query) return;
  state.conversationSearchTimer = window.setTimeout(async () => {
    state.conversationSearchTimer = null;
    try {
      const matchingIds = await invoke("search_session_ids", { query });
      if (
        sequence !== state.conversationSearchSequence ||
        query !== elements.conversationSearch.value.trim()
      ) {
        return;
      }
      state.conversationSearchIds = new Set(matchingIds);
      renderSessions();
    } catch (error) {
      if (sequence !== state.conversationSearchSequence) return;
      addActivity("Could not search transcript text: " + errorText(error), "error");
    }
  }, 250);
}

function selectCurrentRecording() {
  if (!state.recording) return;
  state.navigationRevision += 1;
  state.liveWorkspaceSelected = true;
  state.openQueuedDraft = false;
  state.openQueuedDraftRevision = null;
  state.sessionLoadSequence += 1;
  renderSessions();
  renderSpeakers();
  updateContentVisibility();
}

async function selectSession(sessionId, { userInitiated = true } = {}) {
  const session = state.sessions.find((candidate) => candidate.id === sessionId);
  if (!session) return;
  if (userInitiated) state.navigationRevision += 1;
  state.liveWorkspaceSelected = false;
  state.openQueuedDraft = false;
  state.openQueuedDraftRevision = null;
  const sequence = ++state.sessionLoadSequence;
  state.selectedSessionId = sessionId;
  state.selectedSegments = [];
  state.recapState = null;
  state.importedArtifact = null;
  state.translationIndex = new Map();
  state.renderedSegmentCount = 0;
  state.activeRecapTab = "transcript";
  renderSpeakers();
  elements.segmentsList.replaceChildren();
  elements.loadMoreSegments.hidden = true;
  elements.legacyTranscript.hidden = false;
  renderSessions();
  updateContentVisibility();
  const cached = getCachedConversation(state.conversationCache, sessionId);
  if (cached) {
    applyConversationPayload(session, cached);
    return;
  }
  elements.legacyTranscript.textContent = "Loading conversation…";
  elements.saveState.textContent = "Loading…";
  try {
    const payload = await invoke("load_conversation", { sessionId });
    if (sequence !== state.sessionLoadSequence) return;
    setCachedConversation(
      state.conversationCache,
      sessionId,
      payload,
      CONVERSATION_CACHE_LIMIT,
    );
    applyConversationPayload(session, payload);
  } catch (error) {
    addActivity("Could not load conversation: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
  updateContentVisibility();
}

function applyConversationPayload(session, payload) {
  if (payload?.session) Object.assign(session, payload.session);
  state.selectedSegments = payload?.segments || [];
  state.recapState = payload?.recap_state || null;
  state.importedArtifact = payload?.imported_artifact || null;
  state.translationIndex = indexTranslations(
    state.recapState?.recap?.payload?.translations || [],
  );
  state.renderedSegmentCount = nextRenderedSegmentCount(
    state.selectedSegments.length,
    0,
    SEGMENT_RENDER_BATCH,
  );
  rememberNativeRecapState(session.id, state.recapState);
  renderRecapShell();
  renderTranscript(session);
  renderSpeakers();
  elements.saveState.textContent = "Saved locally";
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
  state.translationIndex = indexTranslations(
    state.recapState?.recap?.payload?.translations || [],
  );
  const cached = state.conversationCache.get(sessionId);
  if (cached) cached.recap_state = recapState;
  rememberNativeRecapState(sessionId, recapState);
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
  const importedExecutive = Boolean(
    state.importedArtifact?.executive_summary?.trim(),
  );
  const importedFull = Boolean(state.importedArtifact?.full_summary?.trim());
  const importedTasks = Boolean(state.importedArtifact?.tasks?.trim());
  elements.executiveTab.hidden = !availableTabs.has("executive");
  elements.fullSummaryTab.hidden = !availableTabs.has("full");
  elements.actionsTab.hidden = !availableTabs.has("actions");
  elements.agendaCoverageTab.hidden = !availableTabs.has("agenda");
  elements.importedExecutiveTab.hidden = !importedExecutive;
  elements.importedFullSummaryTab.hidden = !importedFull;
  elements.importedTasksTab.hidden = !importedTasks;
  elements.recapStaleBanner.hidden = !(
    hasRecap && state.recapState && state.recapState.stale
  );
  if (
    !hasRecap &&
    state.activeRecapTab !== "transcript" &&
    !state.activeRecapTab.startsWith("imported-")
  ) {
    state.activeRecapTab = "transcript";
  }
  if (state.activeRecapTab === "agenda" && !hasAgendaCoverage) {
    state.activeRecapTab = "transcript";
  }
  if (state.activeRecapTab === "imported-executive" && !importedExecutive) {
    state.activeRecapTab = "transcript";
  }
  if (state.activeRecapTab === "imported-full" && !importedFull) {
    state.activeRecapTab = "transcript";
  }
  if (state.activeRecapTab === "imported-tasks" && !importedTasks) {
    state.activeRecapTab = "transcript";
  }
  selectRecapTab(state.activeRecapTab);
  updateContentVisibility();
}

function selectRecapTab(tab) {
  const recapRecord = state.recapState && state.recapState.recap;
  const importedTabs = new Set([
    "imported-executive",
    "imported-full",
    "imported-tasks",
  ]);
  if (tab !== "transcript" && !importedTabs.has(tab) && !recapRecord) {
    tab = "transcript";
  }
  if (importedTabs.has(tab) && !state.importedArtifact) tab = "transcript";
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
  if (state.generatedLanguage === "original") return String(value.original || "");
  return String(value.translated || value.english || value.original || "");
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
  if (state.activeRecapTab.startsWith("imported-")) {
    renderImportedArtifactTab();
    return;
  }
  const payload = state.recapState?.recap?.payload;
  elements.generatedContent.replaceChildren();
  if (!payload) return;
  elements.generatedLanguageToggle.hidden = false;
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
    state.generatedLanguage === "translated",
  );
  elements.showEnglishButton.textContent = translationLanguageName(
    payload.target_language || "en",
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

function renderImportedArtifactTab() {
  const artifact = state.importedArtifact;
  elements.generatedContent.replaceChildren();
  elements.generatedLanguageToggle.hidden = true;
  if (!artifact) return;
  const contentByTab = {
    "imported-executive": [
      "Imported from " + artifact.source_provider,
      "Executive summary",
      artifact.executive_summary,
    ],
    "imported-full": [
      "Imported from " + artifact.source_provider,
      "Full summary",
      artifact.full_summary,
    ],
    "imported-tasks": [
      "Imported from " + artifact.source_provider,
      "Tasks",
      artifact.tasks,
    ],
  };
  const [eyebrow, title, content] =
    contentByTab[state.activeRecapTab] || contentByTab["imported-full"];
  elements.generatedEyebrow.textContent = eyebrow;
  elements.generatedTitle.textContent = title;
  const provenance = document.createElement("p");
  provenance.className = "imported-artifact-provenance";
  provenance.textContent =
    "Saved from the source archive on " +
    new Date(artifact.imported_at).toLocaleString() +
    ". It was not generated by Recall.";
  const body = document.createElement("pre");
  body.className = "imported-artifact-content";
  body.textContent = content || "";
  elements.generatedContent.append(provenance, body);
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
  const conversationLocked = Boolean(session && recapIsRunning(session.id));
  if (conversationLocked) {
    elements.saveState.textContent = "Edits paused while recap runs";
  } else {
    elements.saveState.textContent = "Saved locally";
  }
  if (!state.selectedSegments.length) {
    elements.loadMoreSegments.hidden = true;
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
  if (!state.renderedSegmentCount) {
    state.renderedSegmentCount = nextRenderedSegmentCount(
      state.selectedSegments.length,
      0,
      SEGMENT_RENDER_BATCH,
    );
  }
  const visibleSegments = state.selectedSegments.slice(0, state.renderedSegmentCount);
  for (const segment of visibleSegments) {
    const row = document.createElement("article");
    row.className = "segment";
    row.dataset.segmentId = segment.id;
    const speakerColumn = document.createElement("div");
    speakerColumn.className = "segment-speaker";
    const speakerButton = document.createElement("button");
    speakerButton.type = "button";
    speakerButton.className = "segment-speaker-button";
    speakerButton.textContent = segment.speaker_label || "Unknown speaker";
    speakerButton.setAttribute(
      "aria-label",
      "Speaker for intervention at " + formatTimestamp(segment.start_ms),
    );
    speakerButton.disabled = conversationLocked;
    speakerButton.addEventListener("click", () => {
      openSpeakerPicker(segment);
    });
    const time = document.createElement("time");
    time.textContent = formatTimestamp(segment.start_ms);
    speakerColumn.append(time, speakerButton);

    const body = document.createElement("div");
    body.className = "segment-body";
    const presentation = document.createElement("div");
    presentation.dataset.segmentPresentation = "true";
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
      presentation.append(rich);
      if (plan.fallbacks.length) {
        const fallbacks = document.createElement("div");
        fallbacks.className = "translation-fallbacks";
        for (const fallback of plan.fallbacks) {
          const translation = document.createElement("p");
          translation.className = "translation-fallback";
          translation.textContent =
            "(TRANSLATION: " +
            (fallback.translated_text || fallback.english_translation || "") +
            ")";
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
        presentation.append(fallbacks);
      }
    } else {
      const text = document.createElement("div");
      text.className = "segment-text-display";
      text.textContent = segment.text || "";
      presentation.append(text);
    }
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "text-button segment-edit-button";
    edit.textContent = "Edit transcript";
    edit.disabled = conversationLocked;
    edit.addEventListener("click", () => {
      beginSegmentEdit(segment, session, body, presentation, edit);
    });
    body.append(presentation, edit);
    row.append(speakerColumn, body);
    elements.segmentsList.append(row);
  }
  const remaining = state.selectedSegments.length - visibleSegments.length;
  elements.loadMoreSegments.hidden = remaining <= 0;
  if (remaining > 0) {
    elements.loadMoreSegments.textContent =
      "Show next " +
      Math.min(SEGMENT_RENDER_BATCH, remaining) +
      " interventions · " +
      remaining +
      " remaining";
  }
}

function segmentTranslations(segmentId) {
  if (!state.recapState?.recap || state.recapState.stale) return [];
  return state.translationIndex.get(segmentId) || [];
}

function beginSegmentEdit(segment, session, body, presentation, editButton) {
  if (body.querySelector(".segment-text")) return;
  presentation.hidden = true;
  editButton.hidden = true;
  const textarea = document.createElement("textarea");
  textarea.className = "segment-text";
  textarea.value = segment.text || "";
  textarea.setAttribute("aria-label", "Transcript intervention");
  body.append(textarea);
  autoResize(textarea);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  let cancelled = false;
  textarea.addEventListener("input", () => {
    autoResize(textarea);
    elements.saveState.textContent = "Unsaved changes";
  });
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      cancelled = true;
      renderTranscript(session);
      return;
    }
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") textarea.blur();
  });
  textarea.addEventListener("blur", async () => {
    if (cancelled || !textarea.isConnected) return;
    const value = textarea.value.trim();
    if (value === (segment.text || "").trim()) {
      elements.saveState.textContent = "Saved locally";
      renderTranscript(session);
      return;
    }
    await saveSegmentText(segment, value);
  });
}

function showMoreSegments() {
  if (state.renderedSegmentCount >= state.selectedSegments.length) return;
  state.renderedSegmentCount = nextRenderedSegmentCount(
    state.selectedSegments.length,
    state.renderedSegmentCount + SEGMENT_RENDER_BATCH,
    SEGMENT_RENDER_BATCH,
  );
  const session = state.sessions.find(
    (candidate) => candidate.id === state.selectedSessionId,
  );
  renderTranscript(session);
}

function ensureSegmentRendered(index) {
  const requiredCount = nextRenderedSegmentCount(
    state.selectedSegments.length,
    state.renderedSegmentCount,
    SEGMENT_RENDER_BATCH,
    index,
  );
  if (requiredCount <= state.renderedSegmentCount) return;
  state.renderedSegmentCount = requiredCount;
  const session = state.sessions.find(
    (candidate) => candidate.id === state.selectedSessionId,
  );
  renderTranscript(session);
}

function openSpeakerPicker(segment) {
  if (!segment || recapIsRunning(segment.session_id)) return;
  state.speakerPickerSegmentId = segment.id;
  elements.speakerPickerSearch.value = "";
  renderSpeakerPicker();
  elements.speakerPickerDialog.showModal();
  window.setTimeout(() => elements.speakerPickerSearch.focus(), 0);
}

function renderSpeakerPicker() {
  const segment = state.selectedSegments.find(
    (candidate) => candidate.id === state.speakerPickerSegmentId,
  );
  const query = elements.speakerPickerSearch.value.trim().toLocaleLowerCase();
  const speakers = state.speakers
    .filter((speaker) =>
      String(speaker.label || "Unnamed voice").toLocaleLowerCase().includes(query),
    )
    .sort((left, right) =>
      String(left.label || "Unnamed voice").localeCompare(
        String(right.label || "Unnamed voice"),
        undefined,
        { sensitivity: "base", numeric: true },
      ),
    );
  elements.speakerPickerResults.replaceChildren();
  for (const speaker of speakers) {
    const option = document.createElement("button");
    option.type = "button";
    option.className =
      "speaker-picker-option" +
      (speaker.id === segment?.speaker_id ? " selected" : "");
    option.setAttribute("role", "option");
    option.setAttribute(
      "aria-selected",
      speaker.id === segment?.speaker_id ? "true" : "false",
    );
    const label = document.createElement("strong");
    label.textContent = speaker.label || "Unnamed voice";
    const detail = document.createElement("span");
    const conversations = Number(speaker.conversation_count) || 0;
    detail.textContent =
      conversations +
      " conversation" +
      (conversations === 1 ? "" : "s");
    option.append(label, detail);
    option.addEventListener("click", () => chooseSpeakerFromPicker(speaker.id));
    elements.speakerPickerResults.append(option);
  }
  if (!speakers.length) {
    const empty = document.createElement("p");
    empty.className = "speaker-picker-empty";
    empty.textContent = "No matching people or voices.";
    elements.speakerPickerResults.append(empty);
  }
  elements.speakerPickerUnknown.disabled = !segment?.speaker_id;
}

async function chooseSpeakerFromPicker(speakerId) {
  const segment = state.selectedSegments.find(
    (candidate) => candidate.id === state.speakerPickerSegmentId,
  );
  if (!segment) return;
  if (elements.speakerPickerDialog.open) elements.speakerPickerDialog.close();
  state.speakerPickerSegmentId = null;
  await assignSegmentSpeaker(segment, speakerId || null);
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
  if (recapIsRunning(segment.session_id)) return;
  elements.saveState.textContent = "Saving…";
  try {
    await invoke("update_segment_text", {
      segmentId: segment.id,
      sessionId: segment.session_id,
      text,
    });
    invalidateConversationCache(state.conversationCache, segment.session_id);
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
  if (recapIsRunning(segment.session_id)) return;
  elements.saveState.textContent = "Saving…";
  try {
    await invoke("assign_segment_speaker", {
      segmentId: segment.id,
      sessionId: segment.session_id,
      speakerId,
    });
    invalidateConversationCache(state.conversationCache, segment.session_id);
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
  if (recapIsRunning(state.selectedSessionId)) return;
  const sessionId = state.selectedSessionId;
  const title = elements.conversationTitle.value.trim();
  if (!title) {
    const session = state.sessions.find((candidate) => candidate.id === sessionId);
    elements.conversationTitle.value = sessionTitle(session);
    return;
  }
  try {
    await invoke("update_session_title", {
      sessionId,
      title,
    });
    invalidateConversationCache(state.conversationCache, sessionId);
    const session = state.sessions.find((candidate) => candidate.id === sessionId);
    if (session) session.title = title;
    renderSessions();
    addActivity("Conversation title saved", "success");
  } catch (error) {
    addActivity("Could not save conversation title: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

async function retrySelectedProcessing() {
  if (!state.selectedSessionId) return;
  const sessionId = state.selectedSessionId;
  elements.retryProcessingButton.disabled = true;
  addActivity("Retrying final transcription from the retained recording");
  try {
    const queued = await invoke("retry_processing", { sessionId });
    await loadSessions();
    const stored = state.sessions.find((session) => session.id === sessionId);
    if (stored && ["queued", "processing"].includes(stored.processing_status)) {
      trackRun(queued.run_id);
    } else {
      finishRun(queued.run_id);
    }
    await selectSession(sessionId);
    addActivity(
      "[" + queued.run_id.slice(0, 8) + "] Final transcription retry queued",
      "success",
    );
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not retry final transcription: " + message, "error");
    showToast(message, "error");
    await loadSessions();
  } finally {
    elements.retryProcessingButton.disabled = false;
  }
}

async function discardSelectedRetainedAudio() {
  if (!state.selectedSessionId) return;
  const sessionId = state.selectedSessionId;
  const session = state.sessions.find((candidate) => candidate.id === sessionId);
  if (session && session.recoverable_audio) {
    const confirmed = await requestConfirmation({
      title: "Remove the retained recording?",
      message:
        "The final transcript will stay in Recall, but the recovery WAV cannot be restored after deletion.",
      acceptLabel: "Remove recording",
    });
    if (!confirmed) return;
  }
  elements.discardRetainedAudioButton.disabled = true;
  try {
    await invoke("discard_retained_audio", { sessionId });
    addActivity("Retained recording removed", "success");
    await loadSessions();
    await selectSession(sessionId);
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not remove the retained recording: " + message, "error");
    showToast(message, "error");
  } finally {
    elements.discardRetainedAudioButton.disabled = false;
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
      "” and its transcript will be deleted. Any retained recovery recording will also be removed. Named people are kept. Any unnamed VOICE profiles used only by this conversation will also be removed.",
    acceptLabel: "Delete conversation",
  });
  if (!confirmed) return;
  elements.deleteSessionButton.disabled = true;
  addActivity("Deleting conversation “" + sessionTitle(session) + "”…");
  try {
    const removedVoices = await invoke("delete_session", {
      sessionId,
    });
    invalidateConversationCache(state.conversationCache, sessionId);
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
    state.translationIndex = new Map();
    state.renderedSegmentCount = 0;
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
    const previousSignature = speakerLibrarySignature(state.speakers);
    const speakers = await invoke("list_speakers_with_stats");
    const nextSignature = speakerLibrarySignature(speakers);
    if (previousSignature && previousSignature !== nextSignature) {
      invalidateConversationCache(state.conversationCache);
    }
    state.speakers = speakers;
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

function speakerLibrarySignature(speakers) {
  return (speakers || [])
    .map((speaker) =>
      [
        speaker.id,
        speaker.label,
        speaker.embedding_count,
        speaker.sample_count,
        speaker.conversation_count,
      ].join("\u0000"),
    )
    .sort()
    .join("\u0001");
}

function renderConversationSpeakerFilter() {
  const selectedKey = elements.conversationSpeakerFilter.value;
  elements.conversationSpeakerFilter.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "All voices";
  elements.conversationSpeakerFilter.append(all);
  const filterGroups = groupVoiceFilters(state.speakers);
  for (const group of filterGroups) {
    const option = document.createElement("option");
    option.value = group.key;
    option.textContent = group.label;
    option.dataset.speakerIds = JSON.stringify(group.speakerIds);
    elements.conversationSpeakerFilter.append(option);
  }
  const selectedStillExists = filterGroups.some((group) => group.key === selectedKey);
  if (selectedKey && selectedStillExists) {
    elements.conversationSpeakerFilter.value = selectedKey;
  } else if (selectedKey) {
    state.voiceFilterSequence += 1;
    state.voiceFilteredSessionIds = null;
  }
}

async function applyConversationVoiceFilter() {
  const selectedOption = elements.conversationSpeakerFilter.selectedOptions[0];
  const filterKey = elements.conversationSpeakerFilter.value;
  const sequence = ++state.voiceFilterSequence;
  if (!filterKey) {
    state.voiceFilteredSessionIds = null;
    renderSessions();
    return;
  }
  try {
    const speakerIds = JSON.parse(selectedOption?.dataset.speakerIds || "[]");
    if (!Array.isArray(speakerIds) || !speakerIds.length) {
      throw new Error("The selected voice filter has no profiles.");
    }
    const sessionIds = await invoke("list_session_ids_for_speakers", { speakerIds });
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
  if (state.recording && state.liveWorkspaceSelected) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent = "Voice profiles will appear after the recording is processed.";
    elements.speakersList.append(empty);
    return;
  }
  const selectedSession = state.sessions.find(
    (session) => session.id === state.selectedSessionId,
  );
  if (state.queueingProcessing || isSessionProcessing(selectedSession)) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent = "Detecting and identifying voices for this conversation…";
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
  const unknownSegments = state.selectedSegments.filter((segment) => !segment.speaker_id);
  const currentSpeakers = state.speakers.filter((speaker) => selectedSpeakerIds.has(speaker.id));
  if (!currentSpeakers.length && !unknownSegments.length) {
    const empty = document.createElement("div");
    empty.className = "people-empty";
    empty.textContent = "No manageable voice profiles are attributed in this conversation.";
    elements.speakersList.append(empty);
    return;
  }
  if (unknownSegments.length) {
    elements.speakersList.append(buildUnknownSpeakerCard(unknownSegments));
  }
  for (const speaker of currentSpeakers) {
    elements.speakersList.append(buildSpeakerCard(speaker, true, false));
  }
}

function renderVoiceLibrary() {
  if (!elements.voiceLibraryDialog.open || !state.identityPageData) return;
  renderIdentityManager();
}

function identityGroupKey(key) {
  return JSON.stringify([
    String(key?.session_id || ""),
    key?.speaker_label === null || key?.speaker_label === undefined
      ? null
      : String(key.speaker_label),
  ]);
}

function identityProfileAsSpeaker(profile) {
  return {
    id: profile.id,
    label: profile.label,
    created_at: profile.created_at,
    last_seen_at: profile.last_seen_at,
    sample_count: Number(profile.sample_count || 0),
    embedding_count: Number(profile.active_voiceprint_count || 0),
    inactive_voiceprint_count: Number(profile.inactive_voiceprint_count || 0),
    conversation_count: Number(profile.conversation_count || 0),
    duplicate_name_conflict: Boolean(profile.duplicate_name_conflict),
    duplicate_name_count: Number(profile.duplicate_name_count || 0),
    likely_match: null,
  };
}

function identitySelectionCounts() {
  return {
    profiles: state.selectedIdentityProfiles.size,
    groups: state.selectedUnassignedGroups.size,
  };
}

function canConsolidateIdentitySelection() {
  const { profiles, groups } = identitySelectionCounts();
  return groups > 0 || profiles > 1;
}

function identityStatusOptions() {
  if (state.identityManagerView === "unassigned") {
    return [
      ["all", "All unassigned"],
      ["generic", "Generic speaker labels"],
      ["labelled", "Source-labelled speakers"],
    ];
  }
  return [
    ["all", "All profiles"],
    ["named", "Named people"],
    ["provisional", "Provisional VOICE profiles"],
    ["no_voiceprint", "No current voiceprint"],
    ["conflict", "Duplicate names"],
    ["imported", "Imported profiles"],
  ];
}

function renderIdentityStatusOptions() {
  elements.identityStatusFilter.replaceChildren();
  const options = identityStatusOptions();
  if (!options.some(([value]) => value === state.identityStatus)) {
    state.identityStatus = "all";
  }
  for (const [value, label] of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    option.selected = value === state.identityStatus;
    elements.identityStatusFilter.append(option);
  }
}

function identityTag(label, className = "") {
  const tag = document.createElement("span");
  tag.className = "identity-tag" + (className ? " " + className : "");
  tag.textContent = label;
  return tag;
}

function buildIdentityProfileRow(profile) {
  const row = document.createElement("article");
  row.className = "identity-row identity-profile-row";
  row.dataset.identityProfileId = profile.id;

  const selection = document.createElement("label");
  selection.className = "identity-selection";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = state.selectedIdentityProfiles.has(profile.id);
  checkbox.setAttribute("aria-label", "Select " + profile.label);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      state.selectedIdentityProfiles.set(profile.id, profile);
    } else {
      state.selectedIdentityProfiles.delete(profile.id);
    }
    renderIdentitySelection();
  });
  selection.append(checkbox);

  const main = document.createElement("div");
  main.className = "identity-row-main";
  const title = document.createElement("div");
  title.className = "identity-row-title";
  title.textContent = profile.label;
  const meta = document.createElement("div");
  meta.className = "identity-row-meta";
  meta.textContent =
    Number(profile.conversation_count || 0).toLocaleString() +
    " conversation" +
    (Number(profile.conversation_count) === 1 ? "" : "s") +
    " · " +
    Number(profile.intervention_count || 0).toLocaleString() +
    " intervention" +
    (Number(profile.intervention_count) === 1 ? "" : "s") +
    " · last heard " +
    profileDate(profile.last_seen_at || profile.created_at);
  const counts = document.createElement("div");
  counts.className = "identity-row-counts";
  counts.textContent =
    Number(profile.active_voiceprint_count || 0).toLocaleString() +
    " current voiceprint" +
    (Number(profile.active_voiceprint_count) === 1 ? "" : "s") +
    (Number(profile.inactive_voiceprint_count || 0)
      ? " · " +
        Number(profile.inactive_voiceprint_count).toLocaleString() +
        " inactive"
      : "") +
    (Number(profile.sample_count || 0)
      ? " · " + Number(profile.sample_count).toLocaleString() + " retained sample"
      : "");
  const tags = document.createElement("div");
  tags.className = "identity-row-tags";
  if (profile.provisional) tags.append(identityTag("Provisional VOICE", "provisional"));
  else tags.append(identityTag("Named person", "named"));
  if (profile.active_voiceprint_count === 0) {
    tags.append(identityTag("No current voiceprint", "muted"));
  }
  if (profile.duplicate_name_conflict) {
    tags.append(
      identityTag(
        Number(profile.duplicate_name_count || 2) + " profiles share this name",
        "warning",
      ),
    );
  }
  if (profile.imported) tags.append(identityTag("Imported", "imported"));
  if (profile.sample_count > 0) tags.append(identityTag("Preview available", "sample"));
  main.append(title, meta, counts, tags);

  const actions = document.createElement("div");
  actions.className = "identity-row-actions";
  const speaker = identityProfileAsSpeaker(profile);
  const preview = actionButton("Preview", () => previewSpeaker(speaker));
  preview.disabled = Number(profile.sample_count || 0) === 0 || state.recording;
  preview.title = preview.disabled
    ? state.recording
      ? "Voice preview is unavailable during recording"
      : "No temporary sample is retained for this profile"
    : "Play the retained excerpt";
  const rename = actionButton(
    profile.provisional ? "Name" : "Rename",
    () => openNameDialog(speaker),
  );
  const remove = actionButton("Delete", () => deleteSpeaker(speaker), "danger-mini");
  if (!profile.provisional && Number(profile.conversation_count || 0) > 0) {
    remove.disabled = true;
    remove.title = "Reassign or remove this person's conversation history first";
  }
  actions.append(preview, rename, remove);

  row.append(selection, main, actions);
  return row;
}

function buildUnassignedIdentityRow(group) {
  const row = document.createElement("article");
  row.className = "identity-row identity-unassigned-row";
  const key = identityGroupKey(group.key);
  row.dataset.identityGroupKey = key;

  const selection = document.createElement("label");
  selection.className = "identity-selection";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = state.selectedUnassignedGroups.has(key);
  checkbox.setAttribute(
    "aria-label",
    "Select " + group.display_label + " in " + group.session_title,
  );
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      state.selectedUnassignedGroups.set(key, group);
    } else {
      state.selectedUnassignedGroups.delete(key);
    }
    renderIdentitySelection();
  });
  selection.append(checkbox);

  const main = document.createElement("div");
  main.className = "identity-row-main";
  const title = document.createElement("div");
  title.className = "identity-row-title";
  title.textContent = group.display_label;
  const conversation = document.createElement("div");
  conversation.className = "identity-row-conversation";
  conversation.textContent = group.session_title;
  const meta = document.createElement("div");
  meta.className = "identity-row-meta";
  meta.textContent =
    Number(group.intervention_count || 0).toLocaleString() +
    " intervention" +
    (Number(group.intervention_count) === 1 ? "" : "s") +
    " · " +
    formatTimestamp(Number(group.first_start_ms || 0)) +
    "–" +
    formatTimestamp(Number(group.last_end_ms || 0)) +
    " · " +
    profileDate(group.session_created_at);
  const tags = document.createElement("div");
  tags.className = "identity-row-tags";
  tags.append(
    identityTag(
      group.generic ? "Generic provider label" : "Source-labelled speaker",
      group.generic ? "muted" : "imported",
    ),
  );
  const scope = identityTag("This conversation only", "scoped");
  scope.title =
    "An identical label in another conversation is a separate group and will not be changed unless selected.";
  tags.append(scope);
  main.append(title, conversation, meta, tags);
  row.append(selection, main);
  return row;
}

function renderIdentitySelection() {
  const { profiles, groups } = identitySelectionCounts();
  const parts = [];
  if (profiles) {
    parts.push(
      profiles.toLocaleString() + " profile" + (profiles === 1 ? "" : "s"),
    );
  }
  if (groups) {
    parts.push(
      groups.toLocaleString() +
        " unassigned group" +
        (groups === 1 ? "" : "s"),
    );
  }
  elements.identitySelectionSummary.textContent = parts.length
    ? parts.join(" and ") + " selected"
    : "Nothing selected";
  const hasSelection = profiles + groups > 0;
  elements.identityClearSelection.disabled = !hasSelection;
  elements.identityMergeButton.disabled =
    !canConsolidateIdentitySelection() || state.identityOperationRunning;
}

function renderIdentityManager() {
  const page = state.identityPageData;
  elements.voiceLibraryList.replaceChildren();
  elements.identityProfilesTab.classList.toggle(
    "selected",
    state.identityManagerView === "profiles",
  );
  elements.identityProfilesTab.setAttribute(
    "aria-selected",
    String(state.identityManagerView === "profiles"),
  );
  elements.identityUnassignedTab.classList.toggle(
    "selected",
    state.identityManagerView === "unassigned",
  );
  elements.identityUnassignedTab.setAttribute(
    "aria-selected",
    String(state.identityManagerView === "unassigned"),
  );
  renderIdentityStatusOptions();
  if (!page) {
    elements.identityManagerStatus.textContent = "Loading…";
    renderIdentitySelection();
    return;
  }
  if (!page.items.length) {
    const empty = document.createElement("div");
    empty.className = "people-empty identity-empty";
    empty.textContent =
      state.identityManagerView === "profiles"
        ? "No profiles match this search and status."
        : "No unassigned transcript speakers match this search and status.";
    elements.voiceLibraryList.append(empty);
  } else {
    for (const item of page.items) {
      if (state.identityManagerView === "profiles") {
        if (state.selectedIdentityProfiles.has(item.id)) {
          state.selectedIdentityProfiles.set(item.id, item);
        }
        elements.voiceLibraryList.append(buildIdentityProfileRow(item));
      } else {
        const key = identityGroupKey(item.key);
        if (state.selectedUnassignedGroups.has(key)) {
          state.selectedUnassignedGroups.set(key, item);
        }
        elements.voiceLibraryList.append(buildUnassignedIdentityRow(item));
      }
    }
  }
  const start = page.total ? (page.page - 1) * page.page_size + 1 : 0;
  const end = Math.min(page.total, page.page * page.page_size);
  const noun = state.identityManagerView === "profiles" ? "profiles" : "groups";
  elements.identityManagerStatus.textContent =
    "Showing " +
    start.toLocaleString() +
    "–" +
    end.toLocaleString() +
    " of " +
    Number(page.total || 0).toLocaleString() +
    " " +
    noun;
  elements.identityPageStatus.textContent =
    "Page " + page.page.toLocaleString() + " of " + page.page_count.toLocaleString();
  elements.identityPreviousPage.disabled = page.page <= 1;
  elements.identityNextPage.disabled = page.page >= page.page_count;
  renderIdentitySelection();
}

async function loadIdentityManagerPage() {
  const sequence = ++state.identityLoadSequence;
  state.identityPageData = null;
  renderIdentityManager();
  const command =
    state.identityManagerView === "profiles"
      ? "list_identity_profiles"
      : "list_unassigned_identities";
  try {
    const page = await invoke(command, {
      search: state.identitySearch,
      status: state.identityStatus,
      page: state.identityPage,
      pageSize: state.identityPageSize,
    });
    if (sequence !== state.identityLoadSequence) return;
    state.identityPageData = page;
    state.identityPage = Number(page.page || 1);
    renderIdentityManager();
  } catch (error) {
    if (sequence !== state.identityLoadSequence) return;
    const message = errorText(error);
    elements.identityManagerStatus.textContent = "Could not load: " + message;
    elements.voiceLibraryList.replaceChildren();
    addActivity("Could not load People & Voices: " + message, "error");
  }
}

function buildUnknownSpeakerCard(segments) {
  const card = document.createElement("article");
  card.className = "speaker-card unresolved";
  const header = document.createElement("div");
  header.className = "speaker-header";
  const identity = document.createElement("div");
  identity.className = "speaker-identity";
  const avatar = document.createElement("div");
  avatar.className = "speaker-avatar";
  avatar.textContent = "?";
  const copy = document.createElement("div");
  const name = document.createElement("div");
  name.className = "speaker-name";
  name.textContent = "Unknown speaker";
  const duration = segments.reduce(
    (total, segment) => total + Math.max(0, Number(segment.end_ms) - Number(segment.start_ms)),
    0,
  );
  const meta = document.createElement("div");
  meta.className = "speaker-meta";
  meta.textContent =
    segments.length +
    (segments.length === 1 ? " intervention" : " interventions") +
    " · " +
    formatDuration(duration);
  copy.append(name, meta);
  identity.append(avatar, copy);
  header.append(identity);
  card.append(header);

  const tag = document.createElement("span");
  tag.className = "new-voice-tag";
  tag.textContent = "Needs review";
  const tags = document.createElement("div");
  tags.className = "speaker-tags";
  tags.append(tag);
  card.append(tags);

  const explanation = document.createElement("p");
  explanation.className = "speaker-card-explanation";
  explanation.textContent =
    "No safe voiceprint was available. If these turns belong to one person, group them into a VOICE profile; otherwise assign them individually in the transcript.";
  card.append(explanation);

  const actions = document.createElement("div");
  actions.className = "speaker-actions";
  actions.append(
    actionButton("Group as one voice…", createProfileForUnknownSegments, "primary-mini"),
    actionButton("Review turns", reviewUnknownInterventions),
  );
  card.append(actions);
  return card;
}

function reviewUnknownInterventions() {
  const index = state.selectedSegments.findIndex((candidate) => !candidate.speaker_id);
  const segment = index >= 0 ? state.selectedSegments[index] : null;
  if (index >= 0) ensureSegmentRendered(index);
  const row = segment
    ? Array.from(elements.segmentsList.children).find(
        (candidate) => candidate.dataset.segmentId === segment.id,
      )
    : null;
  if (!row) return;
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  const speakerButton = row.querySelector(".segment-speaker-button");
  if (speakerButton) {
    window.setTimeout(() => {
      speakerButton.focus();
      openSpeakerPicker(segment);
    }, 250);
  }
}

async function createProfileForUnknownSegments() {
  const sessionId = state.selectedSessionId;
  const unknownCount = state.selectedSegments.filter((segment) => !segment.speaker_id).length;
  if (!sessionId || !unknownCount) return;
  const confirmed = await requestConfirmation({
    title: "Group unknown turns as one voice?",
    message:
      "This will assign all " +
      unknownCount +
      " currently unknown interventions in this conversation to one new VOICE profile. Continue only if they belong to the same person; otherwise use the speaker dropdown on each intervention.",
    acceptLabel: "Create VOICE profile",
  });
  if (!confirmed) return;
  try {
    const label = await invoke("create_profile_for_unknown_segments", { sessionId });
    await loadSpeakers();
    await selectSession(sessionId);
    addActivity(label + " created for previously unknown interventions", "success");
    showToast(label + " is ready to name or assign.");
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not group unknown interventions: " + message, "error");
    showToast(message, "error");
  }
}

function buildSpeakerCard(speaker, inSelectedConversation, inVoiceLibrary) {
  const label = speaker.label || "Unnamed voice";
  const provisional = isProvisionalLabel(label);
  const likelyMatch = provisional ? speaker.likely_match : null;
  const duplicateNameConflict = Boolean(speaker.duplicate_name_conflict);
  const card = document.createElement("article");
  card.className =
    "speaker-card" +
    (provisional ? " provisional" : "") +
    (duplicateNameConflict ? " duplicate-conflict" : "");
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
    matchingTag.className = likelyMatch ? "likely-voice-tag" : "legacy-voice-tag";
    matchingTag.textContent = likelyMatch
      ? "Likely " + likelyMatch.label
      : "Not auto-matched";
    tags.append(matchingTag);
  } else if (duplicateNameConflict) {
    const conflictTag = document.createElement("span");
    conflictTag.className = "conflict-voice-tag";
    conflictTag.textContent =
      "Duplicate name · " +
      Number(speaker.duplicate_name_count || 2) +
      " profiles";
    tags.append(conflictTag);
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

  if (likelyMatch) {
    const explanation = document.createElement("p");
    explanation.className = "speaker-card-explanation likely-match-explanation";
    const runnerUp = likelyMatch.runner_up_label
      ? " The next different person was " +
        likelyMatch.runner_up_label +
        (Number.isFinite(Number(likelyMatch.runner_up_score))
          ? " at " + Number(likelyMatch.runner_up_score).toFixed(3)
          : "") +
        "."
      : "";
    explanation.textContent =
      "Best match: " +
      likelyMatch.label +
      " at " +
      Number(likelyMatch.score).toFixed(3) +
      " from " +
      Number(likelyMatch.support_count || 0) +
      " agreeing reference" +
      (Number(likelyMatch.support_count) === 1 ? "" : "s") +
      "." +
      runnerUp;
    card.append(explanation);
  } else if (duplicateNameConflict) {
    const explanation = document.createElement("p");
    explanation.className = "speaker-card-explanation duplicate-match-explanation";
    explanation.textContent =
      "More than one person profile uses this name. Automatic matching ignores all of them until you merge the duplicates or rename one.";
    card.append(explanation);
  }

  const actions = document.createElement("div");
  actions.className = "speaker-actions";
  const preview = actionButton("Preview", () => previewSpeaker(speaker));
  preview.disabled = speaker.sample_count === 0 || state.recording;
  preview.title =
    state.recording
      ? "Voice preview is unavailable during recording"
      : speaker.sample_count === 0
      ? "No sample is retained after a person is named"
      : "Play the excerpt used for this voiceprint";
  actions.append(preview);
  if (provisional) {
    if (likelyMatch) {
      actions.append(
        actionButton(
          "Assign to " + likelyMatch.label,
          () => acceptLikelyMatch(speaker),
          "primary-mini",
        ),
      );
    }
    const nameButton = actionButton(
      "Name person",
      () => openNameDialog(speaker),
      likelyMatch ? undefined : "primary-mini",
    );
    const assignButton = actionButton(
      likelyMatch ? "Choose another person…" : "Assign…",
      () => openAssignDialog(speaker),
    );
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

async function acceptLikelyMatch(speaker) {
  const likelyMatch = speaker.likely_match;
  if (!likelyMatch) {
    showToast("This voice no longer has a likely-person suggestion.", "error");
    return;
  }
  addActivity(
    "Assigning " +
      (speaker.label || "this voice") +
      " to likely person " +
      likelyMatch.label +
      "…",
  );
  try {
    const result = await invoke("accept_voice_match_suggestion", {
      sourceId: speaker.id,
      targetId: likelyMatch.speaker_id,
    });
    const quarantined = Number(result.quarantined_voiceprints || 0);
    addActivity(
      "Assigned voice history to " +
        result.target_label +
        "; " +
        Number(result.activated_voiceprints || 0) +
        " compatible voiceprint" +
        (Number(result.activated_voiceprints) === 1 ? "" : "s") +
        " activated" +
        (quarantined
          ? "; " +
            quarantined +
            " incompatible voiceprint" +
            (quarantined === 1 ? "" : "s") +
            " quarantined"
          : ""),
      quarantined ? undefined : "success",
    );
    showToast(
      quarantined
        ? "Person assigned; an incompatible voiceprint was kept out of automatic matching."
        : "Voice assigned to " + result.target_label + ".",
    );
    await refreshIdentityViews();
  } catch (error) {
    addActivity("Could not accept likely person: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

async function openVoiceLibrary() {
  if (!elements.voiceLibraryDialog.open) elements.voiceLibraryDialog.showModal();
  renderIdentityStatusOptions();
  state.identitySearch = elements.identitySearch.value.trim();
  await Promise.all([loadSpeakers(), loadIdentityManagerPage()]);
}

async function refreshIdentityViews() {
  await loadSpeakers();
  if (elements.voiceLibraryDialog.open) {
    await loadIdentityManagerPage();
  }
  if (state.selectedSessionId) {
    await selectSession(state.selectedSessionId);
  }
}

function setIdentityManagerView(view) {
  if (view === state.identityManagerView) return;
  state.identityManagerView = view;
  state.identityStatus = "all";
  state.identityPage = 1;
  state.identityPageData = null;
  renderIdentityStatusOptions();
  void loadIdentityManagerPage();
}

function scheduleIdentitySearch() {
  if (state.identitySearchTimer) window.clearTimeout(state.identitySearchTimer);
  state.identitySearchTimer = window.setTimeout(() => {
    state.identitySearchTimer = null;
    state.identitySearch = elements.identitySearch.value.trim();
    state.identityPage = 1;
    void loadIdentityManagerPage();
  }, 180);
}

function clearIdentitySelection() {
  state.selectedIdentityProfiles.clear();
  state.selectedUnassignedGroups.clear();
  state.identityPreview = null;
  state.identityPreviewSignature = null;
  renderIdentityManager();
}

function selectedIdentityProfiles() {
  return Array.from(state.selectedIdentityProfiles.values()).sort((left, right) =>
    String(left.label || "").localeCompare(String(right.label || ""), undefined, {
      sensitivity: "base",
      numeric: true,
    }),
  );
}

function namedIdentityTargets() {
  return state.speakers
    .filter((speaker) => !isProvisionalLabel(speaker.label))
    .sort((left, right) =>
      String(left.label || "").localeCompare(String(right.label || ""), undefined, {
        sensitivity: "base",
        numeric: true,
      }),
    );
}

function invalidateIdentityPreview() {
  state.identityPreview = null;
  state.identityPreviewSignature = null;
  elements.identityImpact.hidden = true;
  elements.identityImpactStats.replaceChildren();
  elements.identityImpactWarnings.replaceChildren();
  elements.identityConfirmButton.disabled = true;
  elements.identityMergeFeedback.textContent = "";
}

function setIdentityTargetOptions() {
  const profiles = selectedIdentityProfiles();
  const previous = elements.identityTarget.value;
  elements.identityTarget.replaceChildren();
  if (profiles.length) {
    for (const profile of profiles) {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent =
        profile.label +
        (profile.provisional ? " · provisional" : "") +
        " · " +
        Number(profile.conversation_count || 0).toLocaleString() +
        " conversations";
      elements.identityTarget.append(option);
    }
  } else {
    const create = document.createElement("option");
    create.value = "__new__";
    create.textContent = "Create a new name-only person";
    elements.identityTarget.append(create);
    for (const speaker of namedIdentityTargets()) {
      const option = document.createElement("option");
      option.value = speaker.id;
      option.textContent =
        (speaker.label || "Unnamed person") +
        " · " +
        Number(speaker.conversation_count || 0).toLocaleString() +
        " conversations";
      elements.identityTarget.append(option);
    }
  }
  if (
    previous &&
    Array.from(elements.identityTarget.options).some(
      (option) => option.value === previous,
    )
  ) {
    elements.identityTarget.value = previous;
  }
  syncIdentityFinalLabelToTarget(true);
}

function identityTargetLabel() {
  const targetId = elements.identityTarget.value;
  if (!targetId || targetId === "__new__") return "";
  const selected = state.selectedIdentityProfiles.get(targetId);
  if (selected) return selected.label || "";
  return (
    state.speakers.find((speaker) => speaker.id === targetId)?.label || ""
  );
}

function syncIdentityFinalLabelToTarget(force = false) {
  const previousAutomatic = elements.identityFinalLabel.dataset.automaticValue || "";
  const targetLabel = identityTargetLabel();
  if (
    force ||
    !elements.identityFinalLabel.value.trim() ||
    elements.identityFinalLabel.value === previousAutomatic
  ) {
    elements.identityFinalLabel.value = isProvisionalLabel(targetLabel)
      ? ""
      : targetLabel;
  }
  elements.identityFinalLabel.dataset.automaticValue =
    elements.identityFinalLabel.value;
  invalidateIdentityPreview();
}

async function openIdentityMergeDialog() {
  if (!canConsolidateIdentitySelection()) return;
  const { profiles, groups } = identitySelectionCounts();
  elements.identityMergeSelection.textContent =
    profiles.toLocaleString() +
    " profile" +
    (profiles === 1 ? "" : "s") +
    " and " +
    groups.toLocaleString() +
    " unassigned group" +
    (groups === 1 ? "" : "s") +
    " will be reviewed together. Choose the person whose identity should remain.";
  setIdentityTargetOptions();
  invalidateIdentityPreview();
  if (!elements.identityMergeDialog.open) {
    elements.identityMergeDialog.showModal();
  }
  window.setTimeout(() => {
    if (elements.identityFinalLabel.value) elements.identityFinalLabel.select();
    else elements.identityFinalLabel.focus();
  }, 0);
}

function identityConsolidationRequest() {
  return {
    profile_ids: Array.from(state.selectedIdentityProfiles.keys()),
    unassigned_groups: Array.from(state.selectedUnassignedGroups.values()).map(
      (group) => serializableCopy(group.key),
    ),
    target_speaker_id:
      elements.identityTarget.value &&
      elements.identityTarget.value !== "__new__"
        ? elements.identityTarget.value
        : null,
    final_label: elements.identityFinalLabel.value.trim(),
  };
}

function identityRequestSignature(request) {
  return JSON.stringify({
    profile_ids: [...request.profile_ids].sort(),
    unassigned_groups: [...request.unassigned_groups].sort((left, right) =>
      identityGroupKey(left).localeCompare(identityGroupKey(right)),
    ),
    target_speaker_id: request.target_speaker_id,
    final_label: request.final_label,
  });
}

function identityImpactStat(value, label) {
  const item = document.createElement("div");
  item.className = "identity-impact-stat";
  const number = document.createElement("strong");
  number.textContent = Number(value || 0).toLocaleString();
  const copy = document.createElement("span");
  copy.textContent = label;
  item.append(number, copy);
  return item;
}

function renderIdentityImpact(preview) {
  elements.identityImpactStats.replaceChildren(
    identityImpactStat(preview.affected_conversation_count, "conversations"),
    identityImpactStat(preview.affected_intervention_count, "interventions"),
    identityImpactStat(preview.stale_recap_count, "recaps made out of date"),
    identityImpactStat(preview.active_voiceprint_count, "current voiceprints reviewed"),
    identityImpactStat(preview.inactive_voiceprint_count, "inactive voiceprints kept"),
    identityImpactStat(preview.samples_to_delete, "temporary samples deleted"),
  );
  elements.identityImpactWarnings.replaceChildren();
  const warnings = preview.warnings || [];
  if (!warnings.length) {
    const item = document.createElement("li");
    item.textContent = "No additional warnings.";
    elements.identityImpactWarnings.append(item);
  } else {
    for (const warning of warnings) {
      const item = document.createElement("li");
      item.textContent = warning;
      elements.identityImpactWarnings.append(item);
    }
  }
  elements.identityImpact.hidden = false;
}

async function previewIdentityConsolidation() {
  const request = identityConsolidationRequest();
  if (!request.final_label) {
    elements.identityMergeFeedback.textContent = "Enter the final display name.";
    elements.identityFinalLabel.focus();
    return;
  }
  elements.identityPreviewButton.disabled = true;
  elements.identityMergeFeedback.textContent = "Calculating the exact impact…";
  invalidateIdentityPreview();
  elements.identityMergeFeedback.textContent = "Calculating the exact impact…";
  try {
    const preview = await invoke("preview_identity_consolidation", {
      request: serializableCopy(request),
    });
    state.identityPreview = preview;
    state.identityPreviewSignature = identityRequestSignature(request);
    renderIdentityImpact(preview);
    elements.identityMergeFeedback.textContent =
      "Review the impact below before confirming.";
    elements.identityConfirmButton.disabled = false;
  } catch (error) {
    const message = errorText(error);
    elements.identityMergeFeedback.textContent = message;
    addActivity("Could not preview the people and voices change: " + message, "error");
  } finally {
    elements.identityPreviewButton.disabled = false;
  }
}

async function confirmIdentityConsolidation() {
  const request = identityConsolidationRequest();
  if (
    !state.identityPreview ||
    state.identityPreviewSignature !== identityRequestSignature(request)
  ) {
    invalidateIdentityPreview();
    elements.identityMergeFeedback.textContent =
      "The selection or final name changed. Review the impact again.";
    return;
  }
  const preview = state.identityPreview;
  const affectedSessionIds = new Set(preview.affected_session_ids || []);
  if (elements.identityMergeDialog.open) elements.identityMergeDialog.close();
  if (elements.voiceLibraryDialog.open) elements.voiceLibraryDialog.close();
  state.identityOperationRunning = true;
  elements.identityOperationBadge.hidden = false;
  elements.peopleVoicesButton.classList.add("working");
  renderIdentitySelection();
  const startedAt = performance.now();
  addActivity(
    "People & Voices: changing " +
      Number(preview.affected_conversation_count || 0).toLocaleString() +
      " conversations…",
  );
  showToast("People and voice changes are running in the background.");
  try {
    const result = await invoke("consolidate_identities", {
      request: serializableCopy(request),
    });
    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    addActivity(
      "People & Voices: " +
        result.target_label +
        " saved across " +
        Number(result.affected_conversation_count || 0).toLocaleString() +
        " conversations in " +
        elapsedSeconds.toFixed(1) +
        "s; verified backup " +
        result.backup_path,
      "success",
    );
    showToast(result.target_label + " and the selected history were updated.");
    clearIdentitySelection();
    await loadSpeakers();
    if (
      state.selectedSessionId &&
      affectedSessionIds.has(state.selectedSessionId)
    ) {
      await selectSession(state.selectedSessionId);
    }
  } catch (error) {
    const message = errorText(error);
    addActivity("People & Voices change failed: " + message, "error");
    showToast(message, "error");
  } finally {
    state.identityOperationRunning = false;
    elements.identityOperationBadge.hidden = true;
    elements.peopleVoicesButton.classList.remove("working");
    renderIdentitySelection();
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
  if (state.recording) {
    showToast("Voice preview is unavailable during recording.", "error");
    return;
  }
  addActivity("Loading preview for " + speaker.label);
  try {
    const samples = await invoke("get_speaker_samples", { speakerId: speaker.id });
    if (state.recording) return;
    if (!samples.length) {
      showToast("No sample is retained for this profile.", "error");
      addActivity("No preview is retained for " + speaker.label, "error");
      await loadSpeakers();
      return;
    }
    if (state.previewAudio) state.previewAudio.pause();
    state.previewAudio = new Audio("data:audio/wav;base64," + samples[0].sample_b64);
    if (state.recording) {
      state.previewAudio = null;
      return;
    }
    await state.previewAudio.play();
    if (state.recording) {
      stopVoicePreview();
      return;
    }
    addActivity("Playing voice preview for " + speaker.label, "success");
  } catch (error) {
    addActivity("Could not play voice preview: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

function stopVoicePreview() {
  if (!state.previewAudio) return;
  state.previewAudio.pause();
  state.previewAudio.currentTime = 0;
  state.previewAudio = null;
}

function openNameDialog(speaker) {
  const provisional = isProvisionalLabel(speaker.label);
  const hasVoiceprint = Number(speaker.embedding_count) > 0;
  elements.nameSpeakerId.value = speaker.id;
  elements.nameDialogTitle.textContent = provisional ? "Name this person" : "Rename this person";
  elements.saveSpeakerNameButton.textContent = provisional ? "Save name" : "Save new name";
  elements.speakerName.value = provisional ? "" : speaker.label || "";
  elements.nameDialogHelp.textContent = !hasVoiceprint
    ? provisional
      ? "This labels the selected interventions, but no safe voiceprint was available. Automatic recognition will begin only after you later assign a clean VOICE profile to this person."
      : "This changes the name used in saved transcripts. This person has no current voiceprint, so the name is not used for automatic recognition yet."
    : "Naming enables automatic recognition for this person. The temporary voice excerpt is then deleted; only the local reference voiceprint remains.";
  elements.nameDialog.showModal();
  window.setTimeout(() => elements.speakerName.focus(), 0);
}

async function saveSpeakerName(event) {
  event.preventDefault();
  const speakerId = elements.nameSpeakerId.value;
  const name = elements.speakerName.value.trim();
  if (!speakerId || !name) return;
  try {
    const result = await invoke("rename_speaker", { speakerId, newLabel: name });
    if (result?.status === "conflict" && result.conflicting_speaker_id) {
      elements.nameDialog.close();
      await loadSpeakers();
      const source = state.speakers.find((speaker) => speaker.id === speakerId);
      addActivity(
        "The name " +
          (result.conflicting_label || name) +
          " already belongs to another profile; choose whether to assign or merge this voice",
      );
      showToast(
        "That person already exists. Assign this voice to the existing profile instead.",
        "error",
      );
      if (source) openAssignDialog(source, result.conflicting_speaker_id);
      return;
    }
    elements.nameDialog.close();
    addActivity("Voice profile named " + name + "; temporary sample deleted", "success");
    showToast("Voice profile saved as " + name + ".");
    await refreshIdentityViews();
  } catch (error) {
    addActivity("Could not name voice profile: " + errorText(error), "error");
    showToast(errorText(error), "error");
  }
}

function openAssignDialog(source, preferredTargetId = null) {
  const candidates = state.speakers.filter(
    (candidate) => candidate.id !== source.id && !isProvisionalLabel(candidate.label),
  );
  if (!candidates.length) {
    showToast("There are no other named people to assign this voice to.", "error");
    return;
  }
  elements.assignSourceId.value = source.id;
  elements.assignTarget.replaceChildren();
  candidates.sort((left, right) =>
    String(left.label || "").localeCompare(String(right.label || ""), undefined, {
      sensitivity: "base",
      numeric: true,
    }),
  );
  for (const candidate of candidates) {
    const option = document.createElement("option");
    option.value = candidate.id;
    option.textContent =
      (candidate.label || "Unnamed voice") +
      (candidate.duplicate_name_conflict
        ? " · duplicate profile · " +
          candidate.conversation_count +
          " conversation" +
          (candidate.conversation_count === 1 ? "" : "s")
        : "");
    elements.assignTarget.append(option);
  }
  if (
    preferredTargetId &&
    candidates.some((candidate) => candidate.id === preferredTargetId)
  ) {
    elements.assignTarget.value = preferredTargetId;
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
    const result = await invoke("merge_speakers", {
      sourceId,
      targetId,
      replaceEmbeddings,
    });
    const target = state.speakers.find((speaker) => speaker.id === targetId);
    elements.assignDialog.close();
    addActivity(
      "Voice profile assigned to " +
        (target ? target.label : "existing person") +
        (replaceEmbeddings
          ? "; prior voiceprints replaced"
          : "; " +
            Number(result.activated_voiceprints || 0) +
            " compatible voiceprint" +
            (Number(result.activated_voiceprints) === 1 ? "" : "s") +
            " added") +
        (Number(result.quarantined_voiceprints || 0)
          ? "; " +
            Number(result.quarantined_voiceprints) +
            " incompatible voiceprint" +
            (Number(result.quarantined_voiceprints) === 1 ? "" : "s") +
            " quarantined"
          : ""),
      Number(result.quarantined_voiceprints || 0) ? undefined : "success",
    );
    showToast(
      Number(result.quarantined_voiceprints || 0)
        ? "Person assigned; an incompatible voiceprint was kept out of automatic matching."
        : "Voice profile assignment saved.",
    );
    await refreshIdentityViews();
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
    await refreshIdentityViews();
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
  if (!session) return "";
  const lines = [markdown ? "# " + sessionTitle(session) : sessionTitle(session), ""];
  const heading = (value, level = 2) =>
    markdown ? "#".repeat(level) + " " + value : value.toUpperCase();
  if (state.activeRecapTab.startsWith("imported-")) {
    const artifact = state.importedArtifact;
    if (!artifact) return "";
    const imported = {
      "imported-executive": ["Executive summary", artifact.executive_summary],
      "imported-full": ["Full summary", artifact.full_summary],
      "imported-tasks": ["Tasks", artifact.tasks],
    };
    const [title, content] = imported[state.activeRecapTab] || imported["imported-full"];
    lines.push(heading(title), "", content || "");
    return lines.join("\n").trim();
  }
  const payload = state.recapState?.recap?.payload;
  if (!payload) return "";
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
  const sessionId = state.selectedSessionId;
  const text = elements.agendaText.value;
  if (!text.trim()) {
    elements.agendaFeedback.textContent = "Paste agenda text, or choose a file.";
    return;
  }
  elements.saveAgendaTextButton.disabled = true;
  elements.agendaFeedback.textContent = "Saving locally…";
  try {
    await invoke("save_agenda_text", { sessionId, text });
    invalidateConversationCache(state.conversationCache, sessionId);
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
  const sessionId = state.selectedSessionId;
  elements.attachAgendaButton.disabled = true;
  elements.agendaFeedback.textContent = "Waiting for file selection…";
  try {
    const agenda = await invoke("choose_agenda_file", {
      sessionId,
    });
    if (!agenda) {
      elements.agendaFeedback.textContent = "No file selected.";
      return;
    }
    invalidateConversationCache(state.conversationCache, sessionId);
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
  const sessionId = state.selectedSessionId;
  const confirmed = await requestConfirmation({
    title: "Remove this agenda?",
    message:
      "The locally stored agenda will be removed. Any existing recap remains visible but becomes out of date until regenerated.",
    acceptLabel: "Remove agenda",
  });
  if (!confirmed) return;
  try {
    await invoke("remove_agenda", { sessionId });
    invalidateConversationCache(state.conversationCache, sessionId);
    await refreshRecapState({ rerenderTranscript: false });
    renderAgendaDialog();
    elements.agendaFeedback.textContent = "Agenda removed.";
    addActivity("Agenda removed from the conversation", "success");
  } catch (error) {
    elements.agendaFeedback.textContent = errorText(error);
    addActivity("Could not remove agenda: " + errorText(error), "error");
  }
}

async function requestRecap() {
  if (!state.selectedSessionId || recapIsRunning(state.selectedSessionId)) return;
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
  void runRecap(false);
}

function reviewUnresolvedParticipants() {
  elements.unresolvedDialog.close();
  selectRecapTab("transcript");
  const unresolved = new Set(state.recapState?.unresolved_profiles || []);
  const index = state.selectedSegments.findIndex(
    (candidate) =>
      !candidate.speaker_id || unresolved.has(candidate.speaker_label || "Unknown speaker"),
  );
  const segment = index >= 0 ? state.selectedSegments[index] : null;
  if (index >= 0) ensureSegmentRendered(index);
  const row = segment
    ? Array.from(elements.segmentsList.children).find(
        (candidate) => candidate.dataset.segmentId === segment.id,
      )
    : null;
  if (row) {
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    const speakerButton = row.querySelector(".segment-speaker-button");
    if (speakerButton) {
      window.setTimeout(() => {
        speakerButton.focus();
        openSpeakerPicker(segment);
      }, 250);
    }
  }
  showToast("Assign or name the highlighted participant, then click Recap again.");
}

async function runRecap(allowUnresolved) {
  const sessionId = state.selectedSessionId;
  if (!sessionId || recapIsRunning(sessionId)) return;
  const session = state.sessions.find((candidate) => candidate.id === sessionId);
  const label = sessionTitle(session);
  state.recapJobs.set(sessionId, {
    status: "running",
    stage: "prepare",
    detail: "Preparing transcript and agenda…",
  });
  if (state.selectedSessionId === sessionId && state.recapState) {
    state.recapState.in_flight = true;
  }
  renderSessions();
  updateContentVisibility();
  renderTranscript(session);
  addActivity("[recap · " + label + "] Starting on-demand LLM recap");
  try {
    const commandState = await invoke("generate_recap", { sessionId, allowUnresolved });
    await loadSessions();
    if (state.selectedSessionId === sessionId) {
      const persistedState = await invoke("get_recap_state", { sessionId });
      if (!persistedState?.recap?.payload) {
        throw new Error("The LLM provider finished, but Recall could not load the saved recap.");
      }
      state.recapState = persistedState;
      state.translationIndex = indexTranslations(
        state.recapState?.recap?.payload?.translations || [],
      );
      invalidateConversationCache(state.conversationCache, sessionId);
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
      "[recap · " + label + "] LLM recap saved locally" +
        (usage
          ? " (" + usage.input_tokens + " input / " + usage.output_tokens + " output tokens)"
          : ""),
      "success",
    );
    state.recapJobs.delete(sessionId);
    showToast("Recap ready for “" + label + "”.");
  } catch (error) {
    const message = errorText(error);
    state.recapJobs.set(sessionId, {
      status: "error",
      stage: "error",
      detail: message,
    });
    addActivity("[recap · " + label + "] LLM recap failed: " + message, "error");
    showToast(message, "error");
    if (state.selectedSessionId === sessionId) {
      try {
        await refreshRecapState({ rerenderTranscript: false });
      } catch {
        // Preserve the currently rendered recap if refreshing also fails.
      }
    }
  } finally {
    if (state.selectedSessionId === sessionId && state.recapState) {
      state.recapState.in_flight = recapJob(sessionId)?.status === "running";
    }
    renderSessions();
    updateContentVisibility();
    if (state.selectedSessionId === sessionId) {
      const selected = state.sessions.find((candidate) => candidate.id === sessionId);
      if (selected) renderTranscript(selected);
    }
  }
}

async function loadSettingsData() {
  return loadSettingsDataInner();
}

function normalizedImportName(value) {
  return String(value || "")
    .normalize("NFKC")
    .toLocaleLowerCase()
    .trim()
    .replace(/\s+/g, " ");
}

function jamieIdentityValidationIssue(identity, preview = state.jamieImportPreview) {
  if (!preview || !identity || identity.generic) return null;
  const decision = identity.decision;
  if (!decision || decision.action === "review") {
    return "Choose how to import this source name.";
  }
  if (decision.action === "proposed_map") {
    return "Accept or change the suggested match.";
  }
  if (decision.action === "map_existing") {
    const knownIds = new Set((preview.known_people || []).map((person) => person.id));
    if (!knownIds.has(decision.target_speaker_id)) {
      return "Choose an existing person.";
    }
  }
  if (decision.action === "create_named") {
    const name = normalizedImportName(decision.display_name);
    if (!name) {
      return "Enter a display name.";
    }
    const knownNames = new Set(
      (preview.known_people || []).map((person) =>
        normalizedImportName(person.label),
      ),
    );
    if (knownNames.has(name)) {
      return "That person already exists in Recall. Map the source name to the existing person instead.";
    }
  }
  return null;
}

function jamieImportErrors() {
  const preview = state.jamieImportPreview;
  if (!preview) return ["Choose a Jamie export first."];
  const errors = [];
  for (const warning of preview.archive_warnings || []) {
    if (warning.blocking) errors.push(warning.message);
  }
  for (const identity of preview.identities || []) {
    const issue = jamieIdentityValidationIssue(identity, preview);
    if (issue) errors.push(identity.alias + ": " + issue);
  }
  for (const meeting of preview.meetings || []) {
    if (!meeting.included || meeting.already_imported) continue;
    for (const warning of meeting.warnings || []) {
      if (warning.blocking) errors.push(meeting.title + ": " + warning.message);
    }
  }
  const newMeetingCount = (preview.meetings || []).filter(
    (meeting) => meeting.included && !meeting.already_imported,
  ).length;
  if (!newMeetingCount) errors.push("No new meetings are selected for import.");
  return Array.from(new Set(errors));
}

function jamieDecisionCounts() {
  const identities = (state.jamieImportPreview?.identities || []).filter(
    (identity) => !identity.generic,
  );
  return {
    total: identities.length,
    review: identities.filter((identity) =>
      Boolean(jamieIdentityValidationIssue(identity)),
    ).length,
    mapped: identities.filter(
      (identity) => identity.decision?.action === "map_existing",
    ).length,
    created: identities.filter(
      (identity) => identity.decision?.action === "create_named",
    ).length,
    unresolved: identities.filter(
      (identity) => identity.decision?.action === "unresolved",
    ).length,
  };
}

function importStat(label, value) {
  const item = document.createElement("div");
  item.className = "import-stat";
  const number = document.createElement("strong");
  number.textContent = Number(value || 0).toLocaleString();
  const copy = document.createElement("span");
  copy.textContent = label;
  item.append(number, copy);
  return item;
}

function renderJamieImportOverview() {
  const preview = state.jamieImportPreview;
  if (!preview) return;
  const metadata = preview.metadata || {};
  const selectedMeetings = (preview.meetings || []).filter(
    (meeting) => meeting.included && !meeting.already_imported,
  ).length;
  const genericCount = (preview.identities || []).filter(
    (identity) => identity.generic,
  ).length;
  elements.jamieImportSource.textContent =
    metadata.user ? "Jamie export for " + metadata.user : "Jamie meeting export";
  const sourceParts = [];
  if (metadata.export_date) {
    sourceParts.push("Exported " + new Date(metadata.export_date).toLocaleString());
  }
  sourceParts.push(
    (Number(metadata.source_size_bytes || 0) / 1_000_000).toFixed(1) + " MB",
  );
  sourceParts.push("source " + String(metadata.source_sha256 || "").slice(0, 12));
  elements.jamieImportSummary.textContent = sourceParts.join(" · ");
  elements.jamieImportStats.replaceChildren(
    importStat("meetings", preview.meetings.length),
    importStat("selected", selectedMeetings),
    importStat("interventions", preview.total_intervention_count),
    importStat("source names", jamieDecisionCounts().total),
  );
  elements.jamieGenericIdentityNote.textContent =
    genericCount.toLocaleString() +
    " generic labels such as “Speaker 0” stay local to their meetings and will not create people in the Voice Library.";
}

function renderJamieValidation() {
  const errors = jamieImportErrors();
  const counts = jamieDecisionCounts();
  elements.jamieValidationPanel.replaceChildren();
  const title = document.createElement("strong");
  const detail = document.createElement("span");
  if (!errors.length) {
    elements.jamieValidationPanel.className = "import-validation ready";
    title.textContent = "Ready to import";
    detail.textContent =
      counts.mapped +
      " mapped · " +
      counts.created +
      " new people · " +
      counts.unresolved +
      " left unresolved";
  } else {
    elements.jamieValidationPanel.className = "import-validation warning";
    title.textContent =
      errors.length.toLocaleString() +
      " review item" +
      (errors.length === 1 ? "" : "s") +
      " remaining";
    const visible = errors.slice(0, 4);
    detail.textContent =
      visible.join(" ") +
      (errors.length > visible.length
        ? " " + (errors.length - visible.length) + " more."
        : "");
  }
  elements.jamieValidationPanel.append(title, detail);
  elements.jamieImportButton.disabled = Boolean(errors.length) || state.jamieImportRunning;
  renderJamieImportOverview();
}

function jamieIdentityIssueControl(identity, row) {
  const existing = row.querySelector(".jamie-identity-issue");
  if (existing) existing.remove();
  const issue = jamieIdentityValidationIssue(identity);
  if (!issue) return;
  const message = document.createElement("p");
  message.className = "jamie-identity-issue";
  message.textContent = issue;
  row.append(message);
}

function jamieIdentitySecondaryControl(identity, row) {
  const decision = identity.decision;
  const preview = state.jamieImportPreview;
  const existing = row.querySelector(".jamie-identity-secondary");
  if (existing) existing.remove();
  const container = document.createElement("div");
  container.className = "jamie-identity-secondary";
  if (decision.action === "map_existing" || decision.action === "proposed_map") {
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Existing person for " + identity.alias);
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "Choose a person…";
    select.append(empty);
    for (const person of preview.known_people || []) {
      const option = document.createElement("option");
      option.value = person.id;
      option.textContent = person.label;
      option.selected = person.id === decision.target_speaker_id;
      select.append(option);
    }
    select.addEventListener("change", () => {
      decision.target_speaker_id = select.value || null;
      const person = (preview.known_people || []).find(
        (candidate) => candidate.id === select.value,
      );
      decision.display_name = person?.label || null;
      if (decision.action === "proposed_map" && select.value) {
        decision.action = "map_existing";
        const action = row.querySelector(".jamie-identity-action");
        if (action) action.value = "map_existing";
      }
      scheduleJamieImportDraftSave();
      renderJamieValidation();
      jamieIdentityIssueControl(identity, row);
    });
    container.append(select);
    if (decision.action === "proposed_map" && decision.target_speaker_id) {
      const accept = document.createElement("button");
      accept.type = "button";
      accept.className = "secondary-button compact-action";
      accept.textContent = "Accept match";
      accept.addEventListener("click", () => {
        decision.action = "map_existing";
        const action = row.querySelector(".jamie-identity-action");
        if (action) action.value = "map_existing";
        scheduleJamieImportDraftSave();
        jamieIdentitySecondaryControl(identity, row);
        renderJamieValidation();
        jamieIdentityIssueControl(identity, row);
      });
      container.append(accept);
    }
  } else if (decision.action === "create_named") {
    const input = document.createElement("input");
    input.type = "text";
    input.value = decision.display_name || identity.alias;
    input.placeholder = "Display name";
    input.setAttribute("aria-label", "New Recall name for " + identity.alias);
    input.addEventListener("input", () => {
      decision.display_name = input.value;
      scheduleJamieImportDraftSave();
      renderJamieValidation();
      jamieIdentityIssueControl(identity, row);
    });
    container.append(input);
  } else if (decision.action === "unresolved") {
    const note = document.createElement("span");
    note.className = "field-help";
    note.textContent =
      "Transcript turns keep the source label. No person or voiceprint is created.";
    container.append(note);
  }
  row.append(container);
}

function renderJamieIdentityList() {
  const preview = state.jamieImportPreview;
  if (!preview) return;
  const query = elements.jamieIdentitySearch.value.trim().toLocaleLowerCase();
  const needsReviewOnly = elements.jamieIdentityNeedsReview.checked;
  const identities = preview.identities.filter((identity) => {
    if (identity.generic) return false;
    if (query && !identity.alias.toLocaleLowerCase().includes(query)) return false;
    if (
      needsReviewOnly &&
      !jamieIdentityValidationIssue(identity, preview)
    ) {
      return false;
    }
    return true;
  });
  elements.jamieIdentityList.replaceChildren();
  if (!identities.length) {
    const empty = document.createElement("p");
    empty.className = "import-list-empty";
    empty.textContent = needsReviewOnly
      ? "No source names currently need attention."
      : "No source names match this search.";
    elements.jamieIdentityList.append(empty);
    return;
  }
  for (const identity of identities) {
    const row = document.createElement("article");
    row.className = "jamie-identity-row";
    const heading = document.createElement("div");
    heading.className = "jamie-identity-heading";
    const copy = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = identity.alias;
    const meta = document.createElement("span");
    meta.textContent =
      identity.meeting_count.toLocaleString() +
      " meeting" +
      (identity.meeting_count === 1 ? "" : "s") +
      " · " +
      identity.intervention_count.toLocaleString() +
      " intervention" +
      (identity.intervention_count === 1 ? "" : "s");
    copy.append(name, meta);
    const action = document.createElement("select");
    action.className = "jamie-identity-action";
    action.setAttribute("aria-label", "Import action for " + identity.alias);
    const actionOptions = [
      ["review", "Choose…"],
      ["proposed_map", "Review suggested match"],
      ["map_existing", "Map to existing person"],
      ["create_named", "Create name-only person"],
      ["unresolved", "Leave unresolved"],
    ];
    for (const [value, label] of actionOptions) {
      if (value === "proposed_map" && identity.decision.action !== value) continue;
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      option.selected = identity.decision.action === value;
      action.append(option);
    }
    action.addEventListener("change", () => {
      identity.decision.action = action.value;
      if (action.value === "create_named" && !identity.decision.display_name) {
        identity.decision.display_name = identity.alias;
      }
      if (action.value === "unresolved" || action.value === "review") {
        identity.decision.target_speaker_id = null;
      }
      scheduleJamieImportDraftSave();
      jamieIdentitySecondaryControl(identity, row);
      renderJamieValidation();
      jamieIdentityIssueControl(identity, row);
    });
    heading.append(copy, action);
    row.append(heading);
    if (identity.excerpts?.length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Transcript excerpts";
      details.append(summary);
      for (const excerpt of identity.excerpts) {
        const paragraph = document.createElement("p");
        paragraph.textContent = excerpt;
        details.append(paragraph);
      }
      row.append(details);
    }
    jamieIdentitySecondaryControl(identity, row);
    jamieIdentityIssueControl(identity, row);
    elements.jamieIdentityList.append(row);
  }
}

function setJamieMeetingIncluded(meeting, included) {
  meeting.included = included;
  const excluded = new Set(state.jamieImportPreview.draft.excluded_meetings || []);
  if (included) excluded.delete(meeting.source_fingerprint);
  else excluded.add(meeting.source_fingerprint);
  state.jamieImportPreview.draft.excluded_meetings = Array.from(excluded);
  scheduleJamieImportDraftSave();
  renderJamieValidation();
}

function renderJamieMeetingList() {
  const preview = state.jamieImportPreview;
  if (!preview) return;
  const query = elements.jamieMeetingSearch.value.trim().toLocaleLowerCase();
  const issuesOnly = elements.jamieMeetingIssuesOnly.checked;
  const meetings = preview.meetings.filter((meeting) => {
    if (query && !meeting.title.toLocaleLowerCase().includes(query)) return false;
    if (issuesOnly && !(meeting.warnings || []).length) return false;
    return true;
  });
  elements.jamieMeetingList.replaceChildren();
  if (!meetings.length) {
    const empty = document.createElement("p");
    empty.className = "import-list-empty";
    empty.textContent = "No meetings match this view.";
    elements.jamieMeetingList.append(empty);
    return;
  }
  for (const meeting of meetings) {
    const row = document.createElement("label");
    row.className =
      "jamie-meeting-row" +
      ((meeting.warnings || []).some((warning) => warning.blocking)
        ? " blocking"
        : "") +
      (meeting.already_imported ? " imported" : "");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = meeting.included;
    checkbox.disabled = meeting.already_imported;
    checkbox.addEventListener("change", () =>
      setJamieMeetingIncluded(meeting, checkbox.checked),
    );
    const body = document.createElement("span");
    body.className = "jamie-meeting-copy";
    const title = document.createElement("strong");
    title.textContent = meeting.title || "Untitled meeting";
    const metadata = document.createElement("span");
    const parts = [];
    if (meeting.started_at) {
      parts.push(new Date(meeting.started_at).toLocaleString());
    }
    parts.push(
      meeting.intervention_count.toLocaleString() +
        " intervention" +
        (meeting.intervention_count === 1 ? "" : "s"),
    );
    parts.push(
      meeting.speaker_count.toLocaleString() +
        " source speaker" +
        (meeting.speaker_count === 1 ? "" : "s"),
    );
    metadata.textContent = parts.join(" · ");
    body.append(title, metadata);
    for (const warning of meeting.warnings || []) {
      const warningCopy = document.createElement("em");
      warningCopy.textContent = warning.message;
      body.append(warningCopy);
    }
    if (meeting.already_imported) {
      const badge = document.createElement("span");
      badge.className = "imported-meeting-badge";
      badge.textContent = "Already imported";
      body.append(badge);
    }
    row.append(checkbox, body);
    elements.jamieMeetingList.append(row);
  }
}

function renderJamieImportReview() {
  const preview = state.jamieImportPreview;
  elements.jamieImportLoading.hidden = true;
  elements.jamieImportError.hidden = true;
  elements.jamieImportReview.hidden = !preview;
  if (!preview) return;
  renderJamieImportOverview();
  renderJamieIdentityList();
  renderJamieMeetingList();
  renderJamieValidation();
}

function scheduleJamieImportDraftSave() {
  const preview = state.jamieImportPreview;
  if (!preview) return;
  preview.draft.updated_at = new Date().toISOString();
  preview.draft.identity_decisions = preview.identities.map(
    (identity) => identity.decision,
  );
  const revision = ++state.jamieImportRevision;
  if (state.jamieImportSaveTimer) {
    window.clearTimeout(state.jamieImportSaveTimer);
  }
  elements.jamieImportFeedback.textContent = "Saving review…";
  state.jamieImportSaveTimer = window.setTimeout(async () => {
    state.jamieImportSaveTimer = null;
    try {
      await invoke("save_jamie_import_draft", {
        draft: serializableCopy(preview.draft),
      });
      if (revision === state.jamieImportRevision) {
        elements.jamieImportFeedback.textContent = "Review saved locally.";
      }
    } catch (error) {
      if (revision === state.jamieImportRevision) {
        const message = errorText(error);
        elements.jamieImportFeedback.textContent = message;
        addActivity("Could not save Jamie import review: " + message, "error");
      }
    }
  }, 450);
}

function showJamieImportLoading() {
  state.jamieImportPreview = null;
  elements.jamieImportReview.hidden = true;
  elements.jamieImportError.hidden = true;
  elements.jamieImportErrorMessage.textContent = "";
  elements.jamieImportLoading.hidden = false;
  elements.jamieImportFeedback.textContent = "";
  if (elements.settingsDialog.open) elements.settingsDialog.close();
  if (!elements.jamieImportDialog.open) elements.jamieImportDialog.showModal();
}

function showJamieImportError(message) {
  elements.jamieImportLoading.hidden = true;
  elements.jamieImportReview.hidden = true;
  elements.jamieImportError.hidden = false;
  elements.jamieImportErrorMessage.textContent = message;
  elements.jamieImportFeedback.textContent = "";
  if (!elements.jamieImportDialog.open) elements.jamieImportDialog.showModal();
}

async function openJamieImport(command) {
  const choosingFile = command === "choose_jamie_export";
  const settingsWasOpen = elements.settingsDialog.open;
  addActivity(
    choosingFile
      ? "Choose a Jamie archive to inspect"
      : "Opening the saved Jamie import review",
  );
  try {
    let preview;
    if (choosingFile) {
      if (elements.settingsDialog.open) elements.settingsDialog.close();
      const sourcePath = await invoke("choose_jamie_export");
      if (!sourcePath) {
        if (settingsWasOpen && !elements.settingsDialog.open) {
          elements.settingsDialog.showModal();
        }
        return;
      }
      showJamieImportLoading();
      preview = await invoke("inspect_jamie_export", { sourcePath });
    } else {
      showJamieImportLoading();
      preview = await invoke(command);
    }
    if (!preview) {
      if (elements.jamieImportDialog.open) elements.jamieImportDialog.close();
      if (command === "resume_jamie_import") {
        showToast("There is no saved Jamie import review.");
      }
      return;
    }
    state.jamieImportPreview = preview;
    state.jamieImportRevision = 0;
    elements.jamieIdentitySearch.value = "";
    elements.jamieIdentityNeedsReview.checked = true;
    elements.jamieMeetingSearch.value = "";
    elements.jamieMeetingIssuesOnly.checked = false;
    renderJamieImportReview();
    addActivity(
      "Jamie archive inspected: " +
        preview.meetings.length.toLocaleString() +
        " meetings, " +
        preview.total_intervention_count.toLocaleString() +
        " interventions",
      "success",
    );
  } catch (error) {
    const message = errorText(error);
    showJamieImportError(message);
    addActivity("Could not inspect the Jamie archive: " + message, "error");
    showToast(message, "error");
  }
}

function useJamieSourceNames() {
  const preview = state.jamieImportPreview;
  if (!preview) return;
  const peopleByName = new Map(
    (preview.known_people || []).map((person) => [
      normalizedImportName(person.label),
      person,
    ]),
  );
  for (const identity of preview.identities) {
    if (identity.generic) continue;
    const suggested = identity.decision.target_speaker_id
      ? (preview.known_people || []).find(
          (person) => person.id === identity.decision.target_speaker_id,
        )
      : null;
    const exact = peopleByName.get(normalizedImportName(identity.alias));
    const person = suggested || exact;
    if (person) {
      identity.decision.action = "map_existing";
      identity.decision.target_speaker_id = person.id;
      identity.decision.display_name = person.label;
    } else {
      identity.decision.action = "create_named";
      identity.decision.target_speaker_id = null;
      identity.decision.display_name = identity.alias;
    }
  }
  scheduleJamieImportDraftSave();
  renderJamieIdentityList();
  renderJamieValidation();
}

function excludeUnreadableJamieMeetings() {
  const preview = state.jamieImportPreview;
  if (!preview) return;
  for (const meeting of preview.meetings) {
    if ((meeting.warnings || []).some((warning) => warning.blocking)) {
      meeting.included = false;
    }
  }
  preview.draft.excluded_meetings = preview.meetings
    .filter((meeting) => !meeting.included)
    .map((meeting) => meeting.source_fingerprint);
  scheduleJamieImportDraftSave();
  renderJamieMeetingList();
  renderJamieValidation();
}

async function runJamieImport() {
  const preview = state.jamieImportPreview;
  if (!preview || state.jamieImportRunning) return;
  const errors = jamieImportErrors();
  if (errors.length) {
    renderJamieValidation();
    return;
  }
  const meetingCount = preview.meetings.filter(
    (meeting) => meeting.included && !meeting.already_imported,
  ).length;
  const createdPeople = new Set(
    preview.identities
      .filter((identity) => identity.decision.action === "create_named")
      .map((identity) => normalizedImportName(identity.decision.display_name)),
  ).size;
  const reviewWasOpen = elements.jamieImportDialog.open;
  if (reviewWasOpen) elements.jamieImportDialog.close();
  const confirmed = await requestConfirmation({
    title: "Import this Jamie archive?",
    message:
      meetingCount.toLocaleString() +
      " meetings will be added to Recall. " +
      createdPeople.toLocaleString() +
      " name-only people will be created. Recall will make a verified database backup first.",
    acceptLabel: "Import archive",
  });
  if (!confirmed) {
    if (reviewWasOpen && !elements.jamieImportDialog.open) {
      elements.jamieImportDialog.showModal();
    }
    return;
  }
  if (state.jamieImportSaveTimer) {
    window.clearTimeout(state.jamieImportSaveTimer);
    state.jamieImportSaveTimer = null;
  }
  try {
    await invoke("save_jamie_import_draft", {
      draft: serializableCopy(preview.draft),
    });
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not save the final Jamie import review: " + message, "error");
    showToast(message, "error");
    if (!elements.jamieImportDialog.open) elements.jamieImportDialog.showModal();
    return;
  }
  state.jamieImportRunning = true;
  elements.jamieImportButton.disabled = true;
  elements.jamieImportFeedback.textContent = "Importing in the background…";
  addActivity(
    "Jamie import started: " + meetingCount.toLocaleString() + " meetings",
  );
  showToast("Jamie import is running in the background.");
  try {
    const result = await invoke("run_jamie_import", {
      draft: serializableCopy(preview.draft),
    });
    state.jamieImportPreview = null;
    addActivity(
      "Jamie import finished: " +
        result.imported_meetings.toLocaleString() +
        " meetings and " +
        result.imported_interventions.toLocaleString() +
        " interventions added",
      "success",
    );
    showToast(
      result.imported_meetings.toLocaleString() +
        " Jamie meetings imported. A verified backup was created.",
    );
    await Promise.all([loadSpeakers(), loadSessions(), loadJamieImportHistory()]);
  } catch (error) {
    const message = errorText(error);
    addActivity("Jamie import failed: " + message, "error");
    showToast(message, "error");
    if (!elements.jamieImportDialog.open) elements.jamieImportDialog.showModal();
    renderJamieImportReview();
  } finally {
    state.jamieImportRunning = false;
    renderJamieValidation();
  }
}

function renderJamieImportHistory() {
  elements.jamieImportHistory.replaceChildren();
  if (!state.importBatches.length) {
    const empty = document.createElement("p");
    empty.className = "field-help";
    empty.textContent = "No external archives have been imported.";
    elements.jamieImportHistory.append(empty);
    return;
  }
  for (const batch of state.importBatches) {
    const row = document.createElement("div");
    row.className = "import-history-row";
    const copy = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent =
      batch.source_provider + " · " + batch.meeting_count.toLocaleString() + " meetings";
    const meta = document.createElement("span");
    meta.textContent =
      new Date(batch.imported_at).toLocaleString() +
      " · " +
      (batch.status === "rolled_back" ? "Rolled back" : "Imported");
    copy.append(title, meta);
    row.append(copy);
    if (batch.status === "imported") {
      const rollback = document.createElement("button");
      rollback.type = "button";
      rollback.className = "text-button danger-text";
      rollback.textContent = "Roll back";
      rollback.addEventListener("click", () => rollbackJamieImport(batch));
      row.append(rollback);
    }
    elements.jamieImportHistory.append(row);
  }
}

async function loadJamieImportHistory() {
  try {
    state.importBatches = await invoke("list_import_batches");
    renderJamieImportHistory();
  } catch (error) {
    addActivity("Could not load archive import history: " + errorText(error), "error");
  }
}

async function rollbackJamieImport(batch) {
  const settingsWasOpen = elements.settingsDialog.open;
  if (settingsWasOpen) elements.settingsDialog.close();
  const confirmed = await requestConfirmation({
    title: "Roll back this Jamie import?",
    message:
      batch.meeting_count.toLocaleString() +
      " imported meetings will be removed. People created by the import are removed only when no remaining conversation uses them. Recall will make another verified backup first.",
    acceptLabel: "Roll back import",
  });
  if (!confirmed) {
    if (settingsWasOpen) await openSettings();
    return;
  }
  addActivity("Rolling back Jamie import " + batch.id.slice(0, 8) + "…");
  try {
    const result = await invoke("rollback_jamie_import", {
      importId: batch.id,
    });
    addActivity(
      "Jamie rollback finished: " +
        result.removed_meetings.toLocaleString() +
        " meetings removed",
      "success",
    );
    showToast(result.removed_meetings.toLocaleString() + " imported meetings removed.");
    await Promise.all([loadSpeakers(), loadSessions(), loadJamieImportHistory()]);
    if (!state.selectedSessionId && state.sessions.length && !state.recording) {
      await selectSession(state.sessions[0].id);
    }
    if (settingsWasOpen) await openSettings();
  } catch (error) {
    const message = errorText(error);
    addActivity("Could not roll back the Jamie import: " + message, "error");
    showToast(message, "error");
    if (settingsWasOpen) await openSettings();
  }
}

async function loadSettingsDataInner() {
  const [status, preferences, devices, translationLanguages, importBatches] = await Promise.all([
    invoke("app_status"),
    invoke("get_preferences"),
    invoke("list_input_devices"),
    invoke("list_translation_languages"),
    JAMIE_IMPORT_UI_ENABLED ? invoke("list_import_batches") : Promise.resolve([]),
  ]);
  state.status = status;
  state.preferences = preferences;
  state.translationLanguages = translationLanguages || [];
  state.importBatches = importBatches || [];
  renderJamieImportHistory();
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
  elements.preferredLanguage.replaceChildren();
  const preferredLanguage = normalizePreferredLanguage(preferences.preferred_language);
  let preferredAvailable = false;
  for (const language of state.translationLanguages) {
    const option = document.createElement("option");
    option.value = language.code;
    option.textContent = language.name + " (" + language.code + ")";
    option.selected = language.code === preferredLanguage;
    if (option.selected) preferredAvailable = true;
    elements.preferredLanguage.append(option);
  }
  if (!preferredAvailable) {
    const unavailable = document.createElement("option");
    unavailable.value = preferredLanguage;
    unavailable.textContent = preferredLanguage.toUpperCase() + " (unavailable for live translation)";
    unavailable.selected = true;
    elements.preferredLanguage.prepend(unavailable);
  }
  elements.showEnglishButton.textContent = translationLanguageName(preferredLanguage);
  elements.noTranslationLanguages.value = parseNoTranslationLanguages(
    (preferences.no_translation_languages || []).join(", "),
    preferredLanguage,
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
  const preferredLanguage = normalizePreferredLanguage(elements.preferredLanguage.value);
  const noTranslationLanguages = parseNoTranslationLanguages(
    elements.noTranslationLanguages.value,
    preferredLanguage,
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
      preferredLanguage,
      noTranslationLanguages,
    });
    state.preferences = {
      encryption_enabled: state.preferences ? state.preferences.encryption_enabled : false,
      selected_input_device: selectedInputDevice,
      language_hints: languageHints,
      live_transcription: liveTranscription,
      openai_model: openaiModel,
      preferred_language: preferredLanguage,
      no_translation_languages: noTranslationLanguages,
      onboarding_version: state.preferences?.onboarding_version || null,
    };
    elements.settingsFeedback.textContent = "Saved.";
    elements.noTranslationLanguages.value = noTranslationLanguages.join(", ");
    elements.showEnglishButton.textContent = translationLanguageName(preferredLanguage);
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
    const session = state.sessions.find((candidate) => candidate.id === progress.session_id);
    const label = session ? sessionTitle(session) : String(progress.session_id || "").slice(0, 8);
    addActivity(
      "[recap · " + label + "] " + progress.stage + ": " + (progress.detail || "Working…"),
      kind,
    );
    if (progress.session_id) {
      state.recapJobs.set(progress.session_id, {
        status: progress.stage === "error" ? "error" : "running",
        stage: progress.stage,
        detail: progress.detail || "Working…",
      });
      renderSessions();
      updateContentVisibility();
      if (progress.session_id === state.selectedSessionId) {
        const selected = state.sessions.find(
          (candidate) => candidate.id === state.selectedSessionId,
        );
        if (selected) renderTranscript(selected);
      }
    }
  });
  await listen("transcription:progress", (event) => {
    void handleProgressEvent(event.payload);
  });
  await listen("transcription:queued", (event) => {
    const queued = event.payload || {};
    const runId = queued.run_id;
    const sessionId = queued.session_id;
    if (!runId) return;
    const shouldOpenDraft =
      state.openQueuedDraft &&
      state.openQueuedDraftRevision === state.navigationRevision &&
      !state.recording;
    trackRun(runId);
    state.queueingProcessing = false;
    addActivity("[" + runId.slice(0, 8) + "] Queued from the menu bar");
    setProcessingDetail("Uploading the retained recording to the STT provider…");
    void loadSessions().then(() => {
      const stored = state.sessions.find((session) => session.id === sessionId);
      if (!stored || !["queued", "processing"].includes(stored.processing_status)) {
        finishRun(runId);
      }
      if (
        sessionId &&
        shouldOpenDraft &&
        state.openQueuedDraft &&
        state.openQueuedDraftRevision === state.navigationRevision &&
        !state.recording
      ) {
        state.openQueuedDraft = false;
        state.openQueuedDraftRevision = null;
        return selectSession(sessionId, { userInitiated: false });
      }
      return undefined;
    });
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
  elements.jamieImportSettingsSection.hidden = !JAMIE_IMPORT_UI_ENABLED;
  elements.jamieImportDialog.hidden = !JAMIE_IMPORT_UI_ENABLED;
  if (JAMIE_IMPORT_UI_ENABLED) {
    elements.jamieImportSettingsSection.removeAttribute("aria-hidden");
    elements.jamieImportDialog.removeAttribute("aria-hidden");
  }
  elements.recordButton.addEventListener("click", startRecording);
  elements.emptyRecordButton.addEventListener("click", startRecording);
  elements.refreshSessions.addEventListener("click", () =>
    loadSessions({ invalidateCache: true }),
  );
  elements.refreshSpeakers.addEventListener("click", loadSpeakers);
  elements.conversationSearch.addEventListener("input", scheduleConversationSearch);
  elements.conversationSpeakerFilter.addEventListener("change", applyConversationVoiceFilter);
  elements.loadMoreSegments.addEventListener("click", showMoreSegments);
  elements.speakerPickerSearch.addEventListener("input", renderSpeakerPicker);
  elements.speakerPickerUnknown.addEventListener("click", () =>
    chooseSpeakerFromPicker(null),
  );
  elements.liveTranscript.addEventListener("scroll", handleLiveScroll, { passive: true });
  elements.liveTranslatedTranscript.addEventListener("scroll", handleLiveScroll, {
    passive: true,
  });
  elements.jumpToLiveButton.addEventListener("click", () => setLiveFollow(true));
  elements.voiceLibraryButton.addEventListener("click", openVoiceLibrary);
  elements.peopleVoicesButton.addEventListener("click", openVoiceLibrary);
  elements.identityProfilesTab.addEventListener("click", () =>
    setIdentityManagerView("profiles"),
  );
  elements.identityUnassignedTab.addEventListener("click", () =>
    setIdentityManagerView("unassigned"),
  );
  elements.identitySearch.addEventListener("input", scheduleIdentitySearch);
  elements.identityStatusFilter.addEventListener("change", () => {
    state.identityStatus = elements.identityStatusFilter.value;
    state.identityPage = 1;
    void loadIdentityManagerPage();
  });
  elements.identityRefreshButton.addEventListener("click", () =>
    loadIdentityManagerPage(),
  );
  elements.identityPreviousPage.addEventListener("click", () => {
    if (state.identityPage <= 1) return;
    state.identityPage -= 1;
    void loadIdentityManagerPage();
  });
  elements.identityNextPage.addEventListener("click", () => {
    if (
      state.identityPageData &&
      state.identityPage >= state.identityPageData.page_count
    ) {
      return;
    }
    state.identityPage += 1;
    void loadIdentityManagerPage();
  });
  elements.identityClearSelection.addEventListener("click", clearIdentitySelection);
  elements.identityMergeButton.addEventListener("click", openIdentityMergeDialog);
  elements.identityTarget.addEventListener("change", () =>
    syncIdentityFinalLabelToTarget(false),
  );
  elements.identityFinalLabel.addEventListener("input", invalidateIdentityPreview);
  elements.identityPreviewButton.addEventListener(
    "click",
    previewIdentityConsolidation,
  );
  elements.identityConfirmButton.addEventListener(
    "click",
    confirmIdentityConsolidation,
  );
  elements.agendaButton.addEventListener("click", openAgendaDialog);
  elements.recapButton.addEventListener("click", requestRecap);
  elements.staleRegenerateButton.addEventListener("click", requestRecap);
  elements.retryProcessingButton.addEventListener("click", retrySelectedProcessing);
  elements.discardRetainedAudioButton.addEventListener(
    "click",
    discardSelectedRetainedAudio,
  );
  for (const button of elements.recapTabs.querySelectorAll("[data-recap-tab]")) {
    button.addEventListener("click", () => selectRecapTab(button.dataset.recapTab));
  }
  elements.showOriginalButton.addEventListener("click", () => {
    state.generatedLanguage = "original";
    renderGeneratedTab();
  });
  elements.showEnglishButton.addEventListener("click", () => {
    state.generatedLanguage = "translated";
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
  if (JAMIE_IMPORT_UI_ENABLED) {
    elements.chooseJamieExportButton.addEventListener("click", () => {
      void openJamieImport("choose_jamie_export");
    });
    elements.resumeJamieImportButton.addEventListener("click", () => {
      void openJamieImport("resume_jamie_import");
    });
    elements.jamieUseSourceNamesButton.addEventListener("click", useJamieSourceNames);
    elements.jamieExcludeInvalidButton.addEventListener(
      "click",
      excludeUnreadableJamieMeetings,
    );
    elements.jamieIdentitySearch.addEventListener("input", renderJamieIdentityList);
    elements.jamieIdentityNeedsReview.addEventListener(
      "change",
      renderJamieIdentityList,
    );
    elements.jamieMeetingSearch.addEventListener("input", renderJamieMeetingList);
    elements.jamieMeetingIssuesOnly.addEventListener(
      "change",
      renderJamieMeetingList,
    );
    elements.jamieImportButton.addEventListener("click", runJamieImport);
  }
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
  elements.preferredLanguage.addEventListener("change", () => {
    elements.noTranslationLanguages.value = parseNoTranslationLanguages(
      elements.noTranslationLanguages.value,
      elements.preferredLanguage.value,
    ).join(", ");
  });
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
  elements.recapStatusDismiss.addEventListener("click", () => {
    const job = recapJob(state.selectedSessionId);
    if (job?.status !== "error") return;
    state.recapJobs.delete(state.selectedSessionId);
    renderSessions();
    updateContentVisibility();
  });
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
    if (state.sessions.length && !state.recording) await selectSession(state.sessions[0].id);
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

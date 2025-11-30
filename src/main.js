const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const notesEl = document.getElementById("notes");
const apiInput = document.getElementById("apiBase");
const speakersListEl = document.getElementById("speakersList");
const refreshSpeakersBtn = document.getElementById("refreshSpeakers");

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const activeRuns = new Set();
const progressCounts = new Map();
let pollTimer = null;

// Default API base to local dev port.
if (!apiInput.value) {
  apiInput.value = "http://localhost:8787";
}

function setStatus(text) {
  statusEl.textContent = text;
}

// Placeholder to show where transcripts would land.
function appendNote(text) {
  notesEl.value += `${new Date().toLocaleTimeString()} — ${text}\n`;
}

async function startRecording() {
  appendNote("Start clicked");
  setStatus("Starting…");
  startBtn.disabled = true;
  try {
    await invoke("start_recording");
    setStatus("Recording");
    stopBtn.disabled = false;
  } catch (err) {
    console.error("start_recording error", err);
    appendNote(`Start error: ${err}`);
    setStatus("Failed to start: " + err);
    startBtn.disabled = false;
  }
}

async function stopRecording() {
  appendNote("Stop clicked");
  setStatus("Stopping…");
  try {
    const path = await invoke("stop_recording");
    setStatus(`Stopped. Saved at ${path}. Processing…`);
    // Kick off async transcription (non-blocking) and get run id.
    const runId = await invoke("transcribe_file_async", { path, apiBase: apiInput.value });
    appendNote(`[${runId.slice(0, 8)}] queued`);
    // Track run and start polling fallback.
    activeRuns.add(runId);
    progressCounts.set(runId, 0);
    ensurePolling();
  } catch (err) {
    console.error("stop_recording error", err);
    appendNote(`Stop error: ${err}`);
    setStatus("Failed to stop: " + err);
  }
  stopBtn.disabled = true;
  // Keep start enabled for another recording while processing happens.
  startBtn.disabled = false;
}

async function sendToApi(path) {
  const apiBase = apiInput.value || "http://localhost:8787";
  try {
    const result = await invoke("transcribe_file", { path, apiBase });
    appendNote(result);
  } catch (err) {
    console.error(err);
    appendNote("API error: " + err);
  }
}

startBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);
refreshSpeakersBtn.addEventListener("click", loadSpeakers);

listen("recording:start", () => {
  setStatus("Recording (tray)");
  startBtn.disabled = true;
  stopBtn.disabled = false;
});

listen("recording:stop", () => {
  setStatus("Stopped (tray)");
  stopBtn.disabled = true;
  startBtn.disabled = false;
});

// Listen for backend progress events from async transcription.
(async () => {
  appendNote("Progress listener ready.");
  console.log("Progress listener ready.");
  const unlisten = await listen("transcription:progress", (event) => {
    const { stage, detail, run_id } = event.payload;
    const prefix = run_id ? `[${run_id.slice(0, 8)}] ` : "";
    const message = detail ? `${stage}: ${detail}` : stage;
    console.log("progress", prefix + message);
    appendNote(prefix + message);

    switch (stage) {
      case "complete":
        appendNote(prefix + `Transcript: ${detail ?? ""}`);
        setStatus(detail && detail !== "failed" ? prefix + "Done" : prefix + "Failed");
        startBtn.disabled = false;
        stopBtn.disabled = true;
        if (run_id) activeRuns.delete(run_id);
        break;
      case "error":
        setStatus(`Error: ${detail || "unknown"}`);
        startBtn.disabled = false;
        stopBtn.disabled = true;
        if (run_id) activeRuns.delete(run_id);
        break;
      case "queued":
        setStatus(prefix + message);
        break;
      default:
        setStatus(message);
        break;
    }
  });
  // Optional: window.addEventListener("beforeunload", unlisten);
})();

async function pollProgressAll() {
  for (const runId of Array.from(activeRuns)) {
    try {
      const events = await invoke("get_progress", { runId });
      const lastCount = progressCounts.get(runId) || 0;
      if (events && events.length > lastCount) {
        for (let i = lastCount; i < events.length; i++) {
          const ev = events[i];
          const prefix = ev.run_id ? `[${ev.run_id.slice(0, 8)}] ` : "";
          const message = ev.detail ? `${ev.stage}: ${ev.detail}` : ev.stage;
          appendNote(prefix + message);
          if (ev.stage === "complete" || ev.stage === "error") {
            activeRuns.delete(runId);
          }
        }
        progressCounts.set(runId, events.length);
      }
    } catch (e) {
      console.error("pollProgress error", e);
    }
  }
  if (activeRuns.size === 0 && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function ensurePolling() {
  if (pollTimer) return;
  pollTimer = setInterval(pollProgressAll, 2000);
}

async function loadSpeakers() {
  try {
    const speakers = await invoke("list_speakers_with_stats");
    renderSpeakers(speakers);
  } catch (e) {
    console.error("loadSpeakers error", e);
    appendNote(`Speakers load error: ${e}`);
  }
}

function renderSpeakers(speakers) {
  speakersListEl.innerHTML = "";
  speakers.forEach((sp) => {
    const card = document.createElement("div");
    card.className = "speaker-card";
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = sp.label || sp.id || "Unnamed";
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `Samples: ${sp.sample_count} • Embeddings: ${sp.embedding_count}`;

    const actions = document.createElement("div");
    actions.className = "actions";

    const previewBtn = document.createElement("button");
    previewBtn.textContent = "Preview";
    previewBtn.onclick = () => previewSample(sp.id);

    const renameBtn = document.createElement("button");
    renameBtn.textContent = "Rename";
    renameBtn.onclick = () => renameSpeakerPrompt(sp.id, sp.label);

    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "Delete";
    deleteBtn.onclick = () => deleteSpeaker(sp.id);

    const mergeBtn = document.createElement("button");
    mergeBtn.textContent = "Merge…";
    mergeBtn.onclick = () => mergeSpeakerPrompt(sp.id);

    actions.append(previewBtn, renameBtn, mergeBtn, deleteBtn);
    card.append(title, meta, actions);
    speakersListEl.append(card);
  });
}

async function previewSample(speakerId) {
  try {
    const samples = await invoke("get_speaker_samples", { speakerId });
    if (!samples.length) {
      appendNote(`[${speakerId.slice(0, 8)}] No samples`);
      return;
    }
    const sample = samples[0];
    const audio = new Audio(`data:audio/wav;base64,${sample.sample_b64}`);
    audio.play();
    appendNote(`[${speakerId.slice(0, 8)}] Playing sample`);
  } catch (e) {
    console.error("previewSample error", e);
    appendNote(`Preview error: ${e}`);
  }
}

async function renameSpeakerPrompt(speakerId, currentLabel) {
  const name = prompt("New name for speaker", currentLabel || "");
  if (!name) return;
  try {
    await invoke("rename_speaker", { speakerId, newLabel: name });
    appendNote(`[${speakerId.slice(0, 8)}] Renamed to ${name}`);
    await loadSpeakers();
  } catch (e) {
    console.error("rename error", e);
    appendNote(`Rename error: ${e}`);
  }
}

async function deleteSpeaker(speakerId) {
  if (!confirm("Delete this speaker and its data?")) return;
  try {
    await invoke("delete_speaker", { speakerId });
    appendNote(`[${speakerId.slice(0, 8)}] Deleted`);
    await loadSpeakers();
  } catch (e) {
    console.error("delete error", e);
    appendNote(`Delete error: ${e}`);
  }
}

async function mergeSpeakerPrompt(sourceId) {
  const targetId = prompt("Merge into speaker ID:");
  if (!targetId || targetId === sourceId) return;
  const replace = confirm("Replace target embeddings with source?");
  try {
    await invoke("merge_speakers", {
      targetId,
      sourceId,
      replaceEmbeddings: replace,
    });
    appendNote(`[${sourceId.slice(0, 8)}] merged into ${targetId.slice(0, 8)}`);
    await loadSpeakers();
  } catch (e) {
    console.error("merge error", e);
    appendNote(`Merge error: ${e}`);
  }
}

loadSpeakers();
appendNote("Ready.");

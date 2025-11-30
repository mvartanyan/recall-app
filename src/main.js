const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const notesEl = document.getElementById("notes");
const apiInput = document.getElementById("apiBase");

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

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
    startBtn.disabled = true;
    // Kick off async transcription (non-blocking).
    await invoke("transcribe_file_async", { path, apiBase: apiInput.value });
  } catch (err) {
    console.error("stop_recording error", err);
    appendNote(`Stop error: ${err}`);
    setStatus("Failed to stop: " + err);
  }
  stopBtn.disabled = true;
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
listen("transcription:progress", (event) => {
  const { stage, detail } = event.payload;
  const message = detail ? `${stage}: ${detail}` : stage;
  appendNote(message);
  if (stage === "complete") {
    setStatus("Done");
    startBtn.disabled = false;
    stopBtn.disabled = true;
  } else {
    setStatus(message);
  }
});

appendNote("Ready.");

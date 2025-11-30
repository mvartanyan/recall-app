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
    // Kick off async transcription (non-blocking) and get run id.
    const runId = await invoke("transcribe_file_async", { path, apiBase: apiInput.value });
    appendNote(`[${runId.slice(0, 8)}] queued`);
    // Poll for progress in case live events drop.
    pollProgress(runId);
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
        break;
      case "error":
        setStatus(`Error: ${detail || "unknown"}`);
        startBtn.disabled = false;
        stopBtn.disabled = true;
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

async function pollProgress(runId) {
  try {
    const events = await invoke("get_progress", { runId });
    if (events && events.length) {
      for (const ev of events) {
        const prefix = ev.run_id ? `[${ev.run_id.slice(0, 8)}] ` : "";
        const message = ev.detail ? `${ev.stage}: ${ev.detail}` : ev.stage;
        appendNote(prefix + message);
      }
    }
  } catch (e) {
    console.error("pollProgress error", e);
  }
}

appendNote("Ready.");

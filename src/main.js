import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const notesEl = document.getElementById("notes");
const apiInput = document.getElementById("apiBase");

function setStatus(text) {
  statusEl.textContent = text;
}

async function startRecording() {
  setStatus("Starting…");
  startBtn.disabled = true;
  try {
    await invoke("start_recording");
    setStatus("Recording");
    stopBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus("Failed to start: " + err);
    startBtn.disabled = false;
  }
}

async function stopRecording() {
  setStatus("Stopping…");
  try {
    const path = await invoke("stop_recording");
    setStatus(`Stopped. Saved at ${path}`);
    await sendToApi(path);
  } catch (err) {
    console.error(err);
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

// Placeholder to show where transcripts would land.
function appendNote(text) {
  notesEl.value += `${new Date().toLocaleTimeString()} — ${text}\n`;
}

appendNote("Ready.");

export function isProvisionalLabel(label) {
  return /^VOICE\d+$/i.test((label || "").trim());
}

export function formatDuration(milliseconds) {
  const totalSeconds = Math.max(0, Math.round((Number(milliseconds) || 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return String(hours) + "h " + String(minutes).padStart(2, "0") + "m";
  }
  if (minutes > 0) {
    return String(minutes) + "m " + String(seconds).padStart(2, "0") + "s";
  }
  return String(seconds) + "s";
}

export function formatTimestamp(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor((Number(milliseconds) || 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return [hours, minutes, seconds].map((part) => String(part).padStart(2, "0")).join(":");
  }
  return [minutes, seconds].map((part) => String(part).padStart(2, "0")).join(":");
}

export function transcriptFromSegments(segments) {
  return (segments || [])
    .filter((segment) => (segment.text || "").trim())
    .map((segment) => String(segment.speaker_label || "Unknown speaker") + ": " + segment.text.trim())
    .join("\n");
}

export function parseLanguageHints(value) {
  const seen = new Set();
  return String(value || "")
    .split(",")
    .map((language) => language.trim().toLowerCase().split("-")[0])
    .filter((language) => language && !seen.has(language) && seen.add(language));
}

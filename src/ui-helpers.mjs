export const ONBOARDING_VERSION = "1";

export function shouldShowOnboarding(storedVersion) {
  return String(storedVersion || "") !== ONBOARDING_VERSION;
}

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

export function parseNoTranslationLanguages(value) {
  return parseLanguageHints(value).filter((language) => language !== "en");
}

export function recapTabAvailability(recapState) {
  const payload = recapState?.recap?.payload;
  const tabs = ["transcript"];
  if (!payload) return tabs;
  tabs.push("executive", "full", "actions");
  if (payload.agenda_present) tabs.push("agenda");
  return tabs;
}

export function buildTranslationPlan(text, annotations) {
  const source = String(text || "");
  const candidates = (annotations || [])
    .map((annotation, order) => ({
      annotation,
      order,
      initialIndex: source.indexOf(String(annotation.source_excerpt || "")),
    }))
    .sort((left, right) => {
      const leftIndex = left.initialIndex < 0 ? Number.MAX_SAFE_INTEGER : left.initialIndex;
      const rightIndex = right.initialIndex < 0 ? Number.MAX_SAFE_INTEGER : right.initialIndex;
      return leftIndex - rightIndex || left.order - right.order;
    });
  const chunks = [];
  const fallbacks = [];
  let cursor = 0;
  for (const candidate of candidates) {
    const excerpt = String(candidate.annotation.source_excerpt || "");
    if (!excerpt) {
      fallbacks.push(candidate.annotation);
      continue;
    }
    const index = source.indexOf(excerpt, cursor);
    if (index < cursor || index < 0) {
      fallbacks.push(candidate.annotation);
      continue;
    }
    if (index > cursor) chunks.push({ source: source.slice(cursor, index), translation: null });
    chunks.push({
      source: excerpt,
      translation: String(candidate.annotation.english_translation || ""),
      language: String(candidate.annotation.language || ""),
    });
    cursor = index + excerpt.length;
  }
  if (cursor < source.length || !chunks.length) {
    chunks.push({ source: source.slice(cursor), translation: null });
  }
  return { chunks, fallbacks };
}

export function translatedSegmentText(text, annotations) {
  const plan = buildTranslationPlan(text, annotations);
  let rendered = plan.chunks
    .map((chunk) => {
      if (!chunk.translation) return chunk.source;
      return chunk.source + " (TRANSLATION: " + chunk.translation + ")";
    })
    .join("");
  for (const fallback of plan.fallbacks) {
    rendered +=
      (rendered ? "\n" : "") +
      "(TRANSLATION: " +
      String(fallback.english_translation || "") +
      ")";
  }
  return rendered;
}

export function isNearScrollBottom(
  { scrollTop = 0, clientHeight = 0, scrollHeight = 0 } = {},
  threshold = 32,
) {
  return Number(scrollHeight) - (Number(scrollTop) + Number(clientHeight)) <= threshold;
}

export function filterSessions(sessions, query, allowedSessionIds = null) {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  return (sessions || []).filter((session) => {
    if (allowedSessionIds && !allowedSessionIds.has(session.id)) return false;
    const title = String(session.title || "Untitled conversation");
    const searchable = (title + " " + String(session.transcript || "")).toLowerCase();
    return !normalizedQuery || searchable.includes(normalizedQuery);
  });
}

export function contentMode({
  recording = false,
  queueing = false,
  processingCount = 0,
  selectedSessionId = null,
} = {}) {
  if (recording) return "recording";
  if (queueing || Number(processingCount) > 0) return "processing";
  if (selectedSessionId) return "conversation";
  return "empty";
}

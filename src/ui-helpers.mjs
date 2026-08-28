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

export function normalizePreferredLanguage(value, fallback = "en") {
  return parseLanguageHints(value)[0] || fallback;
}

export function normalizeLiveCaptionLanguage(value) {
  const language = String(value || "")
    .trim()
    .toLowerCase()
    .replaceAll("_", "-")
    .split("-")[0];
  return /^[a-z]{2,3}$/.test(language) ? language : "";
}

export function normalizeLiveCaptionRevision(value) {
  const revision = Number(value);
  return Number.isSafeInteger(revision) && revision > 0 ? revision : 0;
}

export function isNewerLiveCaptionRevision(lastRevision, incomingRevision) {
  const incoming = normalizeLiveCaptionRevision(incomingRevision);
  return incoming === 0 || incoming > normalizeLiveCaptionRevision(lastRevision);
}

export const LIVE_CAPTION_LANGUAGE_PALETTE = [
  { background: "#edf7f2", foreground: "#2d6957" },
  { background: "#edf4f8", foreground: "#37677f" },
  { background: "#f8f2e9", foreground: "#86622d" },
  { background: "#f5eef7", foreground: "#785776" },
  { background: "#faefed", foreground: "#8a5148" },
  { background: "#eaf7f6", foreground: "#2f6f6b" },
  { background: "#eef0fa", foreground: "#4d5f8c" },
  { background: "#f4f6e9", foreground: "#66713d" },
];

export function liveCaptionLanguageStyle(slot) {
  const normalizedSlot = Math.max(0, Math.trunc(Number(slot) || 0));
  const paletteStyle = LIVE_CAPTION_LANGUAGE_PALETTE[normalizedSlot];
  if (paletteStyle) {
    return {
      background: paletteStyle.background,
      foreground: paletteStyle.foreground,
      border: paletteStyle.foreground,
    };
  }
  const hue = (normalizedSlot * 137.508 + 150) % 360;
  const hueText = Number(hue.toFixed(3));
  return {
    background: `hsl(${hueText} 38% 95%)`,
    foreground: `hsl(${hueText} 35% 33%)`,
    border: `hsl(${hueText} 35% 33%)`,
  };
}

export function normalizeLiveCaptionPassages(passages) {
  const seen = new Set();
  return (passages || [])
    .map((passage, index) => {
      const id = String(passage?.id || "live-passage-" + index).trim();
      const sourceText = String(passage?.source_text || "").trim();
      const translationText = String(passage?.translation?.text || "").trim();
      const sourceLanguage = normalizeLiveCaptionLanguage(passage?.source_language);
      const translationSourceLanguage = normalizeLiveCaptionLanguage(
        passage?.translation?.source_language,
      );
      return {
        id,
        sequence: Number.isFinite(Number(passage?.sequence))
          ? Number(passage.sequence)
          : index,
        order: index,
        speaker: String(passage?.speaker || "").trim(),
        sourceText,
        sourceLanguage,
        sourceFinal: Boolean(passage?.source_final),
        translation:
          translationText && sourceLanguage && translationSourceLanguage === sourceLanguage
            ? {
                text: translationText,
                sourceLanguage,
                final: Boolean(passage.translation?.is_final),
              }
            : null,
      };
    })
    .filter((passage) => passage.id && passage.sourceText && !seen.has(passage.id) && seen.add(passage.id))
    .sort((left, right) => left.sequence - right.sequence || left.order - right.order);
}

function joinLiveCaptionText(left, right) {
  const first = String(left || "").trim();
  const second = String(right || "").trim();
  if (!first) return second;
  if (!second) return first;
  return first + " " + second;
}

function normalizeLiveCaptionSegment(segment, index) {
  const id = String(segment?.id || "live-segment-" + index).trim();
  const sourceText = String(segment?.source_text || "").trim();
  const sourceLanguage = normalizeLiveCaptionLanguage(segment?.source_language);
  const translationText = String(segment?.translation?.text || "").trim();
  const translationSourceLanguage = normalizeLiveCaptionLanguage(
    segment?.translation?.source_language,
  );
  return {
    id,
    order: index,
    sourceText,
    sourceLanguage,
    sourceFinal: Boolean(segment?.source_final),
    translation:
      translationText && sourceLanguage && translationSourceLanguage === sourceLanguage
        ? {
            text: translationText,
            sourceLanguage,
            final: Boolean(segment.translation?.is_final),
          }
        : null,
  };
}

export function normalizeLiveCaptionTurns(turns) {
  const seen = new Set();
  return (turns || [])
    .map((turn, index) => {
      const id = String(turn?.id || "live-turn-" + index).trim();
      const segments = (turn?.segments || [])
        .map(normalizeLiveCaptionSegment)
        .filter((segment) => segment.id && segment.sourceText);
      return {
        id,
        sequence: Number.isFinite(Number(turn?.sequence)) ? Number(turn.sequence) : index,
        order: index,
        speaker: String(turn?.speaker || "").trim(),
        segments,
      };
    })
    .filter((turn) => turn.id && turn.segments.length && !seen.has(turn.id) && seen.add(turn.id))
    .sort((left, right) => left.sequence - right.sequence || left.order - right.order);
}

export function liveCaptionTurnsFromPassages(passages) {
  return normalizeLiveCaptionPassages(passages).map((passage) => ({
    id: passage.id,
    sequence: passage.sequence,
    speaker: passage.speaker,
    segments: [
      {
        id: passage.id + "-segment",
        source_text: passage.sourceText,
        source_language: passage.sourceLanguage,
        source_final: passage.sourceFinal,
        translation: passage.translation
          ? {
              text: passage.translation.text,
              source_language: passage.translation.sourceLanguage,
              is_final: passage.translation.final,
            }
          : null,
      },
    ],
  }));
}

export function buildLiveCaptionDisplayRuns(turn) {
  const appendRun = (runs, language, text, translated) => {
    const last = runs.at(-1);
    if (last && last.language === language) {
      last.text = joinLiveCaptionText(last.text, text);
      last.translated = last.translated || translated;
      return;
    }
    runs.push({ language, text: String(text || "").trim(), translated: Boolean(translated) });
  };
  const sourceRuns = [];
  const preferredRuns = [];
  let hasTranslation = false;
  for (const segment of turn?.segments || []) {
    const language = normalizeLiveCaptionLanguage(segment?.sourceLanguage || segment?.source_language);
    const sourceText = String(segment?.sourceText || segment?.source_text || "").trim();
    if (!sourceText) continue;
    const translation = segment?.translation;
    const translationText = String(translation?.text || "").trim();
    const translationLanguage = normalizeLiveCaptionLanguage(
      translation?.sourceLanguage || translation?.source_language,
    );
    const safelyTranslated = Boolean(translationText && language && translationLanguage === language);
    appendRun(sourceRuns, language, sourceText, false);
    appendRun(preferredRuns, language, safelyTranslated ? translationText : sourceText, safelyTranslated);
    hasTranslation = hasTranslation || safelyTranslated;
  }
  return {
    sourceRuns,
    preferredRuns: hasTranslation ? preferredRuns : [],
    hasTranslation,
  };
}

export function parseNoTranslationLanguages(value, preferredLanguage = "en") {
  const preferred = normalizePreferredLanguage(preferredLanguage);
  return parseLanguageHints(value).filter((language) => language !== preferred);
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
      translation: String(
        candidate.annotation.translated_text ||
          candidate.annotation.english_translation ||
          "",
      ),
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
      String(fallback.translated_text || fallback.english_translation || "") +
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

export function filterSessions(
  sessions,
  query,
  allowedSessionIds = null,
  transcriptMatchIds = null,
) {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  return (sessions || []).filter((session) => {
    if (allowedSessionIds && !allowedSessionIds.has(session.id)) return false;
    const title = String(session.title || "Untitled conversation");
    const searchable = (title + " " + String(session.transcript || "")).toLowerCase();
    return (
      !normalizedQuery ||
      searchable.includes(normalizedQuery) ||
      Boolean(transcriptMatchIds?.has(session.id))
    );
  });
}

export function indexTranslations(translations) {
  const index = new Map();
  for (const translation of translations || []) {
    const segmentId = String(translation?.segment_id || "");
    if (!segmentId) continue;
    const entries = index.get(segmentId);
    if (entries) entries.push(translation);
    else index.set(segmentId, [translation]);
  }
  return index;
}

export function getCachedConversation(cache, sessionId) {
  if (!cache?.has(sessionId)) return null;
  const payload = cache.get(sessionId);
  cache.delete(sessionId);
  cache.set(sessionId, payload);
  return payload;
}

export function setCachedConversation(cache, sessionId, payload, limit = 5) {
  if (!cache || !sessionId) return payload;
  cache.delete(sessionId);
  cache.set(sessionId, payload);
  const boundedLimit = Math.max(1, Number(limit) || 1);
  while (cache.size > boundedLimit) {
    cache.delete(cache.keys().next().value);
  }
  return payload;
}

export function invalidateConversationCache(cache, sessionId = null) {
  if (!cache) return;
  if (sessionId) cache.delete(sessionId);
  else cache.clear();
}

export function nextRenderedSegmentCount(
  total,
  current = 0,
  batchSize = 100,
  requiredIndex = null,
) {
  const boundedTotal = Math.max(0, Number(total) || 0);
  const boundedBatch = Math.max(1, Number(batchSize) || 1);
  let desired = Math.max(Number(current) || 0, boundedBatch);
  if (Number.isInteger(requiredIndex) && requiredIndex >= 0) {
    desired = Math.max(
      desired,
      Math.ceil((requiredIndex + 1) / boundedBatch) * boundedBatch,
    );
  }
  return Math.min(boundedTotal, desired);
}

export function groupVoiceFilters(speakers) {
  const groups = new Map();
  for (const speaker of speakers || []) {
    if (!(Number(speaker.conversation_count) > 0)) continue;
    const label = String(speaker.label || "Unnamed voice").trim() || "Unnamed voice";
    if (
      isProvisionalLabel(label) ||
      /^(?:unnamed voice|unknown speaker)$/i.test(label)
    ) {
      continue;
    }
    const key = label.toLocaleLowerCase();
    const existing = groups.get(key);
    if (existing) {
      existing.speakerIds.push(speaker.id);
      continue;
    }
    groups.set(key, {
      key,
      label,
      speakerIds: [speaker.id],
    });
  }
  return Array.from(groups.values()).sort((left, right) =>
    left.label.localeCompare(right.label, undefined, {
      sensitivity: "base",
      numeric: true,
    }),
  );
}

export function isSessionProcessing(session) {
  return ["queued", "processing"].includes(String(session?.processing_status || ""));
}

export function processingRunIds(sessions) {
  return new Set(
    (sessions || [])
      .filter((session) => isSessionProcessing(session) && session.processing_run_id)
      .map((session) => session.processing_run_id),
  );
}

export function contentMode({
  recording = false,
  recordingViewSelected = recording,
  queueing = false,
  processingCount = 0,
  selectedSessionId = null,
} = {}) {
  if (recording && recordingViewSelected) return "recording";
  if (selectedSessionId) return "conversation";
  if (queueing) return "processing";
  if (Number(processingCount) > 0) return "processing";
  return "empty";
}

import test from "node:test";
import assert from "node:assert/strict";

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
  translatedSegmentText,
  transcriptFromSegments,
} from "./ui-helpers.mjs";

test("onboarding is shown until the current copy version is acknowledged", () => {
  assert.equal(shouldShowOnboarding(null), true);
  assert.equal(shouldShowOnboarding(""), true);
  assert.equal(shouldShowOnboarding("0"), true);
  assert.equal(shouldShowOnboarding(ONBOARDING_VERSION), false);
});

test("the selected workspace remains usable while recording and processing continue", () => {
  assert.equal(contentMode(), "empty");
  assert.equal(contentMode({ selectedSessionId: "old" }), "conversation");
  assert.equal(
    contentMode({ selectedSessionId: "old", processingCount: 1 }),
    "conversation",
  );
  assert.equal(
    contentMode({ selectedSessionId: "old", processingCount: 2, recording: true }),
    "recording",
  );
  assert.equal(
    contentMode({
      selectedSessionId: "old",
      processingCount: 2,
      recording: true,
      recordingViewSelected: false,
    }),
    "conversation",
  );
  assert.equal(contentMode({ selectedSessionId: "old", queueing: true }), "conversation");
  assert.equal(contentMode({ processingCount: 1 }), "processing");
});

test("VOICE labels are provisional but human names are not", () => {
  assert.equal(isProvisionalLabel("VOICE12"), true);
  assert.equal(isProvisionalLabel("Michael"), false);
});

test("durations and transcript timestamps are readable", () => {
  assert.equal(formatDuration(65_000), "1m 05s");
  assert.equal(formatTimestamp(3_725_000), "01:02:05");
});

test("language hints are normalized and deduplicated", () => {
  assert.deepEqual(parseLanguageHints("en-US, de-DE, en, ru"), ["en", "de", "ru"]);
});

test("translation exclusions normalize codes and omit the preferred language", () => {
  assert.deepEqual(parseNoTranslationLanguages("EN-us, de-DE, fr, de"), ["de", "fr"]);
  assert.deepEqual(parseNoTranslationLanguages("de-DE, fr, en", "de"), ["fr", "en"]);
  assert.equal(normalizePreferredLanguage("DE-de"), "de");
});

test("a persisted recap always exposes its generated tabs", () => {
  assert.deepEqual(recapTabAvailability(null), ["transcript"]);
  assert.deepEqual(
    recapTabAvailability({ recap: { payload: { agenda_present: false } } }),
    ["transcript", "executive", "full", "actions"],
  );
  assert.deepEqual(
    recapTabAvailability({ recap: { payload: { agenda_present: true } } }),
    ["transcript", "executive", "full", "actions", "agenda"],
  );
});

test("translation annotations are inserted after exact non-overlapping excerpts", () => {
  const plan = buildTranslationPlan("Bonjour, Michael. Danke.", [
    {
      source_excerpt: "Danke.",
      english_translation: "Thank you.",
      language: "de",
    },
    {
      source_excerpt: "Bonjour",
      english_translation: "Hello",
      language: "fr",
    },
  ]);
  assert.equal(plan.fallbacks.length, 0);
  assert.equal(
    translatedSegmentText("Bonjour, Michael. Danke.", [
      {
        source_excerpt: "Bonjour",
        english_translation: "Hello",
        language: "fr",
      },
      {
        source_excerpt: "Danke.",
        english_translation: "Thank you.",
        language: "de",
      },
    ]),
    "Bonjour (TRANSLATION: Hello), Michael. Danke. (TRANSLATION: Thank you.)",
  );
});

test("unanchored translations fall back beneath the intervention", () => {
  const plan = buildTranslationPlan("Bonjour", [
    {
      source_excerpt: "Bonsoir",
      english_translation: "Good evening",
      language: "fr",
    },
  ]);
  assert.equal(plan.fallbacks.length, 1);
  assert.equal(
    translatedSegmentText("Bonjour", plan.fallbacks),
    "Bonjour\n(TRANSLATION: Good evening)",
  );
});

test("speaker-attributed transcript is rebuilt from interventions", () => {
  assert.equal(
    transcriptFromSegments([
      { speaker_label: "Ada", text: "Hello." },
      { speaker_label: null, text: "Hi." },
    ]),
    "Ada: Hello.\nUnknown speaker: Hi.",
  );
});

test("live captions only follow while the viewport is near the bottom", () => {
  assert.equal(
    isNearScrollBottom({ scrollTop: 660, clientHeight: 310, scrollHeight: 1_000 }),
    true,
  );
  assert.equal(
    isNearScrollBottom({ scrollTop: 400, clientHeight: 310, scrollHeight: 1_000 }),
    false,
  );
});

test("conversation filtering combines text and selected voice", () => {
  const sessions = [
    { id: "one", title: "Planning", transcript: "Ada: Hello" },
    { id: "two", title: "Review", transcript: "Grace: Status" },
  ];
  assert.deepEqual(
    filterSessions(sessions, "status", new Set(["two"])).map((session) => session.id),
    ["two"],
  );
  assert.deepEqual(filterSessions(sessions, "planning", new Set(["two"])), []);
});

test("conversation filtering accepts backend transcript matches for metadata-only rows", () => {
  const sessions = [
    { id: "one", title: "Planning" },
    { id: "two", title: "Review" },
  ];
  assert.deepEqual(
    filterSessions(sessions, "spoken phrase", null, new Set(["two"])).map(
      (session) => session.id,
    ),
    ["two"],
  );
});

test("translations are indexed once by segment", () => {
  const first = { segment_id: "one", translated_text: "Hello" };
  const second = { segment_id: "one", translated_text: "Again" };
  const index = indexTranslations([first, second, { segment_id: "", translated_text: "Skip" }]);
  assert.deepEqual(index.get("one"), [first, second]);
  assert.equal(index.size, 1);
});

test("conversation cache is bounded, recent, and explicitly invalidated", () => {
  const cache = new Map();
  setCachedConversation(cache, "one", { id: 1 }, 2);
  setCachedConversation(cache, "two", { id: 2 }, 2);
  assert.deepEqual(getCachedConversation(cache, "one"), { id: 1 });
  setCachedConversation(cache, "three", { id: 3 }, 2);
  assert.equal(cache.has("two"), false);
  assert.equal(cache.has("one"), true);
  invalidateConversationCache(cache, "one");
  assert.equal(cache.has("one"), false);
  invalidateConversationCache(cache);
  assert.equal(cache.size, 0);
});

test("progressive transcript rendering grows by batches and can reveal a required row", () => {
  assert.equal(nextRenderedSegmentCount(2_163), 100);
  assert.equal(nextRenderedSegmentCount(2_163, 100), 100);
  assert.equal(nextRenderedSegmentCount(2_163, 200, 100), 200);
  assert.equal(nextRenderedSegmentCount(2_163, 100, 100, 721), 800);
  assert.equal(nextRenderedSegmentCount(45, 0, 100), 45);
});

test("voice filters collapse named people and omit provisional or unknown profiles", () => {
  assert.deepEqual(
    groupVoiceFilters([
      { id: "z", label: "Zoë", conversation_count: 1 },
      { id: "a1", label: "Alice", conversation_count: 2 },
      { id: "unused", label: "Nobody", conversation_count: 0 },
      { id: "a2", label: "alice", conversation_count: 1 },
      { id: "v2", label: "VOICE10", conversation_count: 1 },
      { id: "v1", label: "VOICE2", conversation_count: 1 },
      { id: "v3", label: " voice3 ", conversation_count: 1 },
      { id: "unknown", label: "Unknown speaker", conversation_count: 1 },
      { id: "blank", label: "   ", conversation_count: 1 },
      { id: "unused", label: "Bob", conversation_count: 0 },
    ]),
    [
      { key: "alice", label: "Alice", speakerIds: ["a1", "a2"] },
      { key: "zoë", label: "Zoë", speakerIds: ["z"] },
    ],
  );
});

test("processing state is derived from persisted conversations, not unrelated stale runs", () => {
  const sessions = [
    { id: "done", processing_status: null, processing_run_id: null },
    { id: "active", processing_status: "processing", processing_run_id: "run-active" },
    { id: "failed", processing_status: "failed", processing_run_id: "run-failed" },
  ];
  assert.equal(isSessionProcessing(sessions[0]), false);
  assert.equal(isSessionProcessing(sessions[1]), true);
  assert.equal(isSessionProcessing(sessions[2]), false);
  assert.deepEqual(Array.from(processingRunIds(sessions)), ["run-active"]);
});

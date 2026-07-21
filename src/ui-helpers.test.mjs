import test from "node:test";
import assert from "node:assert/strict";

import {
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
  translatedSegmentText,
  transcriptFromSegments,
} from "./ui-helpers.mjs";

test("workspace modes are mutually exclusive and recording takes priority", () => {
  assert.equal(contentMode(), "empty");
  assert.equal(contentMode({ selectedSessionId: "old" }), "conversation");
  assert.equal(
    contentMode({ selectedSessionId: "old", processingCount: 1 }),
    "processing",
  );
  assert.equal(
    contentMode({ selectedSessionId: "old", processingCount: 2, recording: true }),
    "recording",
  );
  assert.equal(contentMode({ selectedSessionId: "old", queueing: true }), "processing");
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

test("translation exclusions normalize language codes and omit implicit English", () => {
  assert.deepEqual(parseNoTranslationLanguages("EN-us, de-DE, fr, de"), ["de", "fr"]);
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

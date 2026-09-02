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
  isNewerLiveCaptionRevision,
  isProvisionalLabel,
  isSessionProcessing,
  nextRenderedSegmentCount,
  LIVE_CAPTION_LANGUAGE_PALETTE,
  liveCaptionLanguageStyle,
  buildLiveCaptionDisplayRuns,
  normalizeLiveCaptionLanguage,
  normalizeLiveCaptionPassages,
  normalizeLiveCaptionRevision,
  normalizeLiveCaptionTurns,
  normalizePreferredLanguage,
  normalizeRecapPromptVariables,
  normalizeRecapTypeName,
  parseLanguageHints,
  parseNoTranslationLanguages,
  parseSafeMarkdown,
  processingRunIds,
  recapTabAvailability,
  recapTypeNameLength,
  insertRecapPromptVariable,
  safeMarkdownPlainText,
  setCachedConversation,
  shouldShowOnboarding,
  sortCustomRecaps,
  sortRecapTypes,
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

test("live caption revisions reject stale event or polling snapshots", () => {
  assert.equal(normalizeLiveCaptionRevision("12"), 12);
  assert.equal(normalizeLiveCaptionRevision(0), 0);
  assert.equal(normalizeLiveCaptionRevision("invalid"), 0);
  assert.equal(isNewerLiveCaptionRevision(12, 13), true);
  assert.equal(isNewerLiveCaptionRevision(12, 12), false);
  assert.equal(isNewerLiveCaptionRevision(12, 11), false);
  assert.equal(isNewerLiveCaptionRevision(12, undefined), true);
});

test("translation exclusions normalize codes and omit the preferred language", () => {
  assert.deepEqual(parseNoTranslationLanguages("EN-us, de-DE, fr, de"), ["de", "fr"]);
  assert.deepEqual(parseNoTranslationLanguages("de-DE, fr, en", "de"), ["fr", "en"]);
  assert.equal(normalizePreferredLanguage("DE-de"), "de");
});

test("live caption passages stay ordered, keyed, and safely tagged", () => {
  const passages = normalizeLiveCaptionPassages([
    {
      id: "late",
      sequence: 3,
      speaker: "Speaker 2",
      source_text: "Bonjour",
      source_language: "fr-FR",
      source_final: false,
      translation: {
        text: "Good morning",
        source_language: "fr-FR",
        is_final: false,
      },
    },
    {
      id: "first",
      sequence: 1,
      speaker: "Speaker 1",
      source_text: "Guten Morgen",
      source_language: "de-DE",
      source_final: true,
      translation: {
        text: "Good morning",
        source_language: "de-DE",
        is_final: true,
      },
    },
    {
      id: "source-only",
      sequence: 2,
      speaker: "Speaker 1",
      source_text: "Hello",
      source_language: "en",
      source_final: false,
      translation: null,
    },
    {
      id: "unsafe-translation",
      sequence: 4,
      speaker: "Speaker 3",
      source_text: "Unknown language",
      source_language: null,
      source_final: false,
      translation: { text: "Translation", source_language: "", is_final: false },
    },
    {
      id: "mismatched-translation",
      sequence: 5,
      speaker: "Speaker 3",
      source_text: "Ciao",
      source_language: "it-IT",
      source_final: false,
      translation: { text: "Hello", source_language: "fr-FR", is_final: false },
    },
  ]);

  assert.deepEqual(
    passages.map((passage) => passage.id),
    ["first", "source-only", "late", "unsafe-translation", "mismatched-translation"],
  );
  assert.equal(passages[0].sourceLanguage, "de");
  assert.equal(passages[0].translation.sourceLanguage, "de");
  assert.equal(passages[1].sourceLanguage, "en");
  assert.equal(passages[1].translation, null);
  assert.equal(passages[3].translation, null);
  assert.equal(passages[4].sourceLanguage, "it");
  assert.equal(passages[4].translation, null);
  assert.equal(normalizeLiveCaptionLanguage("RU_ru"), "ru");
  assert.equal(normalizeLiveCaptionLanguage("not a language"), "");
});

test("live caption turns retain inline code switches and build a complete preferred-language turn", () => {
  const [turn] = normalizeLiveCaptionTurns([
    {
      id: "turn-1",
      sequence: 4,
      speaker: "Speaker 1",
      segments: [
        {
          id: "ru-1",
          source_text: "Привет",
          source_language: "ru-RU",
          source_final: false,
          translation: { text: "Hello", source_language: "ru", is_final: false },
        },
        {
          id: "en-1",
          source_text: "and welcome",
          source_language: "en-US",
          source_final: false,
          translation: null,
        },
        {
          id: "ru-2",
          source_text: "друзья",
          source_language: "ru-RU",
          source_final: false,
          translation: { text: "friends", source_language: "ru", is_final: false },
        },
      ],
    },
  ]);
  assert.equal(turn.speaker, "Speaker 1");
  assert.equal(turn.segments.length, 3);
  const display = buildLiveCaptionDisplayRuns(turn);
  assert.deepEqual(
    display.sourceRuns.map((run) => [run.language, run.text]),
    [["ru", "Привет"], ["en", "and welcome"], ["ru", "друзья"]],
  );
  assert.deepEqual(
    display.preferredRuns.map((run) => [run.language, run.text]),
    [["ru", "Hello"], ["en", "and welcome"], ["ru", "friends"]],
  );
  assert.equal(display.hasTranslation, true);
});

test("live caption preferred-language runs are suppressed without a safely matched translation", () => {
  const [turn] = normalizeLiveCaptionTurns([
    {
      id: "turn-no-translation",
      speaker: "Speaker 1",
      segments: [
        {
          id: "source-only",
          source_text: "Hello",
          source_language: "en",
          source_final: false,
          translation: null,
        },
        {
          id: "unsafe",
          source_text: "Ciao",
          source_language: "it",
          source_final: false,
          translation: { text: "Hello", source_language: "fr", is_final: false },
        },
      ],
    },
  ]);
  const display = buildLiveCaptionDisplayRuns(turn);
  assert.equal(display.hasTranslation, false);
  assert.deepEqual(display.preferredRuns, []);
});

test("live caption language styles have a stable curated palette and deterministic fallback", () => {
  assert.deepEqual(LIVE_CAPTION_LANGUAGE_PALETTE[0], {
    background: "#edf7f2",
    foreground: "#2d6957",
  });
  assert.deepEqual(liveCaptionLanguageStyle(0), {
    background: "#edf7f2",
    foreground: "#2d6957",
    border: "#2d6957",
  });
  assert.deepEqual(liveCaptionLanguageStyle(7), {
    background: "#f4f6e9",
    foreground: "#66713d",
    border: "#66713d",
  });
  assert.deepEqual(liveCaptionLanguageStyle(8), {
    background: "hsl(170.064 38% 95%)",
    foreground: "hsl(170.064 35% 33%)",
    border: "hsl(170.064 35% 33%)",
  });
  assert.deepEqual(liveCaptionLanguageStyle(8), liveCaptionLanguageStyle(8));
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
  assert.deepEqual(
    recapTabAvailability({
      recap: null,
      custom_recaps: [
        { recap_type_id: "risk-z", name: "Risk review" },
        { recap_type_id: "board", name: "Board note" },
        { recap_type_id: "risk-a", name: "Risk review" },
      ],
    }),
    ["transcript", "custom:board", "custom:risk-a", "custom:risk-z"],
  );
});

test("recap types keep built-ins fixed and sort duplicate custom names deterministically", () => {
  const sorted = sortRecapTypes([
    { id: "risk-z", kind: "custom", name: "Risk review" },
    { id: "actions", kind: "actions", name: "Actions" },
    { id: "full", kind: "full_summary", name: "Full summary" },
    { id: "board", kind: "custom", name: "board note" },
    { id: "risk-a", kind: "custom", name: "Risk review" },
    { id: "executive", kind: "executive_summary", name: "Executive summary" },
  ]);
  assert.deepEqual(
    sorted.map((recapType) => recapType.id),
    ["executive", "full", "actions", "board", "risk-a", "risk-z"],
  );
  assert.deepEqual(
    sortCustomRecaps([
      { recap_type_id: "risk-z", name: "Risk review" },
      { recap_type_id: "board", name: "board note" },
      { recap_type_id: "risk-a", name: "Risk review" },
    ]).map((recap) => recap.recap_type_id),
    ["board", "risk-a", "risk-z"],
  );
});

test("custom recap names normalize whitespace and count Unicode code points", () => {
  assert.equal(normalizeRecapTypeName("  Risk\n\t review  "), "Risk review");
  assert.equal(recapTypeNameLength("  🌍 climate  "), 9);
});

test("recap prompt variables are normalized from the native registry", () => {
  assert.deepEqual(
    normalizeRecapPromptVariables([
      {
        token: " {{meeting_date}} ",
        label: " Meeting date ",
        description: " The saved local date. ",
        example: " 2026/09/01 ",
      },
      {
        token: "{{future_variable}}",
        label: "",
        description: null,
        example: null,
      },
      { token: "   ", label: "Ignored" },
    ]),
    [
      {
        token: "{{meeting_date}}",
        label: "Meeting date",
        description: "The saved local date.",
        example: "2026/09/01",
      },
      {
        token: "{{future_variable}}",
        label: "{{future_variable}}",
        description: "",
        example: "",
      },
    ],
  );
  assert.deepEqual(normalizeRecapPromptVariables(null), []);
});

test("a recap prompt variable replaces the current selection and returns its caret", () => {
  assert.deepEqual(
    insertRecapPromptVariable("Use OLD here", "{{meeting_date}}", 4, 7),
    {
      value: "Use {{meeting_date}} here",
      selectionStart: 20,
      selectionEnd: 20,
    },
  );
  assert.deepEqual(
    insertRecapPromptVariable("Prompt: ", "{{future_variable}}"),
    {
      value: "Prompt: {{future_variable}}",
      selectionStart: 27,
      selectionEnd: 27,
    },
  );
});

test("safe recap Markdown supports the allowed subset and produces plain text", () => {
  const markdown = [
    "# Risk review",
    "",
    "A **material** risk with *uncertain* timing and `owner_id`.",
    "",
    "- First item",
    "- Second item",
    "",
    "> Quoted context",
    "",
    "```js",
    "const safe = true;",
    "```",
  ].join("\n");
  assert.deepEqual(
    parseSafeMarkdown(markdown).map((block) => block.type),
    ["heading", "paragraph", "list", "blockquote", "code_block"],
  );
  assert.equal(
    safeMarkdownPlainText(markdown),
    [
      "Risk review",
      "",
      "A material risk with uncertain timing and owner_id.",
      "",
      "- First item\n- Second item",
      "",
      "Quoted context",
      "",
      "const safe = true;",
    ].join("\n"),
  );
});

test("safe recap Markdown treats raw HTML and scripts as literal text", () => {
  const malicious = '<script>globalThis.pwned=true</script>\n\n<img src=x onerror="pwned=true">';
  const blocks = parseSafeMarkdown(malicious);
  assert.deepEqual(blocks.map((block) => block.type), ["paragraph", "paragraph"]);
  assert.equal(safeMarkdownPlainText(malicious), malicious);
  assert.doesNotMatch(JSON.stringify(blocks), /"type":"html"/);
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

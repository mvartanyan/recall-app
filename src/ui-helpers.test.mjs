import test from "node:test";
import assert from "node:assert/strict";

import {
  formatDuration,
  formatTimestamp,
  isProvisionalLabel,
  parseLanguageHints,
  transcriptFromSegments,
} from "./ui-helpers.mjs";

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

test("speaker-attributed transcript is rebuilt from interventions", () => {
  assert.equal(
    transcriptFromSegments([
      { speaker_label: "Ada", text: "Hello." },
      { speaker_label: null, text: "Hi." },
    ]),
    "Ada: Hello.\nUnknown speaker: Hi.",
  );
});

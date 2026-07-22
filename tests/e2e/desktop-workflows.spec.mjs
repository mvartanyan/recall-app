import { expect, test } from "@playwright/test";

const oldSession = {
  id: "session-old",
  created_at: "2026-07-23T08:00:00Z",
  title: "Earlier planning meeting",
  duration_ms: 90_000,
  transcript: "Alice: Earlier discussion",
  processing_status: null,
  processing_error: null,
  processing_run_id: null,
  recoverable_audio: false,
};

async function installTauriMock(page) {
  await page.addInitScript(({ session }) => {
    const listeners = new Map();
    const native = {
      recording: false,
      sessions: [session],
      segments: {
        [session.id]: [
          {
            id: "segment-old",
            session_id: session.id,
            start_ms: 0,
            end_ms: 4_000,
            speaker_id: "speaker-alice",
            speaker_label: "Alice",
            text: "Earlier discussion",
          },
        ],
      },
    };
    const speakers = [
      {
        id: "speaker-alice",
        label: "Alice",
        created_at: "2026-07-20T08:00:00Z",
        last_seen_at: "2026-07-23T08:00:00Z",
        sample_count: 1,
        embedding_count: 1,
        conversation_count: 1,
      },
      {
        id: "speaker-voice",
        label: "VOICE12",
        created_at: "2026-07-20T08:00:00Z",
        last_seen_at: "2026-07-23T08:00:00Z",
        sample_count: 1,
        embedding_count: 1,
        conversation_count: 1,
      },
      {
        id: "speaker-alice-duplicate",
        label: "alice",
        created_at: "2026-07-20T08:00:00Z",
        last_seen_at: "2026-07-22T08:00:00Z",
        sample_count: 0,
        embedding_count: 1,
        conversation_count: 1,
      },
      {
        id: "speaker-unknown",
        label: "Unknown speaker",
        created_at: "2026-07-20T08:00:00Z",
        last_seen_at: "2026-07-22T08:00:00Z",
        sample_count: 0,
        embedding_count: 0,
        conversation_count: 1,
      },
    ];
    const preferences = {
      encryption_enabled: false,
      selected_input_device: null,
      language_hints: ["en", "de"],
      live_transcription: true,
      openai_model: "gpt-test",
      preferred_language: "en",
      no_translation_languages: ["de"],
      onboarding_version: "1",
    };
    const recapState = {
      agenda: null,
      recap: null,
      current_fingerprint: "fingerprint",
      stale: false,
      unresolved_profiles: [],
      in_flight: false,
    };
    const invoke = async (command, args = {}) => {
      switch (command) {
        case "app_status":
          return {
            encryption_enabled: false,
            db_open: true,
            needs_password: false,
            recording: native.recording,
            soniox_key_configured: true,
            openai_key_configured: true,
            speaker_model_available: true,
            selected_input_device: null,
            language_hints: preferences.language_hints,
            live_transcription: true,
          };
        case "get_preferences":
          return structuredClone(preferences);
        case "list_input_devices":
          return [{ name: "Test microphone", is_default: true }];
        case "list_translation_languages":
          return [
            { code: "de", name: "German" },
            { code: "en", name: "English" },
            { code: "fr", name: "French" },
          ];
        case "list_sessions":
          return structuredClone(native.sessions);
        case "list_segments":
          return structuredClone(native.segments[args.sessionId] || []);
        case "get_recap_state":
          return structuredClone(recapState);
        case "list_speakers_with_stats":
          return structuredClone(speakers);
        case "start_recording":
          native.recording = true;
          return {
            path: "/tmp/recall-test.wav",
            device_name: "Test microphone",
            sample_rate: 48_000,
            live_started: true,
          };
        case "stop_recording":
          native.recording = false;
          return "/tmp/recall-test.wav";
        case "transcribe_file_async": {
          const draft = {
            id: "session-draft",
            created_at: "2026-07-23T09:00:00Z",
            title: "Processing recording",
            duration_ms: 10_000,
            transcript: "",
            processing_status: "processing",
            processing_error: null,
            processing_run_id: "run-draft",
            recoverable_audio: true,
          };
          native.sessions = [draft, ...native.sessions];
          native.segments[draft.id] = [];
          return { run_id: "run-draft", session_id: draft.id };
        }
        case "get_progress":
          return [];
        case "get_live_transcription":
          return null;
        case "list_session_ids_for_speakers":
          return [session.id];
        case "save_preferences":
          preferences.selected_input_device = args.selectedInputDevice;
          preferences.language_hints = args.languageHints;
          preferences.live_transcription = args.liveTranscription;
          preferences.openai_model = args.openaiModel;
          preferences.preferred_language = args.preferredLanguage;
          preferences.no_translation_languages = args.noTranslationLanguages;
          return null;
        case "generate_recap":
          recapState.recap = {
            generated_at: "2026-07-23T09:05:00Z",
            input_tokens: 20,
            output_tokens: 10,
            payload: {
              target_language: preferences.preferred_language,
              meeting_title: "Earlier planning meeting",
              dominant_language: "en",
              executive_summary: {
                original: "Earlier discussion",
                translated: "Earlier discussion",
              },
              full_summary: [],
              commitments: [],
              actions_already_taken: [],
              agenda_present: false,
              agenda_coverage: [],
              translations: [],
            },
          };
          return structuredClone(recapState);
        default:
          throw new Error("Unhandled mocked command: " + command);
      }
    };
    window.__TAURI__ = {
      core: { invoke },
      event: {
        listen: async (name, handler) => {
          const handlers = listeners.get(name) || [];
          handlers.push(handler);
          listeners.set(name, handlers);
          return () => {};
        },
      },
    };
    window.__emitTauri = async (name, payload) => {
      for (const handler of listeners.get(name) || []) await handler({ payload });
    };
    window.__setMockPreferredLanguage = (language) => {
      preferences.preferred_language = language;
    };
  }, { session: oldSession });
}

test.beforeEach(async ({ page }) => {
  await installTauriMock(page);
  await page.goto("/");
  await expect(page.getByText("Recall is ready", { exact: false })).toBeAttached();
});

test("recording is a selectable workspace and does not block conversation history", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  const currentRecording = page.locator('[data-current-recording="true"]');
  await expect(currentRecording).toBeVisible();
  await expect(currentRecording).toHaveClass(/selected/);
  await expect(page.getByRole("heading", { name: "Connecting…" })).toBeVisible();

  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Guten Morgen",
      final_text: "",
      translated_text: "Good morning",
      translated_final_text: "",
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(page.getByText("Good morning", { exact: true })).toBeVisible();
  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Guten Morgen zusammen",
      final_text: "Speaker 1: Guten Morgen zusammen",
      translated_text: "Good morning everyone",
      translated_final_text: "Good morning everyone",
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(page.getByText("Good morning everyone", { exact: true })).toBeVisible();
  await expect(page.getByText("Good morning", { exact: true })).toHaveCount(0);

  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();
  await expect(page.getByLabel("Conversation title")).toHaveValue("Earlier planning meeting");
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  const preview = page.getByRole("button", { name: "Preview" }).first();
  await expect(preview).toBeDisabled();
  await expect(preview).toHaveAttribute(
    "title",
    "Voice preview is unavailable during recording",
  );

  await page.getByRole("button", { name: "Recap", exact: true }).click();
  await expect(page.getByRole("button", { name: "Regenerate recap" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();

  await page.getByPlaceholder("Search conversations").fill("does not match history");
  await expect(currentRecording).toBeVisible();
  await page.getByPlaceholder("Search conversations").fill("");
  await page.getByLabel("Filter conversations by voice").selectOption("alice");
  await expect(currentRecording).toBeVisible();

  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByLabel("Preferred language").selectOption("de");
  await page.getByRole("button", { name: "Save settings" }).click();

  await currentRecording.click();
  await expect(page.getByRole("heading", { name: "Live" })).toBeVisible();
  await expect(page.getByText("Good morning everyone", { exact: true })).toBeVisible();
  await expect(page.getByText("Translation · English", { exact: true })).toBeVisible();
});

test("browsing history prevents the completed recording from stealing focus", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();
  await page.getByRole("button", { name: "Stop recording" }).click();

  await expect(page.getByLabel("Conversation title")).toHaveValue("Earlier planning meeting");
  await expect(page.getByRole("button", { name: /Processing recording/ })).toBeVisible();
  await expect(page.locator('[data-current-recording="true"]')).toHaveCount(0);
});

test("stopping from the live workspace opens the durable processing draft", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  await page.getByRole("button", { name: "Stop recording" }).click();

  await expect(page.getByLabel("Conversation title")).toHaveValue("Processing recording");
  await expect(page.locator('[data-current-recording="true"]')).toHaveCount(0);
});

test("the conversation filter contains named people only", async ({ page }) => {
  const filter = page.getByLabel("Filter conversations by voice");
  await expect(filter.locator("option")).toHaveText(["All voices", "Alice"]);
});

test("preferred language is selected from provider capabilities and removed from exclusions", async ({ page }) => {
  await page.getByRole("button", { name: "Settings" }).click();
  const preferred = page.getByLabel("Preferred language");
  await expect(preferred.locator("option")).toHaveText([
    "German (de)",
    "English (en)",
    "French (fr)",
  ]);
  const exclusions = page.getByLabel("No translation for languages");
  await exclusions.fill("en, de, fr");
  await preferred.selectOption("de");
  await expect(exclusions).toHaveValue("en, fr");
  await page.getByRole("button", { name: "Save settings" }).click();
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByLabel("Preferred language")).toHaveValue("de");
  await expect(page.getByLabel("No translation for languages")).toHaveValue("en, fr");
});

test("an unavailable saved target stays visible for correction", async ({ page }) => {
  await page.evaluate(() => window.__setMockPreferredLanguage("xx"));
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByLabel("Preferred language")).toHaveValue("xx");
  await expect(page.getByLabel("Preferred language").locator("option:checked")).toHaveText(
    "XX (unavailable for live translation)",
  );
});

test("an unavailable live target keeps original captions and reports the warning", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Original speech continues",
      final_text: "",
      translated_text: "",
      translated_final_text: "",
      target_language: null,
      translation_warning:
        "Preferred language XX is unavailable for live STT translation. Original live captions will continue.",
      finished: false,
      error: null,
    }),
  );

  await expect(page.getByText("Speaker 1: Original speech continues", { exact: true })).toBeVisible();
  await expect(page.locator("#liveTranslationSection")).toBeHidden();
  await expect(page.locator("#activityLog")).toContainText("Original live captions will continue");
});

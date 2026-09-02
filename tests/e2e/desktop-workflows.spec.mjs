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
    let jamieChoiceGate = null;
    let releaseJamieChoice = null;
    let jamieInspectionGate = null;
    let releaseJamieInspection = null;
    let nextAppStatusGate = null;
    let releaseAppStatus = null;
    const conversationLoadGates = new Map();
    const releaseConversationLoads = new Map();
    const customRecapGates = new Map();
    const releaseCustomRecaps = new Map();
    const customRecapFailures = new Map();
    const customRecapMarkdown = new Map();
    const commandCounts = {};
    const commandCalls = [];
    let lastIdentityImpactPreview = null;
    let identityImpactGeneration = 0;
    const native = {
      recording: false,
      sttContext: { language_hints: ["en", "de"], expected_speakers: null },
      sttContextRevision: 0,
      queuedSttContext: null,
      sessions: [session],
      voiceGroups: { [session.id]: [] },
      importBatches: [],
      importedArtifacts: {},
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
          {
            id: "segment-likely",
            session_id: session.id,
            start_ms: 5_000,
            end_ms: 9_000,
            speaker_id: "speaker-voice",
            speaker_label: "VOICE12",
            text: "A later contribution",
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
        likely_match: {
          decision_id: "decision-likely",
          speaker_id: "speaker-alice",
          label: "Alice",
          score: 0.9555,
          runner_up_label: "Dmitrii",
          runner_up_score: 0.9403,
          support_count: 1,
          reason: "Strong but ambiguous",
        },
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
    const makeRecapState = () => ({
      agenda: null,
      recap: null,
      custom_recaps: [],
      current_fingerprint: "fingerprint",
      stale: false,
      unresolved_profiles: [],
      in_flight: false,
    });
    const recapStates = { [session.id]: makeRecapState() };
    const recapStateFor = (sessionId) => {
      if (!recapStates[sessionId]) recapStates[sessionId] = makeRecapState();
      return recapStates[sessionId];
    };
    const recapTypeDefaults = {
      "builtin-executive-summary": "Summarize the purpose, conclusions, decisions, material risks, disagreements and open questions.",
      "builtin-full-summary": "Give a detailed sectioned account of topics, arguments, rationale, decisions, dependencies, risks and next steps.",
      "builtin-actions": "Identify explicit future commitments and already-completed actions with participant, timing and uncertainty.",
    };
    const recapTypes = [
      {
        id: "builtin-executive-summary",
        kind: "builtin",
        name: "Executive summary",
        prompt: recapTypeDefaults["builtin-executive-summary"],
        is_builtin: true,
        created_at: "2026-07-23T08:00:00Z",
        updated_at: "2026-07-23T08:00:00Z",
      },
      {
        id: "builtin-full-summary",
        kind: "builtin",
        name: "Full summary",
        prompt: recapTypeDefaults["builtin-full-summary"],
        is_builtin: true,
        created_at: "2026-07-23T08:00:00Z",
        updated_at: "2026-07-23T08:00:00Z",
      },
      {
        id: "builtin-actions",
        kind: "builtin",
        name: "Actions",
        prompt: recapTypeDefaults["builtin-actions"],
        is_builtin: true,
        created_at: "2026-07-23T08:00:00Z",
        updated_at: "2026-07-23T08:00:00Z",
      },
    ];
    const recapPromptVariables = [
      {
        token: "{{meeting_date}}",
        label: "Meeting date",
        description: "The selected conversation's saved local date.",
        example: "2026/07/23",
      },
      {
        token: "{{meeting_time}}",
        label: "Meeting time",
        description: "The selected conversation's saved local time.",
        example: "10:00",
      },
      {
        token: "{{meeting_datetime}}",
        label: "Meeting date and time",
        description: "The selected conversation's saved local date and time.",
        example: "2026/07/23 10:00",
      },
    ];
    let recapPromptVariablesUnavailable = false;
    let recapTypeSequence = 0;
    const jamiePreview = {
      draft: {
        id: "aaaaaaaaaaaaaaaa",
        source_path: "/tmp/jamie-export.txt",
        source_sha256: "a".repeat(64),
        importer_version: "jamie-text-v1",
        identity_decisions: [
          {
            alias: "Mv",
            action: "proposed_map",
            target_speaker_id: "speaker-michael",
            display_name: "Michael Vartanyan",
          },
          {
            alias: "Bob Example",
            action: "review",
            target_speaker_id: null,
            display_name: null,
          },
          {
            alias: "Speaker 0",
            action: "unresolved",
            target_speaker_id: null,
            display_name: null,
          },
        ],
        excluded_meetings: [],
        updated_at: "2026-07-23T09:01:59Z",
      },
      metadata: {
        user: "Test User",
        export_date: "2026-07-23T09:01:59Z",
        declared_total_meetings: 2,
        includes: ["summaries", "transcripts", "tasks"],
        source_sha256: "a".repeat(64),
        source_size_bytes: 25_024_004,
      },
      known_people: [
        { id: "speaker-alice", label: "Alice" },
        { id: "speaker-michael", label: "Michael Vartanyan" },
      ],
      meetings: [
        {
          source_fingerprint: "meeting-valid",
          title: "Imported planning call",
          started_at: "2026-07-22T12:00:00Z",
          duration_ms: 60_000,
          intervention_count: 2,
          speaker_count: 2,
          has_executive_summary: true,
          has_full_summary: true,
          has_tasks: true,
          included: true,
          already_imported: false,
          warnings: [],
        },
        {
          source_fingerprint: "meeting-empty",
          title: "Empty source meeting",
          started_at: "2026-07-21T12:00:00Z",
          duration_ms: 0,
          intervention_count: 0,
          speaker_count: 0,
          has_executive_summary: false,
          has_full_summary: true,
          has_tasks: false,
          included: true,
          already_imported: false,
          warnings: [
            {
              code: "empty_transcript",
              message: "The meeting has no transcript interventions.",
              blocking: true,
            },
          ],
        },
      ],
      identities: [
        {
          alias: "Bob Example",
          generic: false,
          intervention_count: 1,
          meeting_count: 1,
          excerpts: ["I will send the draft tomorrow."],
          decision: {
            alias: "Bob Example",
            action: "review",
            target_speaker_id: null,
            display_name: null,
          },
        },
        {
          alias: "Mv",
          generic: false,
          intervention_count: 1,
          meeting_count: 1,
          excerpts: ["Let us review the plan."],
          decision: {
            alias: "Mv",
            action: "proposed_map",
            target_speaker_id: "speaker-michael",
            display_name: "Michael Vartanyan",
          },
        },
        {
          alias: "Speaker 0",
          generic: true,
          intervention_count: 1,
          meeting_count: 1,
          excerpts: ["Generic source label."],
          decision: {
            alias: "Speaker 0",
            action: "unresolved",
            target_speaker_id: null,
            display_name: null,
          },
        },
      ],
      archive_warnings: [],
      validation_errors: [
        "Bob Example: choose how this source identity should import.",
      ],
      ready_to_import: false,
      included_meeting_count: 2,
      existing_meeting_count: 0,
      total_intervention_count: 3,
    };
    const invoke = async (command, args = {}) => {
      commandCounts[command] = (commandCounts[command] || 0) + 1;
      commandCalls.push({ command, args: structuredClone(args) });
      if (command === "load_conversation" && args.sessionId) {
        const scopedKey = command + ":" + args.sessionId;
        commandCounts[scopedKey] = (commandCounts[scopedKey] || 0) + 1;
      }
      switch (command) {
        case "app_status": {
          const status = {
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
            current_stt_context: native.recording
              ? structuredClone(native.sttContext)
              : null,
            live_recording_active: native.recording,
          };
          if (nextAppStatusGate) {
            const gate = nextAppStatusGate;
            nextAppStatusGate = null;
            await gate;
          }
          return status;
        }
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
          return structuredClone(
            native.sessions.map(({ transcript: _transcript, ...summary }) => summary),
          );
        case "search_session_ids": {
          const query = String(args.query || "").trim().toLocaleLowerCase();
          return native.sessions
            .filter((candidate) =>
              (String(candidate.title || "") + " " + String(candidate.transcript || ""))
                .toLocaleLowerCase()
                .includes(query),
            )
            .map((candidate) => candidate.id);
        }
        case "load_conversation": {
          const gate = conversationLoadGates.get(args.sessionId);
          if (gate) await gate;
          const selected = native.sessions.find(
            (candidate) => candidate.id === args.sessionId,
          );
          if (!selected) throw new Error("Conversation not found");
          return structuredClone({
            session: selected,
            segments: native.segments[args.sessionId] || [],
            voice_groups: native.voiceGroups[args.sessionId] || [],
            recap_state: recapStateFor(args.sessionId),
            imported_artifact: native.importedArtifacts[args.sessionId] || null,
          });
        }
        case "update_segment_text": {
          const segment = (native.segments[args.sessionId] || []).find(
            (candidate) => candidate.id === args.segmentId,
          );
          if (!segment) throw new Error("Intervention not found");
          segment.text = args.text;
          return null;
        }
        case "assign_segment_speaker": {
          const segment = (native.segments[args.sessionId] || []).find(
            (candidate) => candidate.id === args.segmentId,
          );
          if (!segment) throw new Error("Intervention not found");
          const speaker = speakers.find((candidate) => candidate.id === args.speakerId);
          segment.speaker_id = speaker?.id || null;
          segment.speaker_label = speaker?.label || null;
          return null;
        }
        case "list_import_batches":
          return structuredClone(native.importBatches);
        case "list_segments":
          return structuredClone(native.segments[args.sessionId] || []);
        case "get_recap_state":
          return structuredClone(recapStateFor(args.sessionId));
        case "get_imported_session_artifact":
          return structuredClone(native.importedArtifacts[args.sessionId] || null);
        case "list_speakers_with_stats":
          return structuredClone(speakers);
        case "list_identity_profiles": {
          let items = speakers.map((speaker) => ({
            id: speaker.id,
            label: speaker.label || "Unnamed voice",
            created_at: speaker.created_at,
            last_seen_at: speaker.last_seen_at,
            sample_count: speaker.sample_count || 0,
            active_voiceprint_count: speaker.embedding_count || 0,
            inactive_voiceprint_count: 0,
            conversation_count: speaker.conversation_count || 0,
            intervention_count: Object.values(native.segments)
              .flat()
              .filter((segment) => segment.speaker_id === speaker.id).length,
            provisional: /^VOICE\d+$/i.test(speaker.label || ""),
            imported: speaker.id === "speaker-alice-duplicate",
            duplicate_name_conflict:
              speaker.id === "speaker-alice" ||
              speaker.id === "speaker-alice-duplicate",
            duplicate_name_count:
              speaker.id === "speaker-alice" ||
              speaker.id === "speaker-alice-duplicate"
                ? 2
                : 0,
          }));
          const search = String(args.search || "").toLowerCase();
          if (search) {
            items = items.filter((item) =>
              item.label.toLowerCase().includes(search),
            );
          }
          if (args.status === "named") {
            items = items.filter((item) => !item.provisional);
          } else if (args.status === "provisional") {
            items = items.filter((item) => item.provisional);
          } else if (args.status === "no_voiceprint") {
            items = items.filter((item) => item.active_voiceprint_count === 0);
          } else if (args.status === "conflict") {
            items = items.filter((item) => item.duplicate_name_conflict);
          } else if (args.status === "imported") {
            items = items.filter((item) => item.imported);
          }
          items.sort((left, right) =>
            left.label.localeCompare(right.label, undefined, {
              sensitivity: "base",
              numeric: true,
            }),
          );
          return {
            items: structuredClone(items),
            total: items.length,
            page: 1,
            page_size: 100,
            page_count: 1,
          };
        }
        case "list_unassigned_identities":
          return {
            items: [
              {
                key: {
                  session_id: session.id,
                  speaker_label: "Speaker 1",
                },
                display_label: "Speaker 1",
                session_title: session.title,
                session_created_at: session.created_at,
                intervention_count: 2,
                first_start_ms: 10_000,
                last_end_ms: 16_000,
                generic: true,
              },
            ],
            total: 1,
            page: 1,
            page_size: 100,
            page_count: 1,
          };
        case "preview_identity_consolidation": {
          const targetId = args.request.target_speaker_id;
          const target = speakers.find((speaker) => speaker.id === targetId);
          const affectedSessionIds = Array.from(
            new Set(
              args.request.unassigned_groups.map((group) => group.session_id),
            ),
          ).sort();
          if (!affectedSessionIds.length) affectedSessionIds.push(session.id);
          const affectedInterventionCount = Object.values(native.segments)
            .flat()
            .filter((segment) =>
              args.request.unassigned_groups.some(
                (group) =>
                  group.session_id === segment.session_id &&
                  (group.voice_group_id
                    ? group.voice_group_id === segment.voice_group_id
                    : group.speaker_label === segment.speaker_label) &&
                  !segment.speaker_id,
              ),
            ).length + args.request.profile_ids.length;
          const preview = {
            target_speaker_id: targetId,
            target_label: args.request.final_label,
            source_profiles: [],
            unassigned_groups: [],
            affected_session_ids: affectedSessionIds,
            affected_conversation_count: affectedSessionIds.length,
            affected_intervention_count: affectedInterventionCount,
            stale_recap_count: 1,
            active_voiceprint_count: args.request.profile_ids.length,
            inactive_voiceprint_count: 0,
            samples_to_delete: args.request.profile_ids.length,
            imported_source_profile_count: 0,
            creates_new_person: !target,
            warnings: [
              "1 saved recap will be marked out of date.",
              "Temporary voice samples will be deleted for privacy.",
            ],
          };
          preview.impact_revision =
            "mock-impact-token-" +
            identityImpactGeneration +
            "-" +
            commandCounts.preview_identity_consolidation;
          lastIdentityImpactPreview = {
            request: structuredClone(args.request),
            preview: structuredClone(preview),
            generation: identityImpactGeneration,
          };
          return preview;
        }
        case "consolidate_identities": {
          if (
            !lastIdentityImpactPreview ||
            lastIdentityImpactPreview.generation !== identityImpactGeneration ||
            args.expectedImpactRevision !==
              lastIdentityImpactPreview.preview.impact_revision ||
            JSON.stringify(args.expectedAffectedSessionIds || []) !==
              JSON.stringify(lastIdentityImpactPreview.preview.affected_session_ids)
          ) {
            throw new Error(
              "The people, voices, recaps, or affected conversations changed after the impact preview. Review the operation again.",
            );
          }
          const request = args.request;
          let target = speakers.find(
            (speaker) => speaker.id === request.target_speaker_id,
          );
          if (!target) {
            target = {
              id: "speaker-created",
              label: request.final_label,
              created_at: "2026-07-23T12:00:00Z",
              last_seen_at: session.created_at,
              sample_count: 0,
              embedding_count: 0,
              conversation_count: 1,
            };
            speakers.push(target);
          }
          const sourceIds = new Set(
            request.profile_ids.filter((id) => id !== target.id),
          );
          for (const segments of Object.values(native.segments)) {
            for (const segment of segments) {
              const groupSelected = request.unassigned_groups.some(
                (group) =>
                  group.session_id === segment.session_id &&
                  (group.voice_group_id
                    ? group.voice_group_id === segment.voice_group_id
                    : group.speaker_label === segment.speaker_label) &&
                  !segment.speaker_id,
              );
              if (sourceIds.has(segment.speaker_id) || groupSelected) {
                segment.speaker_id = target.id;
                segment.speaker_label = request.final_label;
              } else if (segment.speaker_id === target.id) {
                segment.speaker_label = request.final_label;
              }
            }
          }
          for (const groupKey of request.unassigned_groups) {
            if (!groupKey.voice_group_id) continue;
            const group = (native.voiceGroups[groupKey.session_id] || []).find(
              (candidate) => candidate.id === groupKey.voice_group_id,
            );
            const groupSegments = (native.segments[groupKey.session_id] || []).filter(
              (segment) => segment.voice_group_id === groupKey.voice_group_id,
            );
            if (
              group &&
              groupSegments.length &&
              groupSegments.every((segment) => segment.speaker_id === target.id)
            ) {
              group.resulting_speaker_id = target.id;
              group.resulting_speaker_label = request.final_label;
            }
          }
          target.label = request.final_label;
          for (let index = speakers.length - 1; index >= 0; index -= 1) {
            if (sourceIds.has(speakers[index].id)) speakers.splice(index, 1);
          }
          return {
            target_speaker_id: target.id,
            target_label: request.final_label,
            merged_profile_count: sourceIds.size,
            assigned_group_count: request.unassigned_groups.length,
            affected_conversation_count: 1,
            affected_intervention_count: 2,
            activated_voiceprints: sourceIds.size,
            quarantined_voiceprints: 0,
            deleted_samples: request.profile_ids.length,
            backup_path: "/tmp/recall.pre-identity-merge.db",
          };
        }
        case "start_recording":
          native.recording = true;
          native.sttContext = {
            language_hints: [...preferences.language_hints],
            expected_speakers: null,
          };
          return {
            path: "/tmp/recall-test.wav",
            device_name: "Test microphone",
            sample_rate: 48_000,
            live_started: true,
            stt_context: structuredClone(native.sttContext),
          };
        case "update_live_context": {
          const next = structuredClone(args.sttContext);
          const changed = JSON.stringify(next) !== JSON.stringify(native.sttContext);
          native.sttContext = next;
          if (changed) native.sttContextRevision += 1;
          return {
            stt_context: structuredClone(next),
            changed,
            live_restart_pending: changed,
            revision: native.sttContextRevision,
            delivery_status: changed ? "pending" : "unchanged",
          };
        }
        case "stop_recording":
          if (!native.recording) throw new Error("There is no active recording");
          native.recording = false;
          return {
            path: "/tmp/recall-test.wav",
            stt_context: structuredClone(native.sttContext),
          };
        case "transcribe_file_async": {
          native.queuedSttContext = structuredClone(args.sttContext);
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
          recapStateFor(draft.id);
          return { run_id: "run-draft", session_id: draft.id };
        }
        case "get_progress":
          return [];
        case "get_live_transcription":
          return null;
        case "choose_jamie_export":
          if (jamieChoiceGate) await jamieChoiceGate;
          return "/tmp/jamie-export.txt";
        case "inspect_jamie_export":
          if (jamieInspectionGate) await jamieInspectionGate;
          return structuredClone(jamiePreview);
        case "resume_jamie_import":
          return structuredClone(jamiePreview);
        case "save_jamie_import_draft":
          jamiePreview.draft = structuredClone(args.draft);
          return null;
        case "run_jamie_import": {
          const imported = {
            id: "session-jamie",
            created_at: "2026-07-22T12:00:00Z",
            title: "Imported planning call",
            duration_ms: 60_000,
            transcript:
              "Michael Vartanyan: Let us review the plan.\nBob Example: I will send the draft tomorrow.",
            processing_status: null,
            processing_error: null,
            processing_run_id: null,
            recoverable_audio: false,
          };
          native.sessions = [native.sessions[0], imported];
          native.segments[imported.id] = [
            {
              id: "segment-jamie-1",
              session_id: imported.id,
              start_ms: 0,
              end_ms: 4_000,
              speaker_id: "speaker-michael",
              speaker_label: "Michael Vartanyan",
              text: "Let us review the plan.",
            },
            {
              id: "segment-jamie-2",
              session_id: imported.id,
              start_ms: 5_000,
              end_ms: 9_000,
              speaker_id: "speaker-bob",
              speaker_label: "Bob Example",
              text: "I will send the draft tomorrow.",
            },
          ];
          speakers.push({
            id: "speaker-bob",
            label: "Bob Example",
            created_at: "2026-07-23T10:00:00Z",
            last_seen_at: "2026-07-22T12:00:00Z",
            sample_count: 0,
            embedding_count: 0,
            conversation_count: 1,
          });
          native.importedArtifacts[imported.id] = {
            session_id: imported.id,
            source_provider: "Jamie",
            source_meeting_sha256: "meeting-valid",
            imported_at: "2026-07-23T10:00:00Z",
            executive_summary: "The team reviewed the plan.",
            full_summary: "## Planning\nThe plan was reviewed.",
            tasks: "[ ] Bob will send the draft.",
          };
          native.importBatches = [
            {
              id: "import-jamie",
              source_provider: "Jamie",
              source_file_sha256: "a".repeat(64),
              imported_at: "2026-07-23T10:00:00Z",
              status: "imported",
              meeting_count: 1,
              rolled_back_at: null,
            },
          ];
          return {
            import_id: "import-jamie",
            backup_path: "/tmp/recall.pre-jamie-import.db",
            imported_meetings: 1,
            already_imported_meetings: 0,
            imported_interventions: 2,
            created_people: 1,
          };
        }
        case "rollback_jamie_import":
          native.sessions = native.sessions.filter(
            (candidate) => candidate.id !== "session-jamie",
          );
          delete native.segments["session-jamie"];
          delete native.importedArtifacts["session-jamie"];
          const bobIndex = speakers.findIndex(
            (speaker) => speaker.id === "speaker-bob",
          );
          if (bobIndex >= 0) speakers.splice(bobIndex, 1);
          native.importBatches[0].status = "rolled_back";
          native.importBatches[0].rolled_back_at = "2026-07-23T11:00:00Z";
          return {
            import_id: args.importId,
            backup_path: "/tmp/recall.pre-jamie-rollback.db",
            removed_meetings: 1,
            removed_people: 1,
            preserved_people: 0,
          };
        case "list_session_ids_for_speakers":
          return [session.id];
        case "accept_voice_match_suggestion": {
          const sourceIndex = speakers.findIndex((speaker) => speaker.id === args.sourceId);
          const target = speakers.find((speaker) => speaker.id === args.targetId);
          if (sourceIndex < 0 || !target) throw new Error("Suggestion is no longer current");
          for (const segments of Object.values(native.segments)) {
            for (const segment of segments) {
              if (segment.speaker_id !== args.sourceId) continue;
              segment.speaker_id = target.id;
              segment.speaker_label = target.label;
            }
          }
          speakers.splice(sourceIndex, 1);
          native.sessions[0].transcript =
            "Alice: Earlier discussion\nAlice: A later contribution";
          return {
            target_speaker_id: target.id,
            target_label: target.label,
            activated_voiceprints: 1,
            quarantined_voiceprints: 0,
          };
        }
        case "split_voice_group": {
          let sourceGroup = null;
          for (const groups of Object.values(native.voiceGroups)) {
            sourceGroup = groups.find((group) => group.id === args.voiceGroupId);
            if (sourceGroup) break;
          }
          if (!sourceGroup) throw new Error("Voice group not found");
          const selectedIds = new Set(args.selectedSegmentIds || []);
          const sessionSegments = native.segments[sourceGroup.session_id] || [];
          const moved = sessionSegments.filter((segment) => selectedIds.has(segment.id));
          if (!moved.length || moved.length >= sourceGroup.intervention_count) {
            throw new Error("Select some, but not all, interventions");
          }
          const newSpeaker = {
            id: "speaker-split",
            label: "VOICE13",
            created_at: "2026-07-23T12:00:00Z",
            last_seen_at: "2026-07-23T12:00:00Z",
            sample_count: 1,
            embedding_count: 1,
            conversation_count: 1,
          };
          speakers.push(newSpeaker);
          for (const segment of moved) {
            segment.speaker_id = newSpeaker.id;
            segment.speaker_label = newSpeaker.label;
            segment.voice_group_id = "voice-group-split";
          }
          sourceGroup.intervention_count -= moved.length;
          sourceGroup.voice_observation_count = Math.max(
            0,
            sourceGroup.voice_observation_count - moved.length,
          );
          sourceGroup.split_status = "reviewed";
          sourceGroup.split_clusters = [];
          native.voiceGroups[sourceGroup.session_id].push({
            ...sourceGroup,
            id: "voice-group-split",
            cluster_index: 1,
            resulting_speaker_id: newSpeaker.id,
            resulting_speaker_label: newSpeaker.label,
            split_status: "reviewed",
            split_clusters: [],
            intervention_count: moved.length,
            voice_observation_count: moved.length,
          });
          return {
            session_id: sourceGroup.session_id,
            original_group_id: sourceGroup.id,
            new_group_id: "voice-group-split",
            new_speaker_id: newSpeaker.id,
            new_speaker_label: newSpeaker.label,
            moved_interventions: moved.length,
            remaining_interventions: sourceGroup.intervention_count,
            backup_path: "/tmp/recall.pre-voice-split.db",
          };
        }
        case "dismiss_voice_group_split": {
          for (const groups of Object.values(native.voiceGroups)) {
            const group = groups.find((candidate) => candidate.id === args.voiceGroupId);
            if (!group) continue;
            group.split_status = "dismissed";
            group.split_clusters = [];
            return null;
          }
          throw new Error("Voice group not found");
        }
        case "preview_voice_recognition_reset":
          return {
            can_reset: true,
            blockers: [],
            preview: {
              voiceprints: speakers.reduce(
                (total, speaker) => total + Number(speaker.embedding_count || 0),
                0,
              ),
              temporary_samples: speakers.reduce(
                (total, speaker) => total + Number(speaker.sample_count || 0),
                0,
              ),
              match_decisions: 1,
              meeting_voice_groups: Object.values(native.voiceGroups).flat().length,
              voice_observations: Object.values(native.voiceGroups)
                .flat()
                .reduce(
                  (total, group) => total + Number(group.voice_observation_count || 0),
                  0,
                ),
              provisional_profiles: speakers.filter((speaker) =>
                /^VOICE\d+$/i.test(speaker.label || ""),
              ).length,
              provisional_attributions_demoted: Object.values(native.segments)
                .flat()
                .filter((segment) => /^VOICE\d+$/i.test(segment.speaker_label || ""))
                .length,
              named_profiles_preserved: speakers.filter(
                (speaker) => !/^VOICE\d+$/i.test(speaker.label || ""),
              ).length,
            },
          };
        case "reset_voice_recognition": {
          const preview = await invoke("preview_voice_recognition_reset");
          const provisionalIds = new Set(
            speakers
              .filter((speaker) => /^VOICE\d+$/i.test(speaker.label || ""))
              .map((speaker) => speaker.id),
          );
          for (const segment of Object.values(native.segments).flat()) {
            if (provisionalIds.has(segment.speaker_id)) segment.speaker_id = null;
            delete segment.voice_group_id;
          }
          for (let index = speakers.length - 1; index >= 0; index -= 1) {
            if (provisionalIds.has(speakers[index].id)) {
              speakers.splice(index, 1);
            } else {
              speakers[index].embedding_count = 0;
              speakers[index].sample_count = 0;
              speakers[index].likely_match = null;
            }
          }
          for (const sessionId of Object.keys(native.voiceGroups)) {
            native.voiceGroups[sessionId] = [];
          }
          return {
            preview: preview.preview,
            backup_path: "/tmp/recall.pre-voice-reset-v4.db",
            integrity_check: "ok",
          };
        }
        case "save_preferences":
          preferences.selected_input_device = args.preferences.selectedInputDevice;
          preferences.language_hints = args.preferences.languageHints;
          preferences.live_transcription = args.preferences.liveTranscription;
          preferences.openai_model = args.preferences.openaiModel;
          preferences.preferred_language = args.preferences.preferredLanguage;
          preferences.no_translation_languages = args.preferences.noTranslationLanguages;
          return null;
        case "list_recap_types":
          return structuredClone(
            recapTypes.map((recapType) => ({
              ...recapType,
              prompt: args.includePrompts ? recapType.prompt : null,
            })),
          );
        case "list_recap_prompt_variables":
          if (recapPromptVariablesUnavailable) {
            throw new Error("Prompt variable registry unavailable");
          }
          return structuredClone(recapPromptVariables);
        case "create_recap_type": {
          const name = String(args.name || "").normalize("NFC").trim().replace(/\s+/gu, " ");
          if (!name) throw new Error("Custom recap type names cannot be empty");
          if (Array.from(name).length > 20) {
            throw new Error("Custom recap type names are limited to 20 characters");
          }
          recapTypeSequence += 1;
          const recapType = {
            id: "custom-type-" + recapTypeSequence,
            kind: "custom",
            name,
            prompt: String(args.prompt || ""),
            is_builtin: false,
            created_at: "2026-07-23T09:0" + recapTypeSequence + ":00Z",
            updated_at: "2026-07-23T09:0" + recapTypeSequence + ":00Z",
          };
          recapTypes.push(recapType);
          return structuredClone(recapType);
        }
        case "update_recap_type": {
          const recapType = recapTypes.find((candidate) => candidate.id === args.recapTypeId);
          if (!recapType) throw new Error("Recap type not found");
          if (!recapType.is_builtin) {
            recapType.name = String(args.name || "").normalize("NFC").trim().replace(/\s+/gu, " ");
          }
          recapType.prompt = String(args.prompt || "");
          recapType.updated_at = "2026-07-23T09:20:00Z";
          return structuredClone(recapType);
        }
        case "delete_recap_type": {
          const index = recapTypes.findIndex((candidate) => candidate.id === args.recapTypeId);
          if (index < 0) throw new Error("Recap type not found");
          if (recapTypes[index].is_builtin) throw new Error("Built-in recap types cannot be deleted");
          recapTypes.splice(index, 1);
          return null;
        }
        case "restore_recap_type_default": {
          const recapType = recapTypes.find((candidate) => candidate.id === args.recapTypeId);
          if (!recapType?.is_builtin) throw new Error("Only built-in recap types have defaults");
          recapType.prompt = recapTypeDefaults[recapType.id];
          recapType.updated_at = "2026-07-23T09:21:00Z";
          return structuredClone(recapType);
        }
        case "generate_recap": {
          const recapState = recapStateFor(args.sessionId);
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
        }
        case "generate_custom_recap": {
          const recapType = structuredClone(
            recapTypes.find((candidate) => candidate.id === args.recapTypeId),
          );
          if (!recapType || recapType.kind !== "custom") {
            throw new Error("Custom recap type not found");
          }
          const recapState = recapStateFor(args.sessionId);
          if (recapState.unresolved_profiles.length && !args.allowUnresolved) {
            throw new Error("Resolve participants or choose Recap anyway");
          }
          recapState.in_flight = true;
          const gate = customRecapGates.get(args.sessionId);
          if (gate) await gate;
          const failure = customRecapFailures.get(args.sessionId);
          if (failure) {
            customRecapFailures.delete(args.sessionId);
            recapState.in_flight = false;
            throw new Error(failure);
          }
          const result = {
            recap_type_id: recapType.id,
            name: recapType.name,
            generated_at: "2026-07-23T09:30:00Z",
            target_language: preferences.preferred_language,
            model: preferences.openai_model,
            source_fingerprint: recapState.current_fingerprint,
            content_markdown:
              customRecapMarkdown.get(recapType.id) ||
              "## " + recapType.name + "\n\nGenerated custom content.",
            input_tokens: 30,
            output_tokens: 15,
            stale: false,
          };
          recapState.custom_recaps = [
            ...recapState.custom_recaps.filter(
              (candidate) => candidate.recap_type_id !== recapType.id,
            ),
            result,
          ];
          recapState.in_flight = false;
          return structuredClone(recapState);
        }
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
    window.__mockCommandCount = (command) => commandCounts[command] || 0;
    window.__mockLastCommandArgs = (command) => {
      const call = commandCalls.findLast((candidate) => candidate.command === command);
      return call ? structuredClone(call.args) : null;
    };
    window.__invalidateIdentityImpactPreview = () => {
      identityImpactGeneration += 1;
    };
    window.__mockIdentityImpactRevision = () =>
      lastIdentityImpactPreview?.preview?.impact_revision || null;
    window.__setMockPromptVariablesUnavailable = (unavailable) => {
      recapPromptVariablesUnavailable = Boolean(unavailable);
    };
    let clipboardText = "";
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (value) => {
          clipboardText = String(value);
        },
      },
    });
    window.__mockClipboardText = () => clipboardText;
    window.__addMockRecapType = (name, prompt = "Create a focused custom recap.") => {
      recapTypeSequence += 1;
      const recapType = {
        id: "custom-type-" + recapTypeSequence,
        kind: "custom",
        name,
        prompt,
        is_builtin: false,
        created_at: "2026-07-23T09:0" + recapTypeSequence + ":00Z",
        updated_at: "2026-07-23T09:0" + recapTypeSequence + ":00Z",
      };
      recapTypes.push(recapType);
      return recapType.id;
    };
    window.__setMockCustomMarkdown = (recapTypeId, markdown) => {
      customRecapMarkdown.set(recapTypeId, String(markdown));
    };
    window.__setMockUnresolvedProfiles = (sessionId, labels) => {
      recapStateFor(sessionId).unresolved_profiles = [...labels];
    };
    window.__setMockCustomRecapStale = (sessionId, recapTypeId, stale) => {
      const recap = recapStateFor(sessionId).custom_recaps.find(
        (candidate) => candidate.recap_type_id === recapTypeId,
      );
      if (recap) recap.stale = Boolean(stale);
    };
    window.__deferCustomRecap = (sessionId) => {
      customRecapGates.set(
        sessionId,
        new Promise((resolve) => releaseCustomRecaps.set(sessionId, resolve)),
      );
    };
    window.__releaseCustomRecap = (sessionId) => {
      releaseCustomRecaps.get(sessionId)?.();
      releaseCustomRecaps.delete(sessionId);
      customRecapGates.delete(sessionId);
    };
    window.__failNextCustomRecap = (sessionId, message = "Mock custom recap failure") => {
      customRecapFailures.set(sessionId, message);
    };
    window.__mockCustomRecaps = (sessionId) =>
      structuredClone(recapStateFor(sessionId).custom_recaps);
    window.__mockConversationLoadCount = (sessionId) =>
      commandCounts["load_conversation:" + sessionId] || 0;
    window.__mockSessionTranscript = (sessionId) =>
      native.sessions.find((candidate) => candidate.id === sessionId)?.transcript ?? null;
    window.__setNativeRecording = (recording) => {
      native.recording = Boolean(recording);
    };
    window.__mockSttContext = () => structuredClone(native.sttContext);
    window.__mockQueuedSttContext = () => structuredClone(native.queuedSttContext);
    window.__deferNextAppStatus = () => {
      nextAppStatusGate = new Promise((resolve) => {
        releaseAppStatus = resolve;
      });
    };
    window.__releaseAppStatus = () => {
      releaseAppStatus?.();
      releaseAppStatus = null;
    };
    window.__setMockSpeakerLabel = (speakerId, label) => {
      const speaker = speakers.find((candidate) => candidate.id === speakerId);
      if (!speaker) throw new Error("Mock speaker not found");
      speaker.label = label;
      for (const segments of Object.values(native.segments)) {
        for (const segment of segments) {
          if (segment.speaker_id === speakerId) segment.speaker_label = label;
        }
      }
    };
    window.__addConversationFixture = ({
      sessionId = "session-large",
      title = "Large archive meeting",
      segmentCount = 2_163,
      unknownIndex = null,
    } = {}) => {
      const fixtureSession = {
        id: sessionId,
        created_at: "2026-07-24T08:00:00Z",
        title,
        duration_ms: 7_200_000,
        transcript: "Person 1: scale-only transcript phrase",
        processing_status: null,
        processing_error: null,
        processing_run_id: null,
        recoverable_audio: false,
      };
      native.sessions = [
        fixtureSession,
        ...native.sessions.filter((candidate) => candidate.id !== sessionId),
      ];
      recapStateFor(sessionId);
      native.segments[fixtureSession.id] = Array.from(
        { length: segmentCount },
        (_, index) => ({
        id: sessionId + "-segment-" + index,
        session_id: fixtureSession.id,
        start_ms: index * 3_000,
        end_ms: index * 3_000 + 2_500,
        speaker_id:
          index === unknownIndex ? null : "large-speaker-" + (index % 260),
        speaker_label:
          index === unknownIndex
            ? "Unknown speaker"
            : "Person " + String((index % 260) + 1).padStart(3, "0"),
        text: "Intervention " + (index + 1) + " with enough text to exercise layout.",
      }),
      );
      for (let index = 0; index < 260; index += 1) {
        if (speakers.some((speaker) => speaker.id === "large-speaker-" + index)) {
          continue;
        }
        speakers.push({
          id: "large-speaker-" + index,
          label: "Person " + String(index + 1).padStart(3, "0"),
          created_at: "2026-07-24T08:00:00Z",
          last_seen_at: "2026-07-24T08:00:00Z",
          sample_count: 0,
          embedding_count: 1,
          conversation_count: 1,
        });
      }
      return fixtureSession.id;
    };
    window.__addVoiceSplitFixture = () => {
      const sessionId = "session-voice-split";
      const fixtureSession = {
        id: sessionId,
        created_at: "2026-07-25T08:00:00Z",
        title: "Mixed voice review",
        duration_ms: 30_000,
        transcript:
          "Speaker 1: First turn\nSpeaker 1: Second turn\nSpeaker 1: Third turn\nSpeaker 1: Fourth turn",
        processing_status: null,
        processing_error: null,
        processing_run_id: null,
        recoverable_audio: false,
      };
      native.sessions = [fixtureSession, ...native.sessions];
      native.segments[sessionId] = [
        {
          id: "split-segment-1",
          session_id: sessionId,
          start_ms: 0,
          end_ms: 5_000,
          speaker_id: null,
          speaker_label: "Speaker 1",
          voice_group_id: "voice-group-mixed",
          text: "First turn from the main voice.",
        },
        {
          id: "split-segment-2",
          session_id: sessionId,
          start_ms: 7_000,
          end_ms: 12_000,
          speaker_id: null,
          speaker_label: "Speaker 1",
          voice_group_id: "voice-group-mixed",
          text: "Second turn from another local cluster.",
        },
        {
          id: "split-segment-3",
          session_id: sessionId,
          start_ms: 14_000,
          end_ms: 20_000,
          speaker_id: null,
          speaker_label: "Speaker 1",
          voice_group_id: "voice-group-mixed",
          text: "Third turn from the main voice.",
        },
        {
          id: "split-segment-4",
          session_id: sessionId,
          start_ms: 22_000,
          end_ms: 28_000,
          speaker_id: null,
          speaker_label: "Speaker 1",
          voice_group_id: "voice-group-mixed",
          text: "Fourth turn from the other local cluster.",
        },
      ];
      native.voiceGroups[sessionId] = [
        {
          id: "voice-group-mixed",
          session_id: sessionId,
          provider_speaker_label: "Speaker 1",
          cluster_index: 0,
          resulting_speaker_id: null,
          resulting_speaker_label: null,
          status: "meeting_local_no_safe_speech",
          selected_duration_ms: 20_000,
          selected_window_count: 4,
          consistency_score: 0.91,
          model_version: "wespeaker-ecapa512-lm-v4-vad",
          split_status: "suggested",
          split_clusters: [
            ["split-segment-1", "split-segment-3"],
            ["split-segment-2", "split-segment-4"],
          ],
          intervention_count: 4,
          voice_observation_count: 4,
        },
      ];
      return sessionId;
    };
    window.__addNoSafeVoiceFixture = (includeAssignedTurn = true) => {
      const sessionId = "session-no-safe-voice";
      const fixtureSession = {
        id: sessionId,
        created_at: "2026-07-26T08:00:00Z",
        title: "Provider-only speaker labels",
        duration_ms: includeAssignedTurn ? 18_000 : 12_000,
        transcript: includeAssignedTurn
          ? "speaker_2: First turn\nspeaker_2: Second turn\nAlice: Already assigned turn"
          : "speaker_2: First turn\nspeaker_2: Second turn",
        processing_status: null,
        processing_error: null,
        processing_run_id: null,
        recoverable_audio: false,
      };
      native.sessions = [
        fixtureSession,
        ...native.sessions.filter((candidate) => candidate.id !== sessionId),
      ];
      recapStateFor(sessionId);
      native.segments[sessionId] = [
        {
          id: "no-safe-segment-1",
          session_id: sessionId,
          start_ms: 0,
          end_ms: 5_000,
          speaker_id: null,
          speaker_label: "speaker_2",
          voice_group_id: "voice-group-no-safe",
          text: "First provider-only turn.",
        },
        {
          id: "no-safe-segment-2",
          session_id: sessionId,
          start_ms: 6_000,
          end_ms: 11_000,
          speaker_id: null,
          speaker_label: "speaker_2",
          voice_group_id: "voice-group-no-safe",
          text: "Second provider-only turn.",
        },
      ];
      if (includeAssignedTurn) {
        native.segments[sessionId].push({
          id: "no-safe-segment-assigned",
          session_id: sessionId,
          start_ms: 12_000,
          end_ms: 17_000,
          speaker_id: "speaker-alice",
          speaker_label: "Alice",
          voice_group_id: "voice-group-no-safe",
          text: "Already assigned turn.",
        });
      }
      native.voiceGroups[sessionId] = [
        {
          id: "voice-group-no-safe",
          session_id: sessionId,
          provider_speaker_label: "speaker_2",
          cluster_index: 0,
          resulting_speaker_id: null,
          resulting_speaker_label: null,
          status: "meeting_local_no_safe_speech",
          selected_duration_ms: 0,
          selected_window_count: 0,
          consistency_score: null,
          model_version: "wespeaker-ecapa512-lm-v4-vad",
          split_status: "none",
          split_clusters: [],
          intervention_count: includeAssignedTurn ? 3 : 2,
          voice_observation_count: 0,
        },
      ];
      return sessionId;
    };
    window.__addLargeConversation = () =>
      window.__addConversationFixture({
        sessionId: "session-large",
        title: "Large archive meeting",
        segmentCount: 2_163,
        unknownIndex: 720,
      });
    window.__deferConversationLoad = (sessionId) => {
      conversationLoadGates.set(
        sessionId,
        new Promise((resolve) => releaseConversationLoads.set(sessionId, resolve)),
      );
    };
    window.__releaseConversationLoad = (sessionId) => {
      releaseConversationLoads.get(sessionId)?.();
      releaseConversationLoads.delete(sessionId);
      conversationLoadGates.delete(sessionId);
    };
    window.__deferJamieChoice = () => {
      jamieChoiceGate = new Promise((resolve) => {
        releaseJamieChoice = resolve;
      });
    };
    window.__releaseJamieChoice = () => {
      releaseJamieChoice?.();
      jamieChoiceGate = null;
      releaseJamieChoice = null;
    };
    window.__deferJamieInspection = () => {
      jamieInspectionGate = new Promise((resolve) => {
        releaseJamieInspection = resolve;
      });
    };
    window.__releaseJamieInspection = () => {
      releaseJamieInspection?.();
      jamieInspectionGate = null;
      releaseJamieInspection = null;
    };
  }, { session: oldSession });
}

test.beforeEach(async ({ page }) => {
  await installTauriMock(page);
  await page.goto("/");
  await expect(page.getByText("Recall is ready", { exact: false })).toBeAttached();
});

test("recap type manager protects dirty edits, restores built-ins, and enables the split action", async ({
  page,
}) => {
  await expect(page.getByRole("button", { name: "Choose a custom recap type" })).toBeHidden();
  await page.getByRole("button", { name: "Recap types" }).click();
  const manager = page.getByRole("dialog", { name: "Recap types" });
  await expect(manager).toBeVisible();
  await expect(manager.locator(".recap-type-option strong")).toHaveText([
    "Executive summary",
    "Full summary",
    "Actions",
  ]);

  await manager.getByRole("option", { name: /Executive summary/ }).click();
  const prompt = manager.getByLabel("Instructions");
  const shippedPrompt = await prompt.inputValue();
  await expect(manager.locator(".recap-prompt-variable code")).toHaveText([
    "{{meeting_date}}",
    "{{meeting_time}}",
    "{{meeting_datetime}}",
  ]);
  await expect(
    manager.getByRole("button", {
      name: "Insert Meeting date variable {{meeting_date}}",
    }),
  ).toHaveAttribute("title", /saved local date.*Example: 2026\/07\/23/s);
  await prompt.fill("Unsaved prompt edit");
  await manager.getByRole("button", { name: "Close recap types" }).click();
  const discard = page.getByRole("alertdialog", { name: "Discard unsaved changes?" });
  await expect(discard).toBeVisible();
  await discard.getByRole("button", { name: "Cancel" }).click();
  await expect(manager).toBeVisible();
  await expect(prompt).toHaveValue("Unsaved prompt edit");

  await prompt.fill("Saved replacement prompt");
  await manager.getByRole("button", { name: "Save", exact: true }).click();
  await expect(manager.getByText("Saved.", { exact: true })).toBeVisible();
  await manager.getByRole("button", { name: "Restore default" }).click();
  await page
    .getByRole("alertdialog", { name: "Restore the shipped prompt?" })
    .getByRole("button", { name: "Restore default" })
    .click();
  await expect(prompt).toHaveValue(shippedPrompt);

  await manager.getByRole("button", { name: "New custom type" }).click();
  await manager.getByLabel("Name").fill("  Risk   review  ");
  await manager.getByLabel("Instructions").fill("Report risks for DATE.");
  await manager.getByLabel("Instructions").evaluate((textarea) => {
    const start = textarea.value.indexOf("DATE");
    textarea.focus();
    textarea.setSelectionRange(start, start + 4);
  });
  await manager
    .getByRole("button", { name: "Insert Meeting date variable {{meeting_date}}" })
    .click();
  await expect(manager.getByLabel("Instructions")).toHaveValue(
    "Report risks for {{meeting_date}}.",
  );
  await expect(manager.getByLabel("Instructions")).toBeFocused();
  expect(
    await manager.getByLabel("Instructions").evaluate((textarea) => ({
      start: textarea.selectionStart,
      end: textarea.selectionEnd,
    })),
  ).toEqual({ start: 33, end: 33 });
  await manager.getByRole("button", { name: "Save", exact: true }).click();
  await expect(manager.getByRole("option", { name: /Risk review/ })).toBeVisible();
  await manager
    .getByLabel("Instructions")
    .fill("Focus on material risks, mitigations, and owners for {{meeting_date}}.");
  await manager.getByRole("button", { name: "Save", exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => window.__mockCommandCount("update_recap_type")))
    .toBe(2);
  await manager.getByRole("button", { name: "Close recap types" }).click();
  await expect(manager).toBeHidden();

  const menuButton = page.getByRole("button", { name: "Choose a custom recap type" });
  await expect(menuButton).toBeVisible();
  await menuButton.click();
  const riskReview = page.getByRole("menuitem", { name: "Risk review" });
  await expect(riskReview).toBeVisible();
  await expect(page.getByRole("button", { name: "Recap", exact: true })).toBeVisible();
  await riskReview.click();
  await expect(page.getByRole("tab", { name: "Risk review" })).toBeVisible();
  expect(await page.evaluate(() => window.__mockCommandCount("generate_custom_recap"))).toBe(1);
});

test("recap prompt editing remains available when the variable registry cannot load", async ({
  page,
}) => {
  await page.evaluate(() => window.__setMockPromptVariablesUnavailable(true));
  await page.getByRole("button", { name: "Recap types" }).click();
  const manager = page.getByRole("dialog", { name: "Recap types" });
  await expect(manager.getByText(
    "Variables are unavailable. You can still edit and save the prompt.",
  )).toBeVisible();
  await manager.getByLabel("Instructions").fill("Prompt editing still works.");
  await expect(manager.getByLabel("Instructions")).toHaveValue("Prompt editing still works.");
});

test("custom recap reuses participant review, renders hostile Markdown safely, exports both formats, and preserves deleted snapshots", async ({
  page,
}) => {
  const markdown = [
    "# Risk review",
    "",
    "Decision **approved** with *uncertain* timing and `owner_id`.",
    "",
    "- First action",
    "- Second action",
    "",
    "> Quoted context",
    "",
    '<script>window.__recapScriptExecuted = true</script>',
    '<img src=x onerror="window.__recapImageExecuted = true">',
  ].join("\n");
  const recapTypeId = await page.evaluate((content) => {
    const id = window.__addMockRecapType("Risk review", "Focus on risk.");
    window.__setMockCustomMarkdown(id, content);
    window.__setMockUnresolvedProfiles("session-old", ["VOICE12"]);
    return id;
  }, markdown);
  await page.getByRole("button", { name: "Recap types" }).click();
  await page.getByRole("button", { name: "Close recap types" }).click();
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();

  await page.getByRole("button", { name: "Choose a custom recap type" }).click();
  await page.getByRole("menuitem", { name: "Risk review" }).click();
  const participantReview = page.getByRole("dialog", { name: "Name participants first?" });
  await expect(participantReview).toBeVisible();
  expect(await page.evaluate(() => window.__mockCommandCount("generate_custom_recap"))).toBe(0);
  await participantReview.getByRole("button", { name: "Recap anyway" }).click();
  await expect(page.getByRole("tab", { name: "Risk review" })).toBeVisible();
  await expect(page.locator("#generatedTitle")).toHaveText("Risk review");
  await expect(page.locator("#generatedContent")).toContainText(
    "<script>window.__recapScriptExecuted = true</script>",
  );
  await expect(page.locator("#generatedContent script")).toHaveCount(0);
  await expect(page.locator("#generatedContent img")).toHaveCount(0);
  expect(
    await page.evaluate(() => [window.__recapScriptExecuted, window.__recapImageExecuted]),
  ).toEqual([undefined, undefined]);
  expect(
    await page.evaluate(() => window.__mockLastCommandArgs("generate_custom_recap")),
  ).toEqual({ sessionId: "session-old", recapTypeId, allowUnresolved: true });

  await page.getByRole("button", { name: "Copy Markdown" }).click();
  await expect.poll(() => page.evaluate(() => window.__mockClipboardText())).toBe(markdown);
  await page.getByRole("button", { name: "Copy text" }).click();
  await expect
    .poll(() => page.evaluate(() => window.__mockClipboardText()))
    .toBe(
      [
        "Risk review",
        "",
        "Decision approved with uncertain timing and owner_id.",
        "",
        "- First action\n- Second action",
        "",
        "Quoted context",
        "",
        '<script>window.__recapScriptExecuted = true</script>\n<img src=x onerror="window.__recapImageExecuted = true">',
      ].join("\n"),
    );

  const replacementMarkdown = "## Risk review\n\nReplacement content after regeneration.";
  await page.evaluate(({ id, content }) => {
    window.__setMockUnresolvedProfiles("session-old", []);
    window.__setMockCustomMarkdown(id, content);
    window.__failNextCustomRecap("session-old", "Replacement failed validation");
  }, { id: recapTypeId, content: replacementMarkdown });
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();
  await page.getByRole("button", { name: "Choose a custom recap type" }).click();
  await page.getByRole("menuitem", { name: "Risk review" }).click();
  await expect(page.getByText("Risk review failed", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Risk review" }).click();
  await expect(page.locator("#generatedContent")).toContainText("Decision approved");
  expect(await page.evaluate(() => window.__mockCustomRecaps("session-old").length)).toBe(1);

  await page.getByRole("button", { name: "Choose a custom recap type" }).click();
  await page.getByRole("menuitem", { name: "Risk review" }).click();
  await expect(page.locator("#generatedContent")).toContainText(
    "Replacement content after regeneration.",
  );
  expect(
    await page.evaluate((id) =>
      window
        .__mockCustomRecaps("session-old")
        .find((recap) => recap.recap_type_id === id)?.content_markdown,
    recapTypeId),
  ).toBe(replacementMarkdown);

  await page.evaluate((id) => {
    window.__setMockCustomRecapStale("session-old", id, true);
  }, recapTypeId);
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();
  await page.getByRole("tab", { name: "Risk review" }).click();
  await expect(page.getByText("This custom recap is out of date.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Regenerate" })).toBeVisible();

  await page.getByRole("button", { name: "Recap types" }).click();
  const manager = page.getByRole("dialog", { name: "Recap types" });
  await manager.getByRole("option", { name: /Risk review/ }).click();
  await manager.getByRole("button", { name: "Delete type" }).click();
  const deletion = page.getByRole("alertdialog", { name: "Delete this recap type?" });
  await expect(deletion).toContainText("Recaps already generated for meetings keep their saved names and content.");
  await deletion.getByRole("button", { name: "Delete type" }).click();
  await manager.getByRole("button", { name: "Close recap types" }).click();
  await expect(page.getByRole("button", { name: "Choose a custom recap type" })).toBeHidden();
  await expect(page.getByRole("tab", { name: "Risk review" })).toBeVisible();
  await expect(page.getByText("Its recap type has since been deleted.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Regenerate" })).toBeHidden();
});

test("custom recap tabs sort duplicate names and background generation reloads after navigation", async ({
  page,
}) => {
  const typeIds = await page.evaluate(() => [
    window.__addMockRecapType("Risk review"),
    window.__addMockRecapType("Board note"),
    window.__addMockRecapType("Risk review"),
  ]);
  await page.getByRole("button", { name: "Recap types" }).click();
  await page.getByRole("button", { name: "Close recap types" }).click();

  const chooseType = async (name, index = 0) => {
    await page.getByRole("button", { name: "Choose a custom recap type" }).click();
    await page.getByRole("menuitem", { name }).nth(index).click();
  };
  await chooseType("Risk review", 1);
  await expect(page.getByRole("tab", { name: "Risk review" })).toHaveCount(1);
  await chooseType("Board note");
  await chooseType("Risk review", 0);
  await expect(page.getByRole("tab", { name: "Risk review" })).toHaveCount(2);
  await expect(page.locator("#recapTabs .recap-tab:visible")).toHaveText([
    "Transcript",
    "Board note",
    "Risk review",
    "Risk review",
  ]);
  expect(await page.evaluate(() => window.__mockCustomRecaps("session-old").length)).toBe(3);

  await page.evaluate((boardTypeId) => {
    window.__deferCustomRecap("session-old");
    window.__setMockCustomMarkdown(boardTypeId, "## Board note\n\nBackground board result.");
    window.__addConversationFixture({
      sessionId: "session-parallel",
      title: "Parallel conversation",
      segmentCount: 2,
    });
  }, typeIds[1]);
  await chooseType("Board note");
  await expect(page.getByText("Creating Board note", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Parallel conversation/ }).click();
  await chooseType("Risk review", 0);
  await expect(page.getByRole("tab", { name: "Risk review" })).toBeVisible();
  await page.evaluate(() => window.__releaseCustomRecap("session-old"));
  await expect
    .poll(() =>
      page.evaluate((boardTypeId) =>
        window
          .__mockCustomRecaps("session-old")
          .find((recap) => recap.recap_type_id === boardTypeId)?.content_markdown,
      typeIds[1]),
    )
    .toContain("Background board result");
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();
  await expect(page.getByRole("tab", { name: "Board note" })).toBeVisible();
  expect(await page.evaluate(() => window.__mockCustomRecaps("session-parallel").length)).toBe(1);
  expect(typeIds).toHaveLength(3);
});

test("live speaker turns keep code switches inline and provide a complete preferred-language line", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Привет and welcome друзья\nSpeaker 2: Guten Tag",
      final_text: "",
      turns: [
        {
          id: "speaker-one-turn",
          sequence: 0,
          speaker: "Speaker 1",
          segments: [
            {
              id: "speaker-one-ru-1",
              source_text: "Привет",
              source_language: "ru-RU",
              source_final: false,
              translation: { text: "Hello", source_language: "ru", is_final: false },
            },
            {
              id: "speaker-one-en",
              source_text: "and welcome",
              source_language: "en-US",
              source_final: false,
              translation: null,
            },
            {
              id: "speaker-one-ru-2",
              source_text: "друзья",
              source_language: "ru-RU",
              source_final: false,
              translation: { text: "friends", source_language: "ru", is_final: false },
            },
          ],
        },
        {
          id: "speaker-two-turn",
          sequence: 1,
          speaker: "Speaker 2",
          segments: [
            {
              id: "speaker-two-de",
              source_text: "Guten Tag",
              source_language: "de-DE",
              source_final: false,
              translation: null,
            },
          ],
        },
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  const firstTurn = page.locator('[data-live-caption-id="speaker-one-turn"]');
  await expect(firstTurn.locator("[data-live-caption-source]")).toHaveCount(1);
  await expect(firstTurn.locator('[data-live-caption-run="source"]')).toHaveCount(3);
  await expect(firstTurn.locator("[data-live-caption-translation]")).toHaveText(
    "[ru] Hello [en] and welcome [ru] friends",
  );
  await expect(page.locator('[data-live-caption-id="speaker-two-turn"] [data-live-caption-translation]')).toHaveCount(0);
  await expect(page.locator(".live-caption-passage")).toHaveCount(2);
  const styles = await firstTurn.evaluate((turn) => {
    const sourceParagraph = turn.querySelector("[data-live-caption-source]");
    const preferredParagraph = turn.querySelector("[data-live-caption-translation]");
    const firstSource = turn.querySelector('[data-live-caption-run="source"][data-live-caption-language="ru"]');
    const firstPreferred = turn.querySelector('[data-live-caption-run="preferred"][data-live-caption-language="ru"]');
    const englishSource = turn.querySelector('[data-live-caption-run="source"][data-live-caption-language="en"]');
    const values = (element) => [getComputedStyle(element).backgroundColor, getComputedStyle(element).color];
    return {
      sourceBackground: getComputedStyle(sourceParagraph).backgroundColor,
      preferredBackground: getComputedStyle(preferredParagraph).backgroundColor,
      ruSource: values(firstSource),
      ruPreferred: values(firstPreferred),
      enSource: values(englishSource),
    };
  });
  expect(styles.sourceBackground).not.toBe("rgba(0, 0, 0, 0)");
  expect(styles.preferredBackground).not.toBe("rgba(0, 0, 0, 0)");
  expect(styles.sourceBackground).not.toEqual(styles.preferredBackground);
  expect(styles.ruSource).toEqual(styles.ruPreferred);
  expect(styles.ruSource).not.toEqual(styles.enSource);

  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Привет and welcome друзья",
      final_text: "",
      turns: [
        {
          id: "speaker-one-turn",
          sequence: 0,
          speaker: "Speaker 1",
          segments: [
            {
              id: "speaker-one-ru-1",
              source_text: "Привет",
              source_language: "ru",
              source_final: false,
              translation: { text: "Hel", source_language: "ru", is_final: false },
            },
            {
              id: "speaker-one-en",
              source_text: "and welcome",
              source_language: "en",
              source_final: false,
              translation: null,
            },
            {
              id: "speaker-one-ru-2",
              source_text: "друзья",
              source_language: "ru",
              source_final: false,
              translation: { text: "friends", source_language: "ru", is_final: false },
            },
          ],
        },
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(firstTurn.locator("[data-live-caption-translation]")).toHaveText(
    "[ru] Hel [en] and welcome [ru] friends",
  );
  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Привет and welcome друзья",
      final_text: "",
      turns: [
        {
          id: "speaker-one-turn",
          sequence: 0,
          speaker: "Speaker 1",
          segments: [
            {
              id: "speaker-one-ru-1",
              source_text: "Привет",
              source_language: "ru",
              source_final: false,
              translation: { text: "Hello", source_language: "ru", is_final: false },
            },
            {
              id: "speaker-one-en",
              source_text: "and welcome",
              source_language: "en",
              source_final: false,
              translation: null,
            },
            {
              id: "speaker-one-ru-2",
              source_text: "друзья",
              source_language: "ru",
              source_final: false,
              translation: { text: "friends", source_language: "ru", is_final: false },
            },
          ],
        },
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(firstTurn.locator("[data-live-caption-translation]")).toHaveText(
    "[ru] Hello [en] and welcome [ru] friends",
  );
  await page.getByRole("button", { name: "Stop recording" }).click();
});

test("a revised provisional source tail keeps pace with its live translation", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  const emitSnapshot = (revision, sourceText, sourceFinal, translationText) =>
    page.evaluate(
      ({ revision, sourceText, sourceFinal, translationText }) =>
        window.__emitTauri("live-transcription", {
          revision,
          final_audio_proc_ms: sourceFinal ? 1_000 : 1_500,
          total_audio_proc_ms: 2_000,
          status: "Live",
          text: "Speaker 1: " + sourceText,
          final_text: sourceFinal ? "Speaker 1: " + sourceText : "Speaker 1: Guten Morgen",
          turns: [
            {
              id: "live-turn-0",
              sequence: 0,
              speaker: "Speaker 1",
              segments: [
                {
                  id: "live-turn-0-segment-0",
                  source_text: sourceText,
                  source_language: "de",
                  source_final: sourceFinal,
                  translation: {
                    text: translationText,
                    source_language: "de",
                    is_final: false,
                  },
                },
              ],
            },
          ],
          target_language: "en",
          translation_warning: null,
          finished: false,
          error: null,
        }),
      { revision, sourceText, sourceFinal, translationText },
    );

  await emitSnapshot(100, "Guten Morgen", true, "Good morning");
  await emitSnapshot(101, "Guten Morgen, ich denke", false, "Good morning, I think");
  await emitSnapshot(102, "Guten Morgen, wir denken", false, "Good morning, we think");

  const turn = page.locator('[data-live-caption-id="live-turn-0"]');
  await expect(turn.locator("[data-live-caption-source]")).toContainText(
    "Guten Morgen, wir denken",
  );
  await expect(turn.locator("[data-live-caption-source]")).not.toContainText("ich denke");
  await expect(turn.locator("[data-live-caption-translation]")).toContainText(
    "[de] Good morning, we think",
  );

  await emitSnapshot(101, "Guten Morgen, ich denke", false, "Good morning, I think");
  await expect(turn.locator("[data-live-caption-source]")).toContainText(
    "Guten Morgen, wir denken",
  );
  await expect(turn.locator("[data-live-caption-translation]")).toContainText(
    "[de] Good morning, we think",
  );
  await page.getByRole("button", { name: "Stop recording" }).click();
});

test("live legacy passages remain fluid and available while working in conversation history", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  const currentRecording = page.locator('[data-current-recording="true"]');
  await expect(currentRecording).toBeVisible();
  await expect(currentRecording).toHaveClass(/selected/);
  await expect(page.getByRole("heading", { name: "Connecting…" })).toBeVisible();

  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Привет\nSpeaker 1: Hello\nSpeaker 2: Guten Tag",
      final_text: "",
      passages: [
        {
          id: "live-passage-0",
          sequence: 0,
          speaker: "Speaker 1",
          source_text: "Привет",
          source_language: "ru-RU",
          source_final: false,
          translation: {
            text: "Hello",
            source_language: "ru-RU",
            is_final: false,
          },
        },
        {
          id: "live-passage-1",
          sequence: 1,
          speaker: "Speaker 1",
          source_text: "Hello",
          source_language: "en",
          source_final: false,
          translation: null,
        },
        {
          id: "live-provisional-de",
          sequence: 2,
          speaker: "Speaker 2",
          source_text: "Guten Tag",
          source_language: "de-DE",
          source_final: false,
          translation: null,
        },
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(page.locator("[data-live-caption-source]").first()).toContainText("Привет");
  await expect(page.getByText("[ru] Hello", { exact: true })).toBeVisible();
  await expect(page.locator("[data-live-caption-translation]")).toHaveCount(1);
  const firstRecordingStyles = await page.evaluate(() => {
    const values = (element) =>
      [
        element.style.getPropertyValue("--live-caption-language-bg"),
        element.style.getPropertyValue("--live-caption-language-fg"),
        element.style.getPropertyValue("--live-caption-language-border"),
      ];
    const computedValues = (element) => {
      const style = getComputedStyle(element);
      return [style.backgroundColor, style.color, style.borderLeftColor];
    };
    const russian = document.querySelector('[data-live-caption-id="live-passage-0"]');
    const english = document.querySelector('[data-live-caption-id="live-passage-1"]');
    const russianSource = russian.querySelector('[data-live-caption-run="source"]');
    const russianTranslation = russian.querySelector('[data-live-caption-run="preferred"]');
    const englishSource = english.querySelector('[data-live-caption-run="source"]');
    window.__firstRecordingRuStyle = values(russianSource);
    return {
      russianSource: values(russianSource),
      russianTranslation: values(russianTranslation),
      russianSourceComputed: computedValues(russianSource),
      russianTranslationComputed: computedValues(russianTranslation),
      englishSource: values(englishSource),
      englishSourceComputed: computedValues(englishSource),
      englishTranslation: english.querySelector("[data-live-caption-translation]"),
    };
  });
  expect(firstRecordingStyles.russianSource).toEqual(firstRecordingStyles.russianTranslation);
  expect(firstRecordingStyles.russianSourceComputed).toEqual(
    firstRecordingStyles.russianTranslationComputed,
  );
  expect(firstRecordingStyles.russianSource).toEqual(["#edf7f2", "#2d6957", "#2d6957"]);
  expect(firstRecordingStyles.englishSource).not.toEqual(firstRecordingStyles.russianSource);
  expect(firstRecordingStyles.englishSourceComputed[0]).not.toEqual(
    firstRecordingStyles.russianSourceComputed[0],
  );
  expect(firstRecordingStyles.englishTranslation).toBeNull();
  const provisionalRowWasRetained = await page.evaluate(() => {
    window.__provisionalLiveRow = document.querySelector('[data-live-caption-id="live-passage-0"]');
    return Boolean(window.__provisionalLiveRow);
  });
  expect(provisionalRowWasRetained).toBe(true);

  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Speaker 1: Привет\nSpeaker 1: Hello\nSpeaker 2: Bonjour\nSpeaker 3: Neutral",
      final_text: "Speaker 1: Привет\nSpeaker 1: Hello",
      passages: [
        {
          id: "live-passage-0",
          sequence: 0,
          speaker: "Speaker 1",
          source_text: "Привет",
          source_language: "ru-RU",
          source_final: true,
          translation: {
            text: "Hello",
            source_language: "ru-RU",
            is_final: true,
          },
        },
        {
          id: "live-passage-1",
          sequence: 1,
          speaker: "Speaker 1",
          source_text: "Hello",
          source_language: "en",
          source_final: true,
          translation: null,
        },
        {
          id: "live-passage-2",
          sequence: 2,
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
          id: "live-invalid",
          sequence: 3,
          speaker: "Speaker 3",
          source_text: "Neutral",
          source_language: "not a language",
          source_final: false,
          translation: {
            text: "Must not render",
            source_language: "fr-FR",
            is_final: false,
          },
        },
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(page.getByText("[ru] Hello", { exact: true })).toBeVisible();
  await expect(page.getByText("[fr] Good morning", { exact: true })).toBeVisible();
  await expect(page.locator("[data-live-caption-translation]")).toHaveCount(2);
  await expect(page.locator(".live-caption-passage")).toHaveCount(4);
  const laterStyles = await page.evaluate(() => {
    const values = (element) =>
      [
        element.style.getPropertyValue("--live-caption-language-bg"),
        element.style.getPropertyValue("--live-caption-language-fg"),
        element.style.getPropertyValue("--live-caption-language-border"),
      ];
    const rowStyle = (id) => values(document.querySelector(`[data-live-caption-id="${id}"] [data-live-caption-run="source"]`));
    const french = document.querySelector('[data-live-caption-id="live-passage-2"]');
    const invalid = document.querySelector('[data-live-caption-id="live-invalid"] [data-live-caption-run="source"]');
    return {
      russian: rowStyle("live-passage-0"),
      frenchSource: values(french.querySelector('[data-live-caption-run="source"]')),
      frenchTranslation: values(french.querySelector('[data-live-caption-run="preferred"]')),
      invalid: values(invalid),
      invalidTranslation: document.querySelector('[data-live-caption-id="live-invalid"] [data-live-caption-translation]'),
    };
  });
  expect(laterStyles.russian).toEqual(firstRecordingStyles.russianSource);
  expect(laterStyles.frenchSource).toEqual(laterStyles.frenchTranslation);
  expect(laterStyles.frenchSource).not.toEqual(firstRecordingStyles.russianSource);
  expect(laterStyles.frenchSource).toEqual(["#f5eef7", "#785776", "#785776"]);
  expect(laterStyles.invalid).toEqual(["", "", ""]);
  expect(laterStyles.invalidTranslation).toBeNull();
  expect(
    await page.evaluate(
      () => window.__provisionalLiveRow === document.querySelector('[data-live-caption-id="live-passage-0"]'),
    ),
  ).toBe(true);

  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      revision: 10,
      status: "Live",
      text: "Speaker 1: Привет всем\nSpeaker 1: Hello\nSpeaker 2: Bonjour\nSpeaker 3: Ciao",
      final_text: "Speaker 1: Привет всем\nSpeaker 1: Hello",
      passages: [
        {
          id: "live-passage-0",
          sequence: 0,
          speaker: "Speaker 1",
          source_text: "Привет всем",
          source_language: "ru",
          source_final: true,
          translation: {
            text: "Hello everyone",
            source_language: "ru",
            is_final: true,
          },
        },
        {
          id: "live-passage-1",
          sequence: 1,
          speaker: "Speaker 1",
          source_text: "Hello",
          source_language: "en",
          source_final: true,
          translation: null,
        },
        {
          id: "live-passage-2",
          sequence: 2,
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
          id: "live-mismatched",
          sequence: 3,
          speaker: "Speaker 3",
          source_text: "Ciao",
          source_language: "it-IT",
          source_final: false,
          translation: {
            text: "Wrong language pair",
            source_language: "fr-FR",
            is_final: false,
          },
        },
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(page.getByText("[ru] Hello everyone", { exact: true })).toBeVisible();
  await expect(page.getByText("[it] Wrong language pair", { exact: true })).toHaveCount(0);
  await expect(page.locator('[data-live-caption-id="live-passage-1"]')).toHaveClass(
    /live-caption-passage/,
  );
  const reusedRuStyle = await page.evaluate(() => {
    const row = document.querySelector('[data-live-caption-id="live-passage-0"]');
    const source = row.querySelector('[data-live-caption-run="source"]');
    const translation = row.querySelector('[data-live-caption-run="preferred"]');
    const values = (element) =>
      [
        element.style.getPropertyValue("--live-caption-language-bg"),
        element.style.getPropertyValue("--live-caption-language-fg"),
        element.style.getPropertyValue("--live-caption-language-border"),
      ];
    return { source: values(source), translation: values(translation) };
  });
  expect(reusedRuStyle.source).toEqual(firstRecordingStyles.russianSource);
  expect(reusedRuStyle.translation).toEqual(firstRecordingStyles.russianSource);

  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      revision: 9,
      status: "Live",
      text: "Speaker 1: Tampered",
      final_text: "Speaker 1: Tampered",
      passages: [
        {
          id: "live-passage-0",
          sequence: 0,
          speaker: "Speaker 1",
          source_text: "Привет",
          source_language: "ru",
          source_final: true,
          translation: {
            text: "Hello",
            source_language: "ru",
            is_final: true,
          },
        },
        {
          id: "live-passage-1",
          sequence: 1,
          speaker: "Speaker 1",
          source_text: "Hello",
          source_language: "en",
          source_final: true,
          translation: null,
        },
        {
          id: "live-passage-2",
          sequence: 2,
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
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  await expect(page.locator('[data-live-caption-id="live-passage-0"] [data-live-caption-translation]')).toContainText(
    "[ru] Hello",
  );

  await page.evaluate(() => {
    const transcript = document.getElementById("liveTranscript");
    Object.defineProperty(transcript, "scrollHeight", { configurable: true, value: 1_000 });
    Object.defineProperty(transcript, "clientHeight", { configurable: true, value: 200 });
    transcript.scrollTop = 100;
    transcript.dispatchEvent(new Event("scroll"));
  });
  await expect(page.getByRole("button", { name: "Jump to latest ↓" })).toBeVisible();

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

  await currentRecording.click();
  await expect(page.getByRole("heading", { name: "Live" })).toBeVisible();
  await expect(page.getByText("[ru] Hello everyone", { exact: true })).toBeVisible();
  await expect(page.locator("[data-live-caption-translation]")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Jump to latest ↓" })).toBeVisible();
  expect(
    await page.evaluate(() => {
      const source = document.querySelector(
        '[data-live-caption-id="live-passage-0"] [data-live-caption-run="source"]',
      );
      return [
        source.style.getPropertyValue("--live-caption-language-bg"),
        source.style.getPropertyValue("--live-caption-language-fg"),
        source.style.getPropertyValue("--live-caption-language-border"),
      ];
    }),
  ).toEqual(firstRecordingStyles.russianSource);

  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect
    .poll(() => page.evaluate(() => window.__mockSessionTranscript("session-draft")))
    .toBe("");

  await page.getByRole("button", { name: "New recording" }).click();
  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      status: "Live",
      text: "Bonjour",
      final_text: "",
      passages: [
        {
          id: "second-recording-fr",
          sequence: 0,
          speaker: "Speaker 1",
          source_text: "Bonjour",
          source_language: "fr-FR",
          source_final: false,
          translation: {
            text: "Hello",
            source_language: "fr-FR",
            is_final: false,
          },
        },
      ],
      target_language: "en",
      translation_warning: null,
      finished: false,
      error: null,
    }),
  );
  expect(
    await page.evaluate(() => {
      const source = document.querySelector(
        '[data-live-caption-id="second-recording-fr"] [data-live-caption-run="source"]',
      );
      return [
        source.style.getPropertyValue("--live-caption-language-bg"),
        source.style.getPropertyValue("--live-caption-language-fg"),
        source.style.getPropertyValue("--live-caption-language-border"),
      ];
    }),
  ).toEqual(firstRecordingStyles.russianSource);
  await page.getByRole("button", { name: "Stop recording" }).click();
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

test("a stale Stop recording state reconciles when the native recorder already stopped", async ({
  page,
}) => {
  await page.getByRole("button", { name: "New recording" }).click();
  await page.evaluate(() => {
    window.__setNativeRecording(false);
    document.getElementById("recordButton").click();
  });

  await expect(page.getByRole("button", { name: "New recording" })).toBeVisible();
  await expect(page.locator('[data-current-recording="true"]')).toHaveCount(0);
  await expect(
    page.getByText("Recording had already stopped. Recall refreshed the archive.", {
      exact: true,
    }),
  ).toBeVisible();
});

test("a missed native stop event self-heals without another user action", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  await page.evaluate(() => window.__setNativeRecording(false));

  await expect(page.getByRole("button", { name: "New recording" })).toBeVisible({
    timeout: 4_000,
  });
  await expect(page.locator('[data-current-recording="true"]')).toHaveCount(0);
  await expect(
    page.getByText("Recording had already stopped; the interface was refreshed", {
      exact: true,
    }),
  ).toBeAttached();
});

test("an old status response cannot stop a newly started recording", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  const statusCalls = await page.evaluate(() => window.__mockCommandCount("app_status"));
  await page.evaluate(() => {
    window.__setNativeRecording(false);
    window.__deferNextAppStatus();
  });
  await page.waitForFunction(
    (previous) => window.__mockCommandCount("app_status") > previous,
    statusCalls,
  );

  await page.evaluate(async () => {
    await window.__emitTauri("recording:stopped", "/tmp/old-recording.wav");
  });
  await page.getByRole("button", { name: "New recording" }).click();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();

  await page.evaluate(() => window.__releaseAppStatus());
  await page.waitForTimeout(250);
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  await expect(page.locator('[data-current-recording="true"]')).toBeVisible();
});

test("large conversations render in bounded batches with one searchable speaker picker and cache", async ({
  page,
}) => {
  await page.evaluate(() => window.__addLargeConversation());
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.locator("#refreshSpeakers").click();
  await page.getByRole("button", { name: /Large archive meeting/ }).click();

  const rows = page.locator("#segmentsList .segment");
  await expect(rows).toHaveCount(100);
  await expect(page.locator("#segmentsList textarea")).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Show next 100 interventions/ })).toBeVisible();

  await rows.first().locator(".segment-speaker-button").click();
  const picker = page.getByRole("dialog", { name: "Choose a person" });
  await expect(picker).toBeVisible();
  await expect(picker.locator(".speaker-picker-option")).toHaveCount(264);
  await picker.getByLabel("Search people and voices").fill("Person 260");
  await expect(picker.locator(".speaker-picker-option")).toHaveCount(1);
  await picker.getByRole("button", { name: "Cancel" }).click();

  await page.getByRole("button", { name: /Show next 100 interventions/ }).click();
  await expect(rows).toHaveCount(200);

  expect(
    await page.evaluate(() => window.__mockConversationLoadCount("session-large")),
  ).toBe(1);
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();
  await page.getByRole("button", { name: /Large archive meeting/ }).click();
  expect(
    await page.evaluate(() => window.__mockConversationLoadCount("session-large")),
  ).toBe(1);

  await rows.first().locator(".segment-speaker-button").click();
  await picker.getByLabel("Search people and voices").fill("Alice");
  await picker.locator(".speaker-picker-option").filter({ hasText: "Alice" }).first().click();
  await expect(rows.first().locator(".segment-speaker-button")).toHaveText("Alice");

  await rows.first().getByRole("button", { name: "Edit transcript" }).click();
  const editor = rows.first().getByLabel("Transcript intervention");
  await editor.fill("Corrected intervention text.");
  await editor.press("Meta+Enter");
  await expect(rows.first().locator(".segment-text-display")).toHaveText(
    "Corrected intervention text.",
  );

  await page.locator("#speakersList").getByRole("button", { name: "Review turns" }).click();
  await expect(rows).toHaveCount(800);
  await expect(picker).toBeVisible();
  await picker.getByRole("button", { name: "Cancel" }).click();
});

test("a 149-intervention conversation remains bounded and fully reachable", async ({
  page,
}) => {
  await page.evaluate(() =>
    window.__addConversationFixture({
      sessionId: "session-medium",
      title: "Medium archive meeting",
      segmentCount: 149,
    }),
  );
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Medium archive meeting/ }).click();
  const rows = page.locator("#segmentsList .segment");
  await expect(rows).toHaveCount(100);
  await page.getByRole("button", { name: /Show next 49 interventions/ }).click();
  await expect(rows).toHaveCount(149);
  await expect(page.locator("#loadMoreSegments")).toBeHidden();
});

test("long participant names use transcript space and remain inside the Voices pane", async ({
  page,
}) => {
  const longName = "Maria de la Trinidad Valdivieso Gonsales (IARC)";
  await page.evaluate(
    ({ speakerId, label }) => window.__setMockSpeakerLabel(speakerId, label),
    { speakerId: "speaker-alice", label: longName },
  );
  await page.locator("#refreshSpeakers").click();
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();

  const speakerButton = page.locator("#segmentsList .segment-speaker-button").first();
  await expect(speakerButton).toHaveText(longName);
  const transcriptFit = await speakerButton.evaluate((button) => {
    const row = button.closest(".segment-speaker").getBoundingClientRect();
    const bounds = button.getBoundingClientRect();
    return {
      usesAvailableWidth: bounds.width >= row.width * 0.75,
      textFits: button.scrollWidth <= button.clientWidth + 1,
    };
  });
  expect(transcriptFit).toEqual({
    usesAvailableWidth: true,
    textFits: true,
  });

  const card = page.locator("#speakersList .speaker-card").filter({ hasText: longName });
  await expect(card).toBeVisible();
  const cardLayout = await card.evaluate((element) => {
    const pane = element.closest(".people-pane").getBoundingClientRect();
    const bounds = element.getBoundingClientRect();
    const name = element.querySelector(".speaker-name");
    const nameBounds = name.getBoundingClientRect();
    const lineHeight = Number.parseFloat(getComputedStyle(name).lineHeight);
    return {
      contained: bounds.left >= pane.left - 1 && bounds.right <= pane.right + 1,
      nameWrapped: nameBounds.height > lineHeight * 1.5,
      pageContained:
        document.documentElement.scrollWidth <= document.documentElement.clientWidth,
    };
  });
  expect(cardLayout).toEqual({
    contained: true,
    nameWrapped: true,
    pageContained: true,
  });
});

test("a stale delayed conversation load cannot replace a newer selection", async ({
  page,
}) => {
  await page.evaluate(() => {
    window.__addConversationFixture({
      sessionId: "session-delayed",
      title: "Delayed archive meeting",
      segmentCount: 149,
    });
    window.__deferConversationLoad("session-delayed");
  });
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Delayed archive meeting/ }).click();
  await page.getByRole("button", { name: /Earlier planning meeting/ }).click();
  await expect(page.getByLabel("Conversation title")).toHaveValue(
    "Earlier planning meeting",
  );
  await page.evaluate(() => window.__releaseConversationLoad("session-delayed"));
  await expect(page.getByLabel("Conversation title")).toHaveValue(
    "Earlier planning meeting",
  );
});

test("metadata-only conversation search still finds transcript text on demand", async ({
  page,
}) => {
  await page.evaluate(() => window.__addLargeConversation());
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  const search = page.getByPlaceholder("Search conversations");
  await search.fill("scale-only transcript phrase");
  await expect(page.getByRole("button", { name: /Large archive meeting/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Earlier planning meeting/ })).toHaveCount(0);
  await search.fill("Earlier discussion");
  await expect(page.getByRole("button", { name: /Earlier planning meeting/ })).toBeVisible();
  expect(await page.evaluate(() => window.__mockCommandCount("search_session_ids"))).toBeGreaterThan(
    0,
  );
});

test("the conversation filter contains named people only", async ({ page }) => {
  const filter = page.getByLabel("Filter conversations by voice");
  await expect(filter.locator("option")).toHaveText(["All voices", "Alice"]);
});

test("a no-safe-voiceprint card assigns all provider turns without per-turn work", async ({
  page,
}) => {
  await page.evaluate(() => window.__addNoSafeVoiceFixture());
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Provider-only speaker labels/ }).click();

  const card = page.locator("#speakersList .speaker-card").filter({ hasText: "speaker_2" });
  await expect(card.getByText("No safe voiceprint", { exact: true })).toBeVisible();
  await expect(card.getByRole("button", { name: "Assign or name…" })).toBeVisible();
  await card.getByRole("button", { name: "Assign or name…" }).click();

  let assignment = page.getByRole("dialog", { name: "Assign or name speaker" });
  await expect(assignment).toContainText(
    "Assign 2 unresolved interventions labelled speaker_2 in this conversation",
  );
  await expect(assignment.getByLabel("Assign to").locator("option").first()).toHaveText(
    "Create a new name-only person",
  );
  await assignment.getByRole("button", { name: "Cancel" }).click();
  await expect(assignment).toBeHidden();

  await page.getByRole("button", { name: "People & Voices" }).click();
  const manager = page.getByRole("dialog", { name: "People & Voices" });
  await expect(manager.getByText("Nothing selected", { exact: true })).toBeVisible();
  await expect(manager.getByRole("button", { name: "Merge or assign selected" })).toBeDisabled();
  await page.getByRole("button", { name: "Close People and Voices" }).click();

  await card.getByRole("button", { name: "Assign or name…" }).click();
  assignment = page.getByRole("dialog", { name: "Assign or name speaker" });
  const previewCallsBeforeSelection = await page.evaluate(() =>
    window.__mockCommandCount("preview_identity_consolidation"),
  );
  await assignment.getByLabel("Assign to").selectOption("speaker-alice");
  await expect(assignment.locator("#identityFinalLabel")).toHaveValue("Alice");
  await expect(assignment.getByLabel("New person's name")).toBeHidden();
  await expect(assignment.getByRole("button", { name: "Review impact" })).toHaveCount(0);
  await expect(
    assignment.locator(".identity-impact-stat").filter({ hasText: "interventions" }),
  ).toContainText("2");
  await expect
    .poll(() => page.evaluate(() => window.__mockCommandCount("preview_identity_consolidation")))
    .toBeGreaterThan(previewCallsBeforeSelection);
  await expect(assignment.getByRole("button", { name: "Confirm changes" })).toBeEnabled();
  await assignment.getByRole("button", { name: "Confirm changes" }).click();

  await expect(assignment).toBeHidden();
  await expect(card).toHaveCount(0);
  await expect(page.locator("#segmentsList .segment-speaker-button")).toHaveText([
    "Alice",
    "Alice",
    "Alice",
  ]);
  const consolidationArgs = await page.evaluate(() =>
    window.__mockLastCommandArgs("consolidate_identities"),
  );
  expect(consolidationArgs.request.unassigned_groups[0].voice_group_id).toBe(
    "voice-group-no-safe",
  );
  expect(consolidationArgs.expectedAffectedSessionIds).toEqual([
    "session-no-safe-voice",
  ]);
  expect(consolidationArgs.expectedImpactRevision).toBe(
    await page.evaluate(() => window.__mockIdentityImpactRevision()),
  );
  expect(consolidationArgs.expectedImpactRevision).toMatch(/^mock-impact-token-\d+-\d+$/);
});

test("a no-safe-voiceprint card can create a name-only person", async ({ page }) => {
  await page.evaluate(() => window.__addNoSafeVoiceFixture(false));
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Provider-only speaker labels/ }).click();

  const card = page.locator("#speakersList .speaker-card").filter({ hasText: "speaker_2" });
  await card.getByRole("button", { name: "Assign or name…" }).click();
  const assignment = page.getByRole("dialog", { name: "Assign or name speaker" });
  await expect(assignment.getByLabel("Assign to")).toHaveValue("__new__");
  expect(
    await page.evaluate(() => window.__mockCommandCount("preview_identity_consolidation")),
  ).toBe(0);
  await assignment.getByLabel("New person's name").fill("Dmitrii");
  await expect(assignment.getByText("Ready to confirm.")).toBeVisible();
  await expect(assignment.getByRole("button", { name: "Review impact" })).toHaveCount(0);
  await assignment.getByRole("button", { name: "Confirm changes" }).click();

  await expect(card).toHaveCount(0);
  await expect(page.locator("#segmentsList .segment-speaker-button")).toHaveText([
    "Dmitrii",
    "Dmitrii",
  ]);
  const namedCard = page.locator("#speakersList .speaker-card").filter({ hasText: "Dmitrii" });
  await expect(namedCard.getByText("No current voiceprint", { exact: true })).toBeVisible();
  const request = await page.evaluate(
    () => window.__mockLastCommandArgs("consolidate_identities").request,
  );
  expect(request.target_speaker_id).toBeNull();
  expect(request.unassigned_groups).toEqual([
    {
      session_id: "session-no-safe-voice",
      speaker_label: "speaker_2",
      voice_group_id: "voice-group-no-safe",
    },
  ]);
});

test("People & Voices keeps cross-view selections and confirms an impact-reviewed merge", async ({
  page,
}) => {
  await page.getByRole("button", { name: "People & Voices" }).click();
  const manager = page.getByRole("dialog", { name: "People & Voices" });
  await expect(manager).toBeVisible();
  await expect(manager.getByText(/Showing 1–4 of 4 profiles/)).toBeVisible();

  await manager.getByRole("tab", { name: "Unassigned" }).click();
  await expect(manager.getByText("Speaker 1", { exact: true })).toBeVisible();
  await expect(manager.getByText("This conversation only", { exact: true })).toBeVisible();
  await manager
    .getByRole("checkbox", {
      name: "Select Speaker 1 in Earlier planning meeting",
    })
    .check();

  await manager.getByRole("tab", { name: "Profiles" }).click();
  await manager
    .locator('[data-identity-profile-id="speaker-alice"] input[type="checkbox"]')
    .check();
  await manager
    .locator('[data-identity-profile-id="speaker-voice"] input[type="checkbox"]')
    .check();
  await expect(manager.getByText(/2 profiles and 1 unassigned group selected/)).toBeVisible();
  await manager.getByRole("button", { name: "Merge or assign selected" }).click();

  const review = page.getByRole("dialog", { name: "Merge or assign selected" });
  await expect(review).toBeVisible();
  await expect(review.getByLabel("Person to keep")).toHaveValue("speaker-alice");
  await review.getByLabel("Final display name").fill("Alice Consolidated");
  await expect(review.getByText("Ready to confirm.")).toBeVisible();
  await expect(review.getByRole("button", { name: "Review impact" })).toHaveCount(0);
  await expect(review.getByText(/1 saved recap will be marked out of date/)).toBeVisible();
  await expect(review.locator(".identity-impact-stat")).toHaveCount(3);
  await expect(
    review.locator(".identity-impact-stat").filter({ hasText: "recaps made out of date" }),
  ).toHaveCount(0);
  await expect(review.getByText("No additional warnings.", { exact: true })).toHaveCount(0);
  await expect(review.getByText(/make and verify a local database backup/)).toBeVisible();
  await review.getByRole("button", { name: "Confirm changes" }).click();

  await expect(review).toBeHidden();
  await expect(page.locator("#activityLog")).toContainText(
    "People & Voices: Alice Consolidated saved across 1 conversations",
  );
  await expect(page.locator("#speakersList").getByText("VOICE12", { exact: true })).toHaveCount(
    0,
  );
});

test("an ambiguous voice suggestion survives review and can be accepted once", async ({
  page,
}) => {
  const currentVoices = page.locator("#speakersList");
  await expect(currentVoices.getByText("Likely Alice", { exact: true })).toBeVisible();
  await expect(currentVoices.getByText(/Best match: Alice at 0\.956/)).toBeVisible();

  await page.getByRole("button", { name: "Voice Library" }).click();
  const manager = page.getByRole("dialog", { name: "People & Voices" });
  await expect(manager.getByText("VOICE12", { exact: true })).toBeVisible();
  await expect(manager.getByText("Provisional VOICE", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Close People and Voices" }).click();
  await expect(currentVoices.getByText("Likely Alice", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Assign to Alice", exact: true }).click();

  await expect(page.getByText("Likely Alice", { exact: true })).toHaveCount(0);
  await expect(page.locator("#activityLog")).toContainText(
    "Assigned voice history to Alice; 1 compatible voiceprint activated",
  );
  const speakerButtons = page.locator("#segmentsList .segment-speaker-button");
  await expect(speakerButtons).toHaveCount(2);
  await expect
    .poll(() => speakerButtons.allTextContents())
    .toEqual(["Alice", "Alice"]);
});

test("a suggested mixed voice stays review-only until selected turns are split", async ({ page }) => {
  await page.evaluate(() => window.__addVoiceSplitFixture());
  await page.getByRole("button", { name: "Refresh conversations" }).click();
  await page.getByRole("button", { name: /Mixed voice review/ }).click();

  await expect(page.getByText("Possible mixed voice", { exact: true })).toBeVisible();
  const mixedCard = page
    .locator("#speakersList .speaker-card")
    .filter({ hasText: "Possible mixed voice" });
  await expect(mixedCard.getByRole("button", { name: "Assign or name…" })).toHaveCount(0);
  await page.getByRole("button", { name: "Review split…" }).click();

  const dialog = page.getByRole("dialog", { name: "Review a possible mixed voice" });
  await expect(dialog).toBeVisible();
  const choices = dialog.locator("input[type='checkbox']");
  await expect(choices).toHaveCount(4);
  await expect(choices.nth(0)).toBeChecked();
  await expect(choices.nth(1)).not.toBeChecked();
  await expect(choices.nth(2)).toBeChecked();
  await expect(choices.nth(3)).not.toBeChecked();
  await expect(dialog).toContainText(
    "Recall preselected the smaller locally detected voice cluster",
  );
  await expect(dialog).toContainText("Nothing is split automatically");

  await page.getByRole("button", { name: "Create separate voice" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.locator("#speakersList").getByText("VOICE13", { exact: true })).toBeVisible();
  await expect(
    page.locator("#segmentsList").getByText("Second turn from another local cluster."),
  ).toBeVisible();
  const remainingCard = page
    .locator("#speakersList .speaker-card")
    .filter({ hasText: "Speaker 1" });
  await expect(remainingCard.getByRole("button", { name: "Assign or name…" })).toBeVisible();
  expect(await page.evaluate(() => window.__mockCommandCount("split_voice_group"))).toBe(1);
});

test("voice recognition reset previews its scope and preserves conversation history", async ({ page }) => {
  await expect(page.locator("#speakersList").getByText("VOICE12", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Review reset…" }).click();

  const dialog = page.getByRole("dialog", { name: "Reset voice recognition data" });
  await expect(dialog).toBeVisible();
  await expect(
    dialog.locator(".voice-reset-stat").filter({ hasText: "global VOICE profiles removed" }),
  ).toContainText("1");
  await expect(dialog).toContainText(
    "Named people, conversations, transcript text, and historical speaker labels stay in place.",
  );
  await expect(page.getByRole("button", { name: "Create backup and reset" })).toBeEnabled();

  await page.getByRole("button", { name: "Create backup and reset" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByText("Earlier discussion", { exact: true })).toBeVisible();
  await expect(page.locator("#speakersList").getByText("VOICE12", { exact: true })).toHaveCount(0);
  await expect(
    page.locator("#speakersList").getByText("Unknown speaker", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => window.__mockCommandCount("reset_voice_recognition")),
  ).toBe(1);
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

test("meeting STT context waits for a pause, marks the handoff, and follows the recording into final processing", async ({ page }) => {
  await page.getByRole("button", { name: "New recording" }).click();
  const livePanel = page.locator("#livePanel");
  await expect(livePanel.getByLabel("Expected speakers")).toHaveValue("");
  await expect(livePanel.getByLabel("Likely languages")).toHaveValue("en, de");

  await livePanel.getByLabel("Expected speakers").selectOption("4");
  await livePanel.getByLabel("Likely languages").fill("en, bn, tr");
  await livePanel.getByRole("button", { name: "Apply to this meeting" }).click();
  await expect(page.locator("#liveContextStatus")).toContainText("Pending - waiting for a quiet pause");
  await expect
    .poll(() => page.evaluate(() => window.__mockSttContext()))
    .toEqual({ language_hints: ["en", "bn", "tr"], expected_speakers: 4 });

  await page.evaluate(() =>
    window.__emitTauri("live-transcription", {
      revision: 40,
      status: "Live",
      text: "Speaker 1: Before.\n\nSpeaker 1: After.",
      final_text: "Speaker 1: Before.\n\nSpeaker 1: After.",
      turns: [
        {
          id: "before-restart",
          sequence: 0,
          speaker: "Speaker 1",
          segments: [{
            id: "before-segment",
            source_text: "Before.",
            source_language: "en",
            source_final: true,
            translation: null,
          }],
        },
        {
          id: "after-restart",
          sequence: 1,
          speaker: "Speaker 1",
          segments: [{
            id: "after-segment",
            source_text: "After.",
            source_language: "en",
            source_final: true,
            translation: null,
          }],
        },
      ],
      markers: [{
        id: "restart-1",
        after_sequence: 0,
        text: "Live captions restarted after a pause · 4 expected speakers · likely languages: en, bn, tr",
      }],
      finished: false,
    }),
  );
  const liveItems = page.locator("#liveTranscript > *");
  await expect(liveItems).toHaveCount(3);
  await expect(liveItems.nth(0)).toContainText("Before.");
  await expect(liveItems.nth(1)).toContainText("restarted after a pause");
  await expect(liveItems.nth(2)).toContainText("After.");

  await page.evaluate(() =>
    window.__emitTauri("live-context:progress", {
      stage: "sent",
      detail: "Sent to STT; live captions resumed",
      revision: 1,
      language_hints: ["en", "bn", "tr"],
      expected_speakers: 4,
    }),
  );
  await expect(page.locator("#liveContextStatus")).toContainText("resumed");
  await expect(page.locator("#activityLog")).toContainText("Live STT context sent");

  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect
    .poll(() => page.evaluate(() => window.__mockQueuedSttContext()))
    .toEqual({ language_hints: ["en", "bn", "tr"], expected_speakers: 4 });
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
      passages: [
        {
          id: "live-passage-0",
          sequence: 0,
          speaker: "Speaker 1",
          source_text: "Original speech continues",
          source_language: "en",
          source_final: false,
          translation: null,
        },
      ],
      target_language: null,
      translation_warning:
        "Preferred language XX is unavailable for live STT translation. Original live captions will continue.",
      finished: false,
      error: null,
    }),
  );

  await expect(page.getByText("Speaker 1: Original speech continues", { exact: true })).toBeVisible();
  await expect(page.locator("[data-live-caption-translation]")).toHaveCount(0);
  await expect(page.locator("#activityLog")).toContainText("Original live captions will continue");
});

async function enableJamieImportUi(page) {
  await page.addInitScript(() => {
    window.__RECALL_ENABLE_JAMIE_IMPORT__ = true;
  });
  await page.reload();
}

test("Jamie import stays hidden in the release interface", async ({ page }) => {
  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByRole("button", { name: "Choose export…" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Resume review" })).toHaveCount(0);
  await expect(page.getByRole("dialog", { name: "Review Jamie import" })).toBeHidden();
});

test("Jamie archives are reviewed, imported with provenance, and rollback safely", async ({
  page,
}) => {
  await enableJamieImportUi(page);
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Choose export…" }).click();

  const review = page.getByRole("dialog", { name: "Review Jamie import" });
  await expect(review).toBeVisible();
  await expect(review.getByText("2", { exact: true }).first()).toBeVisible();
  await expect(review.getByText(/3 review items remaining/)).toBeVisible();
  await expect(review.getByText("Mv", { exact: true })).toBeVisible();
  await expect(review.getByText("Bob Example", { exact: true })).toBeVisible();

  await review.getByRole("button", { name: "Use source names" }).click();
  await expect(review.getByText("No source names currently need attention.")).toBeVisible();
  await review
    .getByRole("button", { name: "Exclude unreadable meetings" })
    .click();
  await expect(review.getByText("Ready to import", { exact: true })).toBeVisible();

  await review.getByRole("button", { name: "Import reviewed meetings" }).click();
  await page.getByRole("button", { name: "Import archive" }).click();
  await expect(page.getByRole("button", { name: /Imported planning call/ })).toBeVisible();

  await page.getByRole("button", { name: /Imported planning call/ }).click();
  await page.getByRole("tab", { name: "Imported executive summary" }).click();
  await expect(page.getByText("The team reviewed the plan.", { exact: true })).toBeVisible();
  await expect(page.getByText(/It was not generated by Recall/)).toBeVisible();
  await page.getByRole("tab", { name: "Imported tasks" }).click();
  await expect(page.getByText("[ ] Bob will send the draft.", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.getByText("Jamie · 1 meetings", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Roll back" }).click();
  await page.getByRole("button", { name: "Roll back import" }).click();
  await expect(page.getByRole("button", { name: /Imported planning call/ })).toHaveCount(0);
  await expect(page.getByText("Jamie · 1 meetings", { exact: true })).toBeVisible();
  await expect(page.getByText(/Rolled back/)).toBeVisible();
});

test("the native Jamie chooser precedes a full-height parsing dialog", async ({ page }) => {
  await enableJamieImportUi(page);
  await page.evaluate(() => {
    window.__deferJamieChoice();
    window.__deferJamieInspection();
  });
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Choose export…" }).click();

  const review = page.getByRole("dialog", { name: "Review Jamie import" });
  await expect(review).toBeHidden();

  await page.evaluate(() => window.__releaseJamieChoice());
  await expect(review).toBeVisible();
  await expect(review.getByText("Reading the archive", { exact: true })).toBeVisible();
  const box = await review.boundingBox();
  expect(box?.height).toBeGreaterThan(500);

  await page.evaluate(() => window.__releaseJamieInspection());
  await expect(review.getByText("Mv", { exact: true })).toBeVisible();
});

test("Jamie aliases that collide with existing people remain visible as blockers", async ({
  page,
}) => {
  await enableJamieImportUi(page);
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Choose export…" }).click();
  const review = page.getByRole("dialog", { name: "Review Jamie import" });
  const bob = review.locator(".jamie-identity-row").filter({ hasText: "Bob Example" });

  await bob.locator(".jamie-identity-action").selectOption("create_named");
  await bob.getByLabel("New Recall name for Bob Example").fill("Alice");
  await expect(
    bob.getByText(
      "That person already exists in Recall. Map the source name to the existing person instead.",
      { exact: true },
    ),
  ).toBeVisible();
  await expect(review.getByRole("button", { name: "Import reviewed meetings" })).toBeDisabled();

  const attentionOnly = review.getByRole("checkbox", {
    name: "Needs attention only",
  });
  await attentionOnly.uncheck();
  await attentionOnly.check();
  await expect(bob).toBeVisible();

  await bob.locator(".jamie-identity-action").selectOption("map_existing");
  await bob.getByLabel("Existing person for Bob Example").selectOption("speaker-alice");
  await expect(bob.locator(".jamie-identity-issue")).toHaveCount(0);
  await attentionOnly.uncheck();
  await attentionOnly.check();
  await expect(bob).toHaveCount(0);
});

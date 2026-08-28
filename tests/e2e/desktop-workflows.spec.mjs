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
    const commandCounts = {};
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
    const recapState = {
      agenda: null,
      recap: null,
      current_fingerprint: "fingerprint",
      stale: false,
      unresolved_profiles: [],
      in_flight: false,
    };
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
            recap_state: recapState,
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
          return structuredClone(recapState);
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
          return {
            target_speaker_id: targetId,
            target_label: args.request.final_label,
            source_profiles: [],
            unassigned_groups: [],
            affected_session_ids: [session.id],
            affected_conversation_count: 1,
            affected_intervention_count:
              args.request.profile_ids.length + args.request.unassigned_groups.length,
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
        }
        case "consolidate_identities": {
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
                  group.speaker_label === segment.speaker_label &&
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
    window.__mockCommandCount = (command) => commandCounts[command] || 0;
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
  await expect(page.getByText("[ru] Hello", { exact: true })).toBeVisible();
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
  await expect(review.getByLabel("Canonical person")).toHaveValue("speaker-alice");
  await review.getByLabel("Final display name").fill("Alice Consolidated");
  await review.getByRole("button", { name: "Review impact" }).click();
  await expect(review.getByText(/1 saved recap will be marked out of date/)).toBeVisible();
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

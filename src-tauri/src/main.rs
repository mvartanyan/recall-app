#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::Duration,
};

#[cfg(target_os = "macos")]
use std::process::Command;

use base64::Engine;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SampleFormat, StreamConfig,
};
use serde::{Deserialize, Serialize};
use tauri::{
    image::Image,
    menu::{MenuBuilder, MenuId, MenuItem},
    path::BaseDirectory,
    tray::TrayIconBuilder,
    Emitter, Manager, State,
};
use tokio::sync::mpsc as tokio_mpsc;
use uuid::Uuid;

mod config;
mod credentials;
mod db;
mod embedding;
mod jamie_import;
mod openai;
mod recap;
mod recap_prompt_variables;
mod soniox;
mod state;
mod vad;

use config::AppConfig;
use db::{
    AgendaMetadata, AgendaRecord, Crypto, CustomRecapSave, Db, ImportedSessionArtifact,
    RecapRecord, RecapSave, RecapType, SegmentRecord, Session, SessionSummary, SessionVoiceGroup,
    SessionVoiceGroupSave, Speaker, StoredEmbedding, VoiceGroupSplitResult, VoiceMatchDecisionSave,
    VoiceObservationSave, VoiceRecognitionResetPreview, VoiceRecognitionResetResult,
    RECAP_TYPE_KIND_CUSTOM,
};
use embedding::EMBEDDING_VERSION;
use jamie_import::{JamieImportDraft, JamieImportPreview};
use recap::{
    AgendaFingerprint, RecapSourceSegment, StandardRecapPrompts, BUILTIN_ACTIONS_ID,
    BUILTIN_EXECUTIVE_SUMMARY_ID, BUILTIN_FULL_SUMMARY_ID,
};
use recap_prompt_variables::{
    expand_recap_prompt, RecapPromptVariableContext, RecapPromptVariableDefinition,
};
use soniox::{LiveAudioMessage, LiveTranscriptEvent, TranscriptSegment};
use state::AppState;

const TARGET_SPEAKER_MS: u64 = 12_000;
const MIN_SPEAKER_MS: u64 = 3_000;
const SAMPLE_EDGE_TRIM_MS: u64 = 350;
const SAMPLE_WINDOW_MS: u64 = 4_000;
const SAMPLE_OVERLAP_TOLERANCE_MS: u64 = 200;
// Spread each bounded candidate batch across interventions before taking a
// second window from any one intervention. If a batch has no strict consistent
// majority, a small number of later batches may recover from a locally noisy
// or mixed first selection without turning the full recording into evidence.
const SAMPLE_WINDOWS_PER_CANDIDATE_BATCH: usize = 8;
const MAX_SAMPLE_CANDIDATE_BATCHES: usize = 3;
const MAX_SAMPLE_WINDOWS_PER_SPEAKER: usize =
    SAMPLE_WINDOWS_PER_CANDIDATE_BATCH * MAX_SAMPLE_CANDIDATE_BATCHES;
const SAMPLE_CONSISTENCY_THRESHOLD: f32 = 0.90;
const SAME_VOICE_SPLIT_THRESHOLD: f32 = 0.995;
const MIN_COALESCE_WINDOWS_PER_LABEL: usize = 2;
const MIN_COALESCE_DURATION_MS_PER_LABEL: u64 = 6_000;
const MIN_COALESCE_CONSISTENCY: f32 = 0.95;
const SPLIT_WITHIN_CLUSTER_THRESHOLD: f32 = 0.94;
const SPLIT_BETWEEN_CLUSTER_MAX: f32 = 0.90;
const MIN_SPLIT_INTERVENTIONS_PER_CLUSTER: usize = 2;
const MIN_SPLIT_SPEECH_MS_PER_CLUSTER: u64 = 6_000;
const MATCH_THRESHOLD: f32 = 0.94;
const STRONG_MATCH_THRESHOLD: f32 = 0.97;
const STRONG_MATCH_MARGIN: f32 = 0.03;
const PROFILE_CLAIM_MARGIN: f32 = 0.06;
const MAX_AGENDA_BYTES: usize = 50 * 1024 * 1024;
const ONBOARDING_VERSION: &str = "1";
const ALLOWED_EXTERNAL_URLS: &[&str] = &[
    "https://console.soniox.com",
    "https://platform.openai.com/api-keys",
    "https://platform.openai.com/settings/organization/billing/overview",
    "https://github.com/mvartanyan/recall-app",
];

#[derive(Debug, Clone, PartialEq)]
struct IdentityMatch {
    speaker_id: String,
    label: String,
    score: f32,
    support_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoiceMatchKind {
    Automatic,
    Suggested,
    New,
    Skipped,
}

impl VoiceMatchKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Automatic => "automatic",
            Self::Suggested => "suggested",
            Self::New => "new",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct VoiceMatchCandidate {
    kind: VoiceMatchKind,
    best: Option<IdentityMatch>,
    runner_up: Option<IdentityMatch>,
    reason: String,
}

#[derive(Debug)]
struct VoiceObservation {
    diarized_speaker: String,
    pcm: Vec<f32>,
    embedding: Vec<f32>,
    clean_window_count: usize,
    selected_duration_ms: u64,
    consistency_score: f32,
}

#[derive(Debug, Clone)]
struct InterventionVoiceObservation {
    segment_index: usize,
    start_ms: u64,
    end_ms: u64,
    embedding: Vec<f32>,
    selected_duration_ms: u64,
    consistency_score: f32,
}

#[derive(Debug, Clone)]
struct SampleWindow {
    start_ms: u64,
    end_ms: u64,
    segment_index: usize,
    candidate_batch: usize,
    pcm: Vec<f32>,
}

#[derive(Debug, PartialEq, Eq)]
struct TrustedSampleBatch {
    batch_index: usize,
    window_indices: Vec<usize>,
    candidate_count: usize,
}

#[derive(Debug)]
struct VoiceObservationGroup {
    observation_indices: Vec<usize>,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
struct VoiceGroupAssignment {
    speaker_id: Option<String>,
    display_label: String,
    group_id: String,
}

#[derive(Debug)]
struct MeetingLocalPreviewPersistence {
    diarized_speaker: String,
    result: Result<(), String>,
}

#[derive(Debug, Serialize, Clone)]
struct ProgressEvent {
    event_id: String,
    stage: String,
    detail: Option<String>,
    run_id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct RecapProgressEvent {
    session_id: String,
    stage: String,
    detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    recap_type_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recap_type_name: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct RecordingLevel {
    level: f32,
}

#[derive(Debug, Serialize)]
struct AudioDeviceInfo {
    name: String,
    is_default: bool,
}

#[derive(Debug, Serialize, Clone)]
struct RecordingStarted {
    path: String,
    device_name: String,
    sample_rate: u32,
    live_started: bool,
    stt_context: MeetingSttContext,
}

#[derive(Debug, Serialize, Clone)]
struct RecordingStopped {
    path: String,
    stt_context: MeetingSttContext,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct MeetingSttContext {
    language_hints: Vec<String>,
    expected_speakers: Option<u8>,
}

impl MeetingSttContext {
    fn normalized(self) -> Result<Self, String> {
        if self
            .expected_speakers
            .is_some_and(|count| !(1..=15).contains(&count))
        {
            return Err("Expected speakers must be between 1 and 15".into());
        }
        let mut seen = HashSet::new();
        let mut language_hints = Vec::new();
        let mut unsupported = Vec::new();
        for raw in self.language_hints {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Some(normalized) = soniox::normalize_language_hint(trimmed) else {
                unsupported.push(trimmed.to_string());
                continue;
            };
            if seen.insert(normalized.clone()) {
                language_hints.push(normalized);
            }
        }
        if !unsupported.is_empty() {
            return Err(format!(
                "Unsupported language hint{}: {}",
                if unsupported.len() == 1 { "" } else { "s" },
                unsupported.join(", ")
            ));
        }
        Ok(Self {
            language_hints,
            expected_speakers: self.expected_speakers,
        })
    }
}

#[derive(Debug, Serialize)]
struct LiveContextUpdate {
    stt_context: MeetingSttContext,
    changed: bool,
    live_restart_pending: bool,
    revision: u64,
    delivery_status: String,
}

#[derive(Debug, Serialize, Clone)]
struct QueuedTranscription {
    run_id: String,
    session_id: String,
}

#[derive(Debug, Serialize)]
struct AppStatus {
    encryption_enabled: bool,
    db_open: bool,
    needs_password: bool,
    recording: bool,
    soniox_key_configured: bool,
    openai_key_configured: bool,
    speaker_model_available: bool,
    selected_input_device: Option<String>,
    language_hints: Vec<String>,
    live_transcription: bool,
    current_stt_context: Option<MeetingSttContext>,
    live_recording_active: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PreferenceUpdate {
    selected_input_device: Option<String>,
    language_hints: Vec<String>,
    live_transcription: bool,
    openai_model: String,
    preferred_language: String,
    no_translation_languages: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RecapStateView {
    agenda: Option<AgendaMetadata>,
    recap: Option<RecapRecord>,
    custom_recaps: Vec<CustomRecapStateView>,
    current_fingerprint: String,
    stale: bool,
    unresolved_profiles: Vec<String>,
    in_flight: bool,
}

#[derive(Debug, Serialize)]
struct CustomRecapStateView {
    recap_type_id: String,
    name: String,
    content_markdown: String,
    target_language: String,
    model: String,
    source_fingerprint: String,
    input_tokens: u64,
    output_tokens: u64,
    generated_at: chrono::DateTime<chrono::Utc>,
    stale: bool,
}

#[derive(Debug, Serialize)]
struct RecapTypeView {
    id: String,
    kind: String,
    name: String,
    prompt: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

impl RecapTypeView {
    fn from_record(value: RecapType, include_prompt: bool) -> Self {
        Self {
            id: value.id,
            kind: value.kind,
            name: value.name,
            prompt: include_prompt.then_some(value.prompt),
            created_at: value.created_at,
            updated_at: value.updated_at,
        }
    }
}

#[derive(Debug, Serialize)]
struct ConversationPayload {
    session: Session,
    segments: Vec<SegmentRecord>,
    voice_groups: Vec<SessionVoiceGroup>,
    recap_state: RecapStateView,
    imported_artifact: Option<ImportedSessionArtifact>,
}

#[derive(Debug, Serialize)]
struct VoiceRecognitionResetReadiness {
    preview: VoiceRecognitionResetPreview,
    can_reset: bool,
    blockers: Vec<String>,
}

struct RecapSnapshot {
    meeting_created_at: chrono::DateTime<chrono::Utc>,
    segments: Vec<RecapSourceSegment>,
    agenda: Option<AgendaRecord>,
    source_fingerprint: String,
    unresolved_profiles: Vec<String>,
}

#[derive(Debug)]
struct AudioClip {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl AudioClip {
    fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            0
        } else {
            (self.samples.len() as u64 * 1_000) / self.sample_rate as u64
        }
    }
}

#[derive(Debug)]
struct Recorder {
    stop_tx: Option<mpsc::Sender<()>>,
    handle: Option<JoinHandle<Result<PathBuf, String>>>,
    live_tx: Option<tokio_mpsc::UnboundedSender<LiveAudioMessage>>,
    stt_context: Arc<Mutex<MeetingSttContext>>,
    stt_context_revision: AtomicU64,
}

#[derive(Debug)]
struct LiveRecordingConfig {
    api_key: String,
    options: soniox::RealtimeOptions,
}

#[derive(Default)]
struct RecordingManager {
    current: Mutex<Option<Recorder>>,
}

impl RecordingManager {
    fn start(
        &self,
        requested_device: Option<&str>,
        stt_context: MeetingSttContext,
        live: Option<LiveRecordingConfig>,
        app_handle: tauri::AppHandle,
    ) -> Result<RecordingStarted, String> {
        let mut guard = self.current.lock().map_err(|_| "Recording lock poisoned")?;
        if guard.is_some() {
            return Err("Recording is already in progress".into());
        }

        let host = cpal::default_host();
        let default_name = host
            .default_input_device()
            .and_then(|device| device.name().ok());
        let device = find_input_device(&host, requested_device)?;
        let device_name = device
            .name()
            .unwrap_or_else(|_| default_name.unwrap_or_else(|| "Default input".into()));
        let input_config = device
            .default_input_config()
            .map_err(|error| format!("Could not read input device configuration: {error}"))?;
        let sample_format = input_config.sample_format();
        let config: StreamConfig = input_config.into();
        let sample_rate = config.sample_rate.0;
        let channels = config.channels.max(1) as usize;
        let output = std::env::temp_dir().join(format!("recall-{}.wav", Uuid::new_v4()));
        let output_for_result = output.clone();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let stt_context_state = Arc::new(Mutex::new(stt_context.clone()));
        let live_tx = live.map(|live| {
            let (tx, rx) = tokio_mpsc::unbounded_channel();
            let handle = app_handle.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(error) = soniox::run_realtime(
                    live.api_key,
                    live.options,
                    sample_rate,
                    rx,
                    handle.clone(),
                )
                .await
                {
                    soniox::emit_realtime_error(&handle, error);
                }
            });
            tx
        });
        let live_started = live_tx.is_some();
        let thread_live_tx = live_tx.clone();

        let output_for_thread = output.clone();
        let callback_handle = app_handle.clone();
        let handle = thread::spawn(move || -> Result<PathBuf, String> {
            let wav_spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let (data_tx, data_rx) = mpsc::channel::<Vec<i16>>();
            let meter_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let err_handle = callback_handle.clone();
            let err_fn = move |error: cpal::StreamError| {
                eprintln!("[recording] input stream error: {error}");
                let _ = err_handle.emit("recording:error", error.to_string());
            };

            let stream = match sample_format {
                SampleFormat::F32 => {
                    let writer_tx = data_tx.clone();
                    let live_tx = thread_live_tx.clone();
                    let handle = callback_handle.clone();
                    let counter = meter_counter.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[f32], _| {
                                dispatch_samples(
                                    downmix_f32(data, channels),
                                    &writer_tx,
                                    live_tx.as_ref(),
                                    &handle,
                                    &counter,
                                );
                            },
                            err_fn,
                            None,
                        )
                        .map_err(|error| format!("Could not build input stream: {error}"))?
                }
                SampleFormat::I16 => {
                    let writer_tx = data_tx.clone();
                    let live_tx = thread_live_tx.clone();
                    let handle = callback_handle.clone();
                    let counter = meter_counter.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[i16], _| {
                                dispatch_samples(
                                    downmix_i16(data, channels),
                                    &writer_tx,
                                    live_tx.as_ref(),
                                    &handle,
                                    &counter,
                                );
                            },
                            err_fn,
                            None,
                        )
                        .map_err(|error| format!("Could not build input stream: {error}"))?
                }
                SampleFormat::U16 => {
                    let writer_tx = data_tx.clone();
                    let live_tx = thread_live_tx.clone();
                    let handle = callback_handle.clone();
                    let counter = meter_counter.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[u16], _| {
                                dispatch_samples(
                                    downmix_u16(data, channels),
                                    &writer_tx,
                                    live_tx.as_ref(),
                                    &handle,
                                    &counter,
                                );
                            },
                            err_fn,
                            None,
                        )
                        .map_err(|error| format!("Could not build input stream: {error}"))?
                }
                other => return Err(format!("Unsupported input sample format: {other:?}")),
            };

            let writer_output = output_for_thread.clone();
            let writer = thread::spawn(move || -> Result<(), String> {
                let mut writer = hound::WavWriter::create(&writer_output, wav_spec)
                    .map_err(|error| format!("Could not create WAV recording: {error}"))?;
                for chunk in data_rx {
                    for sample in chunk {
                        writer
                            .write_sample(sample)
                            .map_err(|error| format!("Could not write WAV recording: {error}"))?;
                    }
                }
                writer
                    .finalize()
                    .map_err(|error| format!("Could not finalize WAV recording: {error}"))
            });

            stream
                .play()
                .map_err(|error| format!("Could not start input stream: {error}"))?;
            let _ = stop_rx.recv();
            drop(stream);
            thread::sleep(Duration::from_millis(30));
            drop(data_tx);
            if let Some(live_tx) = thread_live_tx {
                let _ = live_tx.send(LiveAudioMessage::Finish);
            }
            writer
                .join()
                .map_err(|_| "Recording writer stopped unexpectedly".to_string())??;
            Ok(output)
        });

        *guard = Some(Recorder {
            stop_tx: Some(stop_tx),
            handle: Some(handle),
            live_tx,
            stt_context: stt_context_state,
            stt_context_revision: AtomicU64::new(0),
        });
        Ok(RecordingStarted {
            path: output_for_result.to_string_lossy().to_string(),
            device_name,
            sample_rate,
            live_started,
            stt_context,
        })
    }

    fn stop(&self) -> Result<RecordingStopped, String> {
        let mut guard = self.current.lock().map_err(|_| "Recording lock poisoned")?;
        let mut recorder = guard
            .take()
            .ok_or_else(|| "There is no active recording".to_string())?;
        let stt_context = recorder
            .stt_context
            .lock()
            .map_err(|_| "Recording context lock poisoned")?
            .clone();
        if let Some(tx) = recorder.stop_tx.take() {
            let _ = tx.send(());
        }
        let path = recorder
            .handle
            .take()
            .ok_or_else(|| "Recording worker is missing".to_string())?
            .join()
            .map_err(|_| "Recording worker stopped unexpectedly".to_string())??;
        Ok(RecordingStopped {
            path: path.to_string_lossy().to_string(),
            stt_context,
        })
    }

    fn update_stt_context(
        &self,
        stt_context: MeetingSttContext,
    ) -> Result<LiveContextUpdate, String> {
        let guard = self.current.lock().map_err(|_| "Recording lock poisoned")?;
        let recorder = guard
            .as_ref()
            .ok_or_else(|| "There is no active recording".to_string())?;
        let mut current = recorder
            .stt_context
            .lock()
            .map_err(|_| "Recording context lock poisoned")?;
        let changed = *current != stt_context;
        if changed {
            *current = stt_context.clone();
        }
        let revision = if changed {
            recorder
                .stt_context_revision
                .fetch_add(1, AtomicOrdering::Relaxed)
                + 1
        } else {
            recorder.stt_context_revision.load(AtomicOrdering::Relaxed)
        };
        let live_restart_pending = changed
            && recorder.live_tx.as_ref().is_some_and(|live_tx| {
                live_tx
                    .send(LiveAudioMessage::Reconfigure {
                        revision,
                        language_hints: stt_context.language_hints.clone(),
                        expected_speakers: stt_context.expected_speakers,
                    })
                    .is_ok()
            });
        let delivery_status = if !changed {
            "unchanged"
        } else if live_restart_pending {
            "pending"
        } else {
            "saved_for_final"
        };
        Ok(LiveContextUpdate {
            stt_context,
            changed,
            live_restart_pending,
            revision,
            delivery_status: delivery_status.into(),
        })
    }

    fn is_recording(&self) -> bool {
        self.current
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    fn current_stt_context(&self) -> Option<(MeetingSttContext, bool)> {
        let guard = self.current.lock().ok()?;
        let recorder = guard.as_ref()?;
        let context = recorder.stt_context.lock().ok()?.clone();
        Some((context, recorder.live_tx.is_some()))
    }
}

fn find_input_device(host: &cpal::Host, requested: Option<&str>) -> Result<Device, String> {
    if let Some(requested) = requested.filter(|value| !value.trim().is_empty()) {
        let devices = host
            .input_devices()
            .map_err(|error| format!("Could not list input devices: {error}"))?;
        for device in devices {
            if device.name().ok().as_deref() == Some(requested) {
                return Ok(device);
            }
        }
        return Err(format!("Input device is no longer available: {requested}"));
    }
    host.default_input_device()
        .ok_or_else(|| "No microphone or audio input device is available".to_string())
}

fn dispatch_samples(
    samples: Vec<i16>,
    writer_tx: &mpsc::Sender<Vec<i16>>,
    live_tx: Option<&tokio_mpsc::UnboundedSender<LiveAudioMessage>>,
    app_handle: &tauri::AppHandle,
    meter_counter: &std::sync::atomic::AtomicUsize,
) {
    if samples.is_empty() {
        return;
    }
    if let Some(live_tx) = live_tx {
        let bytes = bytemuck::cast_slice(&samples).to_vec();
        let _ = live_tx.send(LiveAudioMessage::Audio(bytes));
    }
    if meter_counter
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        .is_multiple_of(8)
    {
        let rms = (samples
            .iter()
            .map(|sample| {
                let value = *sample as f32 / i16::MAX as f32;
                value * value
            })
            .sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        let _ = app_handle.emit(
            "recording:level",
            RecordingLevel {
                level: (rms * 4.0).clamp(0.0, 1.0),
            },
        );
    }
    let _ = writer_tx.send(samples);
}

fn downmix_f32(data: &[f32], channels: usize) -> Vec<i16> {
    data.chunks(channels)
        .map(|frame| {
            let average = frame.iter().copied().sum::<f32>() / frame.len() as f32;
            (average.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
        })
        .collect()
}

fn downmix_i16(data: &[i16], channels: usize) -> Vec<i16> {
    data.chunks(channels)
        .map(|frame| {
            (frame.iter().map(|sample| *sample as i64).sum::<i64>() / frame.len() as i64) as i16
        })
        .collect()
}

fn downmix_u16(data: &[u16], channels: usize) -> Vec<i16> {
    data.chunks(channels)
        .map(|frame| {
            let average =
                frame.iter().map(|sample| *sample as i64).sum::<i64>() / frame.len() as i64;
            (average - 32_768) as i16
        })
        .collect()
}

fn emit_progress(
    handle: &tauri::AppHandle,
    stage: &str,
    detail: Option<String>,
    run_id: Option<&str>,
) {
    if let Some(id) = run_id {
        eprintln!("[progress {id}] {stage} {detail:?}");
    } else {
        eprintln!("[progress] {stage} {detail:?}");
    }
    let payload = ProgressEvent {
        event_id: Uuid::new_v4().to_string(),
        stage: stage.to_string(),
        detail,
        run_id: run_id.map(str::to_string),
    };
    if let Some(id) = run_id {
        if let Ok(mut progress) = handle.state::<AppState>().progress.lock() {
            progress
                .entry(id.to_string())
                .or_default()
                .push(payload.clone());
        }
    }
    let _ = handle.emit("transcription:progress", payload);
}

#[tauri::command]
fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|device| device.name().ok());
    let devices = host
        .input_devices()
        .map_err(|error| format!("Could not list input devices: {error}"))?;
    let mut output = Vec::new();
    for device in devices {
        if let Ok(name) = device.name() {
            output.push(AudioDeviceInfo {
                is_default: default_name.as_deref() == Some(name.as_str()),
                name,
            });
        }
    }
    output.sort_by(|left, right| {
        right
            .is_default
            .cmp(&left.is_default)
            .then(left.name.cmp(&right.name))
    });
    Ok(output)
}

fn start_recording_impl(
    manager: &RecordingManager,
    app_state: &AppState,
    app_handle: tauri::AppHandle,
    input_device: Option<String>,
) -> Result<RecordingStarted, String> {
    let maintenance = app_state
        .maintenance_in_flight
        .lock()
        .map_err(|_| "Maintenance lock poisoned".to_string())?;
    if *maintenance {
        return Err(
            "Voice recognition maintenance is running. Start recording when it finishes.".into(),
        );
    }
    let api_key = app_state.load_soniox_key()?;
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?
        .clone();
    let requested = input_device.or(config.selected_input_device.clone());
    let stt_context = MeetingSttContext {
        language_hints: config.language_hints.clone(),
        expected_speakers: None,
    }
    .normalized()?;
    app_state.reset_live_transcript(config.live_transcription)?;
    let live = config.live_transcription.then_some(LiveRecordingConfig {
        api_key,
        options: soniox::RealtimeOptions {
            language_hints: stt_context.language_hints.clone(),
            expected_speakers: stt_context.expected_speakers,
            preferred_language: config.preferred_language.clone(),
            no_translation_languages: config.no_translation_languages.clone(),
        },
    });
    let started = manager.start(requested.as_deref(), stt_context, live, app_handle.clone())?;
    drop(maintenance);
    let _ = app_handle.emit("recording:started", started.clone());
    Ok(started)
}

#[tauri::command]
fn start_recording(
    input_device: Option<String>,
    manager: State<RecordingManager>,
    app_state: State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<RecordingStarted, String> {
    start_recording_impl(&manager, &app_state, app_handle, input_device)
}

#[tauri::command]
fn stop_recording(
    manager: State<RecordingManager>,
    app_handle: tauri::AppHandle,
) -> Result<RecordingStopped, String> {
    let stopped = manager.stop()?;
    let _ = app_handle.emit("recording:stopped", stopped.clone());
    Ok(stopped)
}

#[tauri::command]
fn update_live_context(
    stt_context: MeetingSttContext,
    manager: State<RecordingManager>,
) -> Result<LiveContextUpdate, String> {
    manager.update_stt_context(stt_context.normalized()?)
}

fn live_transcript_fallback(app_state: &AppState) -> String {
    app_state
        .live_transcript
        .lock()
        .map(|snapshot| {
            if snapshot.text.trim().len() >= snapshot.final_text.trim().len() {
                snapshot.text.trim().to_string()
            } else {
                snapshot.final_text.trim().to_string()
            }
        })
        .unwrap_or_default()
}

fn wav_duration_ms(path: &Path) -> Result<i64, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|error| format!("Could not inspect the recorded WAV: {error}"))?;
    let sample_rate = reader.spec().sample_rate;
    if sample_rate == 0 {
        return Err("The recorded WAV has an invalid sample rate".into());
    }
    Ok(((reader.duration() as u64 * 1_000) / sample_rate as u64) as i64)
}

#[cfg(unix)]
fn set_private_permissions(path: &Path, mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|error| {
        format!(
            "Could not restrict permissions for {}: {error}",
            path.display()
        )
    })
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path, _mode: u32) -> Result<(), String> {
    Ok(())
}

fn persist_recording_audio(
    source: &Path,
    app_state: &AppState,
    session_id: &str,
) -> Result<PathBuf, String> {
    let source_size = source
        .metadata()
        .map_err(|error| format!("Could not inspect the completed recording: {error}"))?
        .len();
    if source_size == 0 {
        return Err("The completed recording is empty".into());
    }
    let directory = app_state.data_dir.join("processing");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("Could not create the recovery-audio directory: {error}"))?;
    set_private_permissions(&directory, 0o700)?;
    let target = directory.join(format!("{session_id}.wav"));
    if target.exists() {
        return Err("A recovery recording with this identifier already exists".into());
    }

    if let Err(rename_error) = fs::rename(source, &target) {
        let copy_result = (|| -> Result<(), String> {
            let mut input = File::open(source)
                .map_err(|error| format!("Could not reopen the completed recording: {error}"))?;
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&target)
                .map_err(|error| format!("Could not create the recovery recording: {error}"))?;
            io::copy(&mut input, &mut output).map_err(|error| {
                format!("Could not copy the recording into safe storage: {error}")
            })?;
            output
                .sync_all()
                .map_err(|error| format!("Could not flush the recovery recording: {error}"))?;
            Ok(())
        })();
        if let Err(error) = copy_result {
            let _ = fs::remove_file(&target);
            return Err(format!(
                "Could not move the recording into safe storage ({rename_error}); {error}"
            ));
        }
        if let Err(error) = fs::remove_file(source) {
            eprintln!(
                "[recording] durable copy saved at {}; original temporary copy could not be removed: {error}",
                target.display()
            );
        }
    }

    set_private_permissions(&target, 0o600)?;
    let target_size = target
        .metadata()
        .map_err(|error| format!("Could not verify the recovery recording: {error}"))?
        .len();
    if target_size != source_size {
        return Err(format!(
            "The recovery recording failed verification (expected {source_size} bytes, found {target_size})"
        ));
    }
    File::open(&target)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("Could not flush the recovery recording: {error}"))?;
    if let Ok(directory_handle) = File::open(&directory) {
        let _ = directory_handle.sync_all();
    }
    Ok(target)
}

fn validate_managed_audio_path(path: &Path, app_state: &AppState) -> Result<(), String> {
    let directory = app_state.data_dir.join("processing");
    let canonical_directory = directory
        .canonicalize()
        .map_err(|error| format!("Could not open the recovery-audio directory: {error}"))?;
    let canonical_path = path
        .canonicalize()
        .map_err(|error| format!("The retained recording is unavailable: {error}"))?;
    if !canonical_path.starts_with(&canonical_directory) {
        return Err("Recall refused to access a recording outside its recovery directory".into());
    }
    Ok(())
}

fn remove_managed_audio(path: &Path, app_state: &AppState) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    validate_managed_audio_path(path, app_state)?;
    fs::remove_file(path)
        .map_err(|error| format!("Could not delete the retained recording: {error}"))
}

fn transcribe_file_inner(
    path: &str,
    session_id: &str,
    draft_transcript: &str,
    stt_context: &MeetingSttContext,
    app_state: &AppState,
    app_handle: &tauri::AppHandle,
    run_id: &str,
) -> Result<String, String> {
    emit_progress(
        app_handle,
        "transcription:start",
        Some("Preparing final transcription".into()),
        Some(run_id),
    );
    let api_key = app_state.load_soniox_key()?;
    let result = soniox::transcribe_file(
        Path::new(path),
        &api_key,
        &stt_context.language_hints,
        stt_context.expected_speakers,
        |stage, detail| emit_progress(app_handle, stage, Some(detail), Some(run_id)),
    )?;
    emit_progress(
        app_handle,
        "audio:read:start",
        Some("Reading local audio for speaker fingerprints".into()),
        Some(run_id),
    );
    let audio = read_audio_clip(path)?;
    emit_progress(
        app_handle,
        "audio:read:done",
        Some(format!(
            "Loaded {:.1} seconds of mono audio",
            audio.duration_ms() as f64 / 1_000.0
        )),
        Some(run_id),
    );
    let segments = merge_segments(&normalize_segments(
        result.segments,
        &result.transcript,
        &audio,
    ));
    let db = app_state.db_handle()?;

    emit_progress(
        app_handle,
        "vad:start",
        Some("Detecting speech locally before speaker fingerprinting".into()),
        Some(run_id),
    );
    let speech_intervals = {
        let loaded = app_state
            .vad
            .lock()
            .map_err(|_| "VAD model lock poisoned")?
            .is_some();
        if !loaded {
            if let Err(error) = app_state.load_vad() {
                emit_progress(
                    app_handle,
                    "vad:warning",
                    Some(format!(
                        "Speech detection unavailable: {error}. Speaker labels will remain meeting-local."
                    )),
                    Some(run_id),
                );
            }
        }
        let detector = app_state
            .vad
            .lock()
            .map_err(|_| "VAD model lock poisoned")?;
        match detector.as_ref() {
            Some(detector) => match detector.speech_intervals(&audio.samples, audio.sample_rate) {
                Ok(intervals) => {
                    emit_progress(
                        app_handle,
                        "vad:done",
                        Some(format!(
                            "Detected {:.1} seconds of speech in {} interval{}",
                            vad::total_duration(&intervals) as f64 / 1_000.0,
                            intervals.len(),
                            if intervals.len() == 1 { "" } else { "s" }
                        )),
                        Some(run_id),
                    );
                    Some(intervals)
                }
                Err(error) => {
                    emit_progress(
                        app_handle,
                        "vad:warning",
                        Some(format!(
                            "Speech detection failed: {error}. Speaker labels will remain meeting-local."
                        )),
                        Some(run_id),
                    );
                    None
                }
            },
            None => None,
        }
    };

    emit_progress(
        app_handle,
        "voiceprints:start",
        Some(format!(
            "Extracting and matching local voiceprints with {EMBEDDING_VERSION}"
        )),
        Some(run_id),
    );
    let embedder_available = {
        let loaded = app_state
            .embedder
            .lock()
            .map_err(|_| "Speaker model lock poisoned")?
            .is_some();
        if loaded {
            true
        } else {
            match app_state.load_embedder() {
                Ok(()) => true,
                Err(error) => {
                    emit_progress(
                        app_handle,
                        "voiceprints:warning",
                        Some(format!("Automatic matching unavailable: {error}")),
                        Some(run_id),
                    );
                    false
                }
            }
        }
    };
    if embedder_available {
        let embedder = app_state
            .embedder
            .lock()
            .map_err(|_| "Speaker model lock poisoned")?;
        process_segments(
            &audio,
            &segments,
            session_id,
            &db,
            embedder.as_ref(),
            speech_intervals.as_deref(),
            (app_handle, run_id),
        )?;
    } else {
        process_segments(
            &audio,
            &segments,
            session_id,
            &db,
            None,
            speech_intervals.as_deref(),
            (app_handle, run_id),
        )?;
    }
    emit_progress(
        app_handle,
        "voiceprints:done",
        Some("Speaker attribution finished".into()),
        Some(run_id),
    );
    let saved_segments = db.list_segments(session_id)?;
    let provider_fallback = if result.transcript.trim().is_empty() {
        draft_transcript
    } else {
        &result.transcript
    };
    let final_display = build_saved_transcript(&saved_segments, provider_fallback);
    let title = make_conversation_title(provider_fallback);
    db.finalize_processing_session(
        session_id,
        &title,
        &final_display,
        audio.duration_ms() as i64,
    )?;
    emit_progress(
        app_handle,
        "transcription:done",
        Some("Conversation saved locally".into()),
        Some(run_id),
    );
    Ok(session_id.to_string())
}

fn spawn_transcription_worker(
    path: String,
    session_id: String,
    draft_transcript: String,
    stt_context: MeetingSttContext,
    state: AppState,
    app_handle: tauri::AppHandle,
    run_id: String,
) {
    tauri::async_runtime::spawn_blocking(move || {
        let result = transcribe_file_inner(
            &path,
            &session_id,
            &draft_transcript,
            &stt_context,
            &state,
            &app_handle,
            &run_id,
        );
        match result {
            Ok(session_id) => {
                let audio_path = Path::new(&path);
                match remove_managed_audio(audio_path, &state) {
                    Ok(()) => {
                        if let Ok(db) = state.db_handle() {
                            if let Err(error) = db.complete_processing_session(&session_id) {
                                emit_progress(
                                    &app_handle,
                                    "audio:cleanup:warning",
                                    Some(format!(
                                        "The recording was removed, but cleanup bookkeeping failed: {error}"
                                    )),
                                    Some(&run_id),
                                );
                            }
                        }
                        emit_progress(
                            &app_handle,
                            "audio:cleanup:done",
                            Some("Retained recording deleted after successful processing".into()),
                            Some(&run_id),
                        );
                    }
                    Err(error) => {
                        if let Ok(db) = state.db_handle() {
                            let _ = db.mark_processing_cleanup_failed(&session_id, &error);
                        }
                        emit_progress(
                            &app_handle,
                            "audio:cleanup:warning",
                            Some(format!("{error}. The final transcript is safe.")),
                            Some(&run_id),
                        );
                    }
                }
                emit_progress(&app_handle, "complete", Some(session_id), Some(&run_id));
            }
            Err(error) => {
                let persisted_error = match state.db_handle() {
                    Ok(db) => match db.fail_processing_session(&session_id, &error) {
                        Ok(()) => error,
                        Err(save_error) => format!(
                            "{error}. Recall also could not persist the failure state: {save_error}"
                        ),
                    },
                    Err(save_error) => format!(
                        "{error}. Recall also could not reopen the local database: {save_error}"
                    ),
                };
                emit_progress(
                    &app_handle,
                    "audio:retained",
                    Some("The recording and live-caption draft were kept for retry".into()),
                    Some(&run_id),
                );
                emit_progress(&app_handle, "error", Some(persisted_error), Some(&run_id));
            }
        }
    });
}

fn queue_transcription(
    path: String,
    stt_context: MeetingSttContext,
    state: AppState,
    app_handle: tauri::AppHandle,
) -> Result<QueuedTranscription, String> {
    let maintenance = state
        .maintenance_in_flight
        .lock()
        .map_err(|_| "Maintenance lock poisoned".to_string())?;
    if *maintenance {
        return Err(
            "Voice recognition maintenance is running. Final processing can start when it finishes."
                .into(),
        );
    }
    let stt_context = stt_context.normalized()?;
    let source = Path::new(&path);
    if !source.is_file() {
        return Err("Recording file does not exist".into());
    }
    let db = state.db_handle()?;
    let run_id = Uuid::new_v4().to_string();
    let session_id = Uuid::new_v4().to_string();
    if let Ok(mut progress) = state.progress.lock() {
        progress.entry(run_id.clone()).or_default();
    }
    emit_progress(
        &app_handle,
        "audio:persist:start",
        Some("Saving a recovery copy before final transcription".into()),
        Some(&run_id),
    );
    let draft_transcript = live_transcript_fallback(&state);
    let title = make_conversation_title(&draft_transcript);
    let expected_path = state
        .data_dir
        .join("processing")
        .join(format!("{session_id}.wav"));
    let retained_path = match persist_recording_audio(source, &state, &session_id) {
        Ok(path) => path,
        Err(error) if expected_path.is_file() => {
            let durable_error = format!(
                "The recording reached recovery storage but local verification failed: {error}"
            );
            if let Err(database_error) = db.create_processing_session_with_context(
                &session_id,
                &run_id,
                &title,
                &draft_transcript,
                0,
                &expected_path.to_string_lossy(),
                &stt_context.language_hints,
                stt_context.expected_speakers,
            ) {
                let combined = format!(
                    "{durable_error}. The recording remains at {}, but its conversation row could not be created: {database_error}",
                    expected_path.display()
                );
                emit_progress(&app_handle, "error", Some(combined.clone()), Some(&run_id));
                return Err(combined);
            }
            if let Err(database_error) = db.fail_processing_session(&session_id, &durable_error) {
                let combined = format!(
                    "{durable_error}. Recall could not mark the draft as failed: {database_error}"
                );
                emit_progress(&app_handle, "error", Some(combined.clone()), Some(&run_id));
                return Err(combined);
            }
            emit_progress(
                &app_handle,
                "audio:retained",
                Some("The recording and live-caption draft were kept for inspection".into()),
                Some(&run_id),
            );
            emit_progress(&app_handle, "error", Some(durable_error), Some(&run_id));
            return Ok(QueuedTranscription { run_id, session_id });
        }
        Err(error) => {
            emit_progress(&app_handle, "error", Some(error.clone()), Some(&run_id));
            return Err(error);
        }
    };
    let duration_result = wav_duration_ms(&retained_path);
    let duration_ms = duration_result.as_ref().copied().unwrap_or(0);
    if let Err(error) = db.create_processing_session_with_context(
        &session_id,
        &run_id,
        &title,
        &draft_transcript,
        duration_ms,
        &retained_path.to_string_lossy(),
        &stt_context.language_hints,
        stt_context.expected_speakers,
    ) {
        let message = format!(
            "The recording is safe at {}, but Recall could not create its conversation record: {error}",
            retained_path.display()
        );
        emit_progress(&app_handle, "error", Some(message.clone()), Some(&run_id));
        return Err(message);
    }
    emit_progress(
        &app_handle,
        "audio:persisted",
        Some("Recording and live-caption draft saved locally".into()),
        Some(&run_id),
    );
    if let Err(error) = duration_result {
        if let Err(database_error) = db.fail_processing_session(&session_id, &error) {
            let combined = format!(
                "{error}. Recall could not mark the saved draft as failed: {database_error}"
            );
            emit_progress(&app_handle, "error", Some(combined.clone()), Some(&run_id));
            return Err(combined);
        }
        emit_progress(
            &app_handle,
            "audio:retained",
            Some("The recording and live-caption draft were kept for inspection".into()),
            Some(&run_id),
        );
        emit_progress(&app_handle, "error", Some(error), Some(&run_id));
        return Ok(QueuedTranscription { run_id, session_id });
    }
    emit_progress(
        &app_handle,
        "queued",
        Some("Final transcription queued".into()),
        Some(&run_id),
    );
    drop(maintenance);
    spawn_transcription_worker(
        retained_path.to_string_lossy().to_string(),
        session_id.clone(),
        draft_transcript,
        stt_context,
        state,
        app_handle,
        run_id.clone(),
    );
    Ok(QueuedTranscription { run_id, session_id })
}

#[tauri::command]
fn transcribe_file_async(
    path: String,
    stt_context: MeetingSttContext,
    app_state: State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<QueuedTranscription, String> {
    queue_transcription(path, stt_context, app_state.inner().clone(), app_handle)
}

#[tauri::command]
fn retry_processing(
    session_id: String,
    app_state: State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<QueuedTranscription, String> {
    let maintenance = app_state
        .maintenance_in_flight
        .lock()
        .map_err(|_| "Maintenance lock poisoned".to_string())?;
    if *maintenance {
        return Err("Voice recognition maintenance is running. Retry when it finishes.".into());
    }
    let db = app_state.db_handle()?;
    let job = db
        .processing_job(&session_id)?
        .ok_or_else(|| "This conversation has no retained recording to retry".to_string())?;
    if job.status != "failed" {
        return Err("This conversation is not waiting for a transcription retry".into());
    }
    let path = PathBuf::from(&job.audio_path);
    validate_managed_audio_path(&path, app_state.inner())?;
    let session = db
        .get_session(&session_id)?
        .ok_or_else(|| "Conversation not found".to_string())?;
    let run_id = Uuid::new_v4().to_string();
    db.restart_processing_session(&session_id, &run_id)?;
    if let Ok(mut progress) = app_state.progress.lock() {
        progress.entry(run_id.clone()).or_default();
    }
    emit_progress(
        &app_handle,
        "retry:queued",
        Some("Retrying final transcription from the retained recording".into()),
        Some(&run_id),
    );
    drop(maintenance);
    spawn_transcription_worker(
        job.audio_path,
        session_id.clone(),
        session.transcript,
        MeetingSttContext {
            language_hints: job.language_hints,
            expected_speakers: job.expected_speakers,
        },
        app_state.inner().clone(),
        app_handle,
        run_id.clone(),
    );
    Ok(QueuedTranscription { run_id, session_id })
}

#[tauri::command]
fn discard_retained_audio(session_id: String, app_state: State<AppState>) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let job = db
        .processing_job(&session_id)?
        .ok_or_else(|| "This conversation has no retained recording".to_string())?;
    if !matches!(job.status.as_str(), "finalized" | "cleanup_failed") {
        return Err("The retained recording is still needed to recover this conversation".into());
    }
    remove_managed_audio(Path::new(&job.audio_path), app_state.inner())?;
    db.complete_processing_session(&session_id)
}

#[tauri::command]
fn get_progress(run_id: String, app_state: State<AppState>) -> Result<Vec<ProgressEvent>, String> {
    Ok(app_state
        .progress
        .lock()
        .map_err(|_| "Progress lock poisoned")?
        .get(&run_id)
        .cloned()
        .unwrap_or_default())
}

#[tauri::command]
fn get_live_transcription(app_state: State<AppState>) -> Result<LiveTranscriptEvent, String> {
    app_state
        .live_transcript
        .lock()
        .map_err(|_| "Live transcription lock poisoned".to_string())
        .map(|snapshot| snapshot.clone())
}

fn read_audio_clip(path: &str) -> Result<AudioClip, String> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| format!("Could not open recorded WAV: {error}"))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let mut interleaved = Vec::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            for sample in reader.samples::<i16>() {
                interleaved.push(
                    sample.map_err(|error| format!("Could not decode WAV: {error}"))? as f32
                        / i16::MAX as f32,
                );
            }
        }
        (hound::SampleFormat::Int, 24 | 32) => {
            for sample in reader.samples::<i32>() {
                interleaved.push(
                    sample.map_err(|error| format!("Could not decode WAV: {error}"))? as f32
                        / i32::MAX as f32,
                );
            }
        }
        (hound::SampleFormat::Float, _) => {
            for sample in reader.samples::<f32>() {
                interleaved.push(sample.map_err(|error| format!("Could not decode WAV: {error}"))?);
            }
        }
        _ => return Err("The recorded WAV format is unsupported".into()),
    }
    if interleaved.is_empty() {
        return Err("The recording contains no audio samples".into());
    }
    let samples = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
        .collect();
    Ok(AudioClip {
        samples,
        sample_rate: spec.sample_rate,
    })
}

fn normalize_segments(
    mut segments: Vec<TranscriptSegment>,
    transcript: &str,
    audio: &AudioClip,
) -> Vec<TranscriptSegment> {
    if segments.is_empty() && !transcript.trim().is_empty() {
        segments.push(TranscriptSegment {
            speaker: "unknown".into(),
            start_ms: 0,
            end_ms: audio.duration_ms(),
            text: transcript.trim().into(),
        });
    }
    let duration = audio.duration_ms();
    for segment in &mut segments {
        if segment.end_ms < segment.start_ms {
            segment.end_ms = segment.start_ms;
        }
        segment.end_ms = segment.end_ms.min(duration);
    }
    segments.sort_by_key(|segment| segment.start_ms);
    segments
}

fn merge_segments(segments: &[TranscriptSegment]) -> Vec<TranscriptSegment> {
    let mut merged: Vec<TranscriptSegment> = Vec::new();
    for segment in segments {
        if let Some(previous) = merged.last_mut() {
            if previous.speaker == segment.speaker && segment.start_ms <= previous.end_ms + 1_000 {
                previous.end_ms = previous.end_ms.max(segment.end_ms);
                if !previous
                    .text
                    .chars()
                    .last()
                    .map(char::is_whitespace)
                    .unwrap_or(false)
                {
                    previous.text.push(' ');
                }
                previous.text.push_str(segment.text.trim());
                continue;
            }
        }
        merged.push(segment.clone());
    }
    merged
}

fn build_saved_transcript(segments: &[SegmentRecord], fallback: &str) -> String {
    let lines = segments
        .iter()
        .filter(|segment| !segment.text.trim().is_empty())
        .map(|segment| {
            format!(
                "{}: {}",
                segment
                    .speaker_label
                    .as_deref()
                    .unwrap_or("Unknown speaker"),
                segment.text.trim()
            )
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        fallback.trim().to_string()
    } else {
        lines.join("\n")
    }
}

fn refresh_session_transcript(db: &Db, session_id: &str) -> Result<(), String> {
    let segments = db.list_segments(session_id)?;
    let transcript = build_saved_transcript(&segments, "");
    db.update_session_transcript(session_id, &transcript)
}

fn ensure_sessions_not_recapping(
    app_state: &AppState,
    session_ids: &[String],
) -> Result<(), String> {
    ensure_maintenance_not_running(app_state)?;
    let in_flight = app_state
        .recap_in_flight
        .lock()
        .map_err(|_| "Recap lock poisoned".to_string())?;
    if session_ids
        .iter()
        .any(|session_id| in_flight.contains(session_id))
    {
        return Err(
            "That change is paused because an affected conversation is being recapped. Try again after the recap finishes."
                .into(),
        );
    }
    Ok(())
}

fn ensure_maintenance_not_running(app_state: &AppState) -> Result<(), String> {
    if *app_state
        .maintenance_in_flight
        .lock()
        .map_err(|_| "Maintenance lock poisoned".to_string())?
    {
        return Err("Voice recognition maintenance is running. Try again when it finishes.".into());
    }
    Ok(())
}

fn ensure_session_not_recapping(app_state: &AppState, session_id: &str) -> Result<(), String> {
    ensure_sessions_not_recapping(app_state, &[session_id.to_string()])
}

fn claim_identity_sessions(app_state: &AppState, session_ids: &[String]) -> Result<(), String> {
    let maintenance = app_state
        .maintenance_in_flight
        .lock()
        .map_err(|_| "Maintenance lock poisoned".to_string())?;
    if *maintenance {
        return Err("Voice recognition maintenance is running. Try again when it finishes.".into());
    }
    let recap_in_flight = app_state
        .recap_in_flight
        .lock()
        .map_err(|_| "Recap lock poisoned".to_string())?;
    if session_ids
        .iter()
        .any(|session_id| recap_in_flight.contains(session_id))
    {
        return Err(
            "That change is paused because an affected conversation is being recapped. Try again after the recap finishes."
                .into(),
        );
    }
    let mut identity_in_flight = app_state
        .identity_in_flight
        .lock()
        .map_err(|_| "Identity lock poisoned".to_string())?;
    if session_ids
        .iter()
        .any(|session_id| identity_in_flight.contains(session_id))
    {
        return Err(
            "Those people or voices are already being changed. Wait for that operation to finish."
                .into(),
        );
    }
    identity_in_flight.extend(session_ids.iter().cloned());
    drop(maintenance);
    Ok(())
}

fn release_identity_sessions(app_state: &AppState, session_ids: &[String]) {
    if let Ok(mut identity_in_flight) = app_state.identity_in_flight.lock() {
        for session_id in session_ids {
            identity_in_flight.remove(session_id);
        }
    }
}

fn make_conversation_title(transcript: &str) -> String {
    let words = transcript.split_whitespace().take(9).collect::<Vec<_>>();
    if words.is_empty() {
        chrono::Local::now()
            .format("Conversation %d %b %Y, %H:%M")
            .to_string()
    } else {
        let title = words.join(" ");
        if transcript.split_whitespace().count() > words.len() {
            format!("{title}…")
        } else {
            title
        }
    }
}

#[derive(Debug, Default)]
struct SampleWindowSet {
    windows: Vec<SampleWindow>,
    overlapping_segments: usize,
    short_segments: usize,
    no_speech_segments: usize,
    short_speech_intervals: usize,
}

fn overlaps_other_speaker(segment: &TranscriptSegment, segments: &[TranscriptSegment]) -> bool {
    segments.iter().any(|other| {
        if other.speaker == segment.speaker {
            return false;
        }
        let overlap_start = segment.start_ms.max(other.start_ms);
        let overlap_end = segment.end_ms.min(other.end_ms);
        overlap_end.saturating_sub(overlap_start) > SAMPLE_OVERLAP_TOLERANCE_MS
    })
}

fn sample_range(audio: &AudioClip, start_ms: u64, end_ms: u64) -> Option<Vec<f32>> {
    if audio.sample_rate == 0 || end_ms <= start_ms {
        return None;
    }
    let start = ((start_ms as u128 * audio.sample_rate as u128) / 1_000) as usize;
    let end = ((end_ms as u128 * audio.sample_rate as u128) / 1_000) as usize;
    let start = start.min(audio.samples.len());
    let end = end.min(audio.samples.len());
    (end > start).then(|| audio.samples[start..end].to_vec())
}

fn bounded_centered_sample_ranges(
    speech_intervals: &[vad::SpeechInterval],
    segment_midpoint: u64,
) -> (Vec<(u64, u64)>, usize) {
    let mut ranges = Vec::new();
    let mut short_speech_intervals = 0usize;
    let range_cap = MAX_SAMPLE_WINDOWS_PER_SPEAKER as u64;

    for interval in speech_intervals {
        let speech_duration = interval.duration_ms();
        if speech_duration < MIN_SPEAKER_MS {
            short_speech_intervals += 1;
            continue;
        }
        let full_windows = speech_duration / SAMPLE_WINDOW_MS;
        if full_windows == 0 {
            ranges.push((interval.start_ms, interval.end_ms));
        } else {
            let used = full_windows * SAMPLE_WINDOW_MS;
            let offset = (speech_duration - used) / 2;
            let first_start = interval.start_ms + offset;
            let first_center = first_start + (SAMPLE_WINDOW_MS / 2);
            let nearest_index = segment_midpoint
                .saturating_sub(first_center)
                .saturating_add(SAMPLE_WINDOW_MS / 2)
                / SAMPLE_WINDOW_MS;
            let nearest_index = nearest_index.min(full_windows.saturating_sub(1));
            let selected_window_count = full_windows.min(range_cap);
            let first_index = nearest_index
                .saturating_sub(selected_window_count / 2)
                .min(full_windows - selected_window_count);
            ranges.extend(
                (first_index..first_index + selected_window_count).map(|index| {
                    let start = first_start + (index * SAMPLE_WINDOW_MS);
                    (start, start + SAMPLE_WINDOW_MS)
                }),
            );
        }

        // Keep only the globally best bounded set while walking VAD intervals,
        // rather than materializing every possible window in a long recording.
        ranges.sort_by_key(|(start, end)| {
            start
                .saturating_add(end.saturating_sub(*start) / 2)
                .abs_diff(segment_midpoint)
        });
        ranges.truncate(MAX_SAMPLE_WINDOWS_PER_SPEAKER);
    }

    (ranges, short_speech_intervals)
}

fn clean_sample_windows(
    audio: &AudioClip,
    segments: &[TranscriptSegment],
    diarized_speaker: &str,
    speech_intervals: &[vad::SpeechInterval],
) -> SampleWindowSet {
    let mut speaker_segments = segments
        .iter()
        .enumerate()
        .filter(|(_, segment)| {
            segment.speaker == diarized_speaker && segment.end_ms > segment.start_ms
        })
        .collect::<Vec<_>>();
    speaker_segments.sort_by(|(_, left), (_, right)| {
        (right.end_ms - right.start_ms).cmp(&(left.end_ms - left.start_ms))
    });
    let mut result = SampleWindowSet::default();
    let mut candidate_ranges_by_intervention = Vec::new();

    for (segment_index, segment) in speaker_segments {
        if overlaps_other_speaker(segment, segments) {
            result.overlapping_segments += 1;
            continue;
        }
        let duration_ms = segment.end_ms - segment.start_ms;
        if duration_ms < MIN_SPEAKER_MS + (SAMPLE_EDGE_TRIM_MS * 2) {
            result.short_segments += 1;
            continue;
        }
        let safe_start = segment.start_ms + SAMPLE_EDGE_TRIM_MS;
        let safe_end = segment.end_ms - SAMPLE_EDGE_TRIM_MS;
        let safe_duration = safe_end.saturating_sub(safe_start);
        if safe_duration < MIN_SPEAKER_MS {
            result.short_segments += 1;
            continue;
        }
        let speech = vad::intersections(speech_intervals, safe_start, safe_end);
        if speech.is_empty() {
            result.no_speech_segments += 1;
            continue;
        }
        let segment_midpoint = segment.start_ms.saturating_add(duration_ms / 2);
        let (ranges, short_speech_intervals) =
            bounded_centered_sample_ranges(&speech, segment_midpoint);
        result.short_speech_intervals += short_speech_intervals;
        if !ranges.is_empty() {
            candidate_ranges_by_intervention.push(
                ranges
                    .into_iter()
                    .map(|(start_ms, end_ms)| (start_ms, end_ms, segment_index))
                    .collect::<Vec<_>>(),
            );
            // One window from each of this many eligible interventions fills
            // the complete bounded candidate set, so later interventions
            // cannot be selected by the round-robin scheduler.
            if candidate_ranges_by_intervention.len() >= MAX_SAMPLE_WINDOWS_PER_SPEAKER {
                break;
            }
        }
    }

    // Walk the eligible interventions round-robin. This makes the first
    // candidate round representative of distinct interventions instead of
    // allowing the longest intervention to consume the complete window cap.
    let mut next_range_indices = vec![0usize; candidate_ranges_by_intervention.len()];
    let mut intervention_cursor = 0usize;
    let mut consecutive_exhausted = 0usize;
    while result.windows.len() < MAX_SAMPLE_WINDOWS_PER_SPEAKER
        && !candidate_ranges_by_intervention.is_empty()
        && consecutive_exhausted < candidate_ranges_by_intervention.len()
    {
        let intervention_index = intervention_cursor % candidate_ranges_by_intervention.len();
        intervention_cursor += 1;
        let range_index = next_range_indices[intervention_index];
        let Some(&(start_ms, end_ms, segment_index)) =
            candidate_ranges_by_intervention[intervention_index].get(range_index)
        else {
            consecutive_exhausted += 1;
            continue;
        };
        next_range_indices[intervention_index] += 1;
        consecutive_exhausted = 0;
        let Some(pcm) = sample_range(audio, start_ms, end_ms) else {
            continue;
        };
        result.windows.push(SampleWindow {
            start_ms,
            end_ms,
            segment_index,
            candidate_batch: result.windows.len() / SAMPLE_WINDOWS_PER_CANDIDATE_BATCH,
            pcm,
        });
    }
    result
}

fn dominant_consistent_indices(vectors: &[Vec<f32>]) -> Vec<usize> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let mut best_members = Vec::new();
    let mut best_similarity = f32::NEG_INFINITY;
    for vector in vectors {
        let members = vectors
            .iter()
            .enumerate()
            .filter_map(|(candidate_index, candidate)| {
                (embedding::cosine_similarity(vector, candidate) >= SAMPLE_CONSISTENCY_THRESHOLD)
                    .then_some(candidate_index)
            })
            .collect::<Vec<_>>();
        let similarity = members
            .iter()
            .map(|member| embedding::cosine_similarity(vector, &vectors[*member]))
            .sum::<f32>();
        if members.len() > best_members.len()
            || (members.len() == best_members.len() && similarity > best_similarity)
        {
            best_members = members;
            best_similarity = similarity;
        }
    }
    if best_members.len() * 2 > vectors.len() {
        best_members
    } else {
        Vec::new()
    }
}

fn first_trusted_sample_batch(
    embedded_windows: &[(SampleWindow, Vec<f32>)],
) -> Option<TrustedSampleBatch> {
    for batch_index in 0..MAX_SAMPLE_CANDIDATE_BATCHES {
        let candidate_indices = embedded_windows
            .iter()
            .enumerate()
            .filter_map(|(index, (window, _))| {
                (window.candidate_batch == batch_index).then_some(index)
            })
            .collect::<Vec<_>>();
        if candidate_indices.is_empty() {
            continue;
        }
        let vectors = candidate_indices
            .iter()
            .map(|index| embedded_windows[*index].1.clone())
            .collect::<Vec<_>>();
        let consistent_indices = dominant_consistent_indices(&vectors);
        if consistent_indices.is_empty() {
            continue;
        }
        return Some(TrustedSampleBatch {
            batch_index,
            window_indices: consistent_indices
                .into_iter()
                .map(|index| candidate_indices[index])
                .collect(),
            candidate_count: candidate_indices.len(),
        });
    }
    None
}

fn average_embeddings(vectors: impl Iterator<Item = Vec<f32>>) -> Vec<f32> {
    let vectors = vectors.collect::<Vec<_>>();
    let Some(first) = vectors.first() else {
        return Vec::new();
    };
    let mut average = vec![0.0; first.len()];
    let mut count = 0usize;
    for vector in vectors {
        if vector.len() != average.len() {
            continue;
        }
        for (target, value) in average.iter_mut().zip(vector) {
            *target += value;
        }
        count += 1;
    }
    if count == 0 {
        return Vec::new();
    }
    for value in &mut average {
        *value /= count as f32;
    }
    let norm = average
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut average {
            *value /= norm;
        }
    }
    average
}

fn mean_pairwise_similarity(vectors: &[Vec<f32>]) -> f32 {
    if vectors.len() < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    let mut pairs = 0usize;
    for left in 0..vectors.len() {
        for right in (left + 1)..vectors.len() {
            total += embedding::cosine_similarity(&vectors[left], &vectors[right]);
            pairs += 1;
        }
    }
    if pairs == 0 {
        1.0
    } else {
        total / pairs as f32
    }
}

fn group_voice_observations(observations: &[VoiceObservation]) -> Vec<VoiceObservationGroup> {
    let mut groups: Vec<VoiceObservationGroup> = Vec::new();
    for (index, observation) in observations.iter().enumerate() {
        let compatible_group = groups.iter().position(|group| {
            group.observation_indices.iter().all(|member| {
                let existing = &observations[*member];
                observation.clean_window_count >= MIN_COALESCE_WINDOWS_PER_LABEL
                    && existing.clean_window_count >= MIN_COALESCE_WINDOWS_PER_LABEL
                    && observation.selected_duration_ms >= MIN_COALESCE_DURATION_MS_PER_LABEL
                    && existing.selected_duration_ms >= MIN_COALESCE_DURATION_MS_PER_LABEL
                    && observation.consistency_score >= MIN_COALESCE_CONSISTENCY
                    && existing.consistency_score >= MIN_COALESCE_CONSISTENCY
                    && embedding::cosine_similarity(&observation.embedding, &existing.embedding)
                        >= SAME_VOICE_SPLIT_THRESHOLD
            })
        });
        if let Some(group_index) = compatible_group {
            groups[group_index].observation_indices.push(index);
            groups[group_index].embedding = average_embeddings(
                groups[group_index]
                    .observation_indices
                    .iter()
                    .map(|member| observations[*member].embedding.clone()),
            );
        } else {
            groups.push(VoiceObservationGroup {
                observation_indices: vec![index],
                embedding: observation.embedding.clone(),
            });
        }
    }
    groups
}

fn intervention_observations(
    embedded_windows: &[(SampleWindow, Vec<f32>)],
    selected_indices: &[usize],
) -> Vec<InterventionVoiceObservation> {
    let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in selected_indices {
        grouped
            .entry(embedded_windows[*index].0.segment_index)
            .or_default()
            .push(*index);
    }
    grouped
        .into_iter()
        .filter_map(|(segment_index, indices)| {
            let vectors = indices
                .iter()
                .map(|index| embedded_windows[*index].1.clone())
                .collect::<Vec<_>>();
            let embedding = average_embeddings(vectors.iter().cloned());
            if embedding.is_empty() {
                return None;
            }
            let start_ms = indices
                .iter()
                .map(|index| embedded_windows[*index].0.start_ms)
                .min()?;
            let end_ms = indices
                .iter()
                .map(|index| embedded_windows[*index].0.end_ms)
                .max()?;
            let selected_duration_ms = indices
                .iter()
                .map(|index| {
                    embedded_windows[*index]
                        .0
                        .end_ms
                        .saturating_sub(embedded_windows[*index].0.start_ms)
                })
                .sum();
            Some(InterventionVoiceObservation {
                segment_index,
                start_ms,
                end_ms,
                embedding,
                selected_duration_ms,
                consistency_score: mean_pairwise_similarity(&vectors),
            })
        })
        .collect()
}

fn suggested_split_clusters(
    observations: &[InterventionVoiceObservation],
) -> Option<Vec<Vec<usize>>> {
    if observations.len() < MIN_SPLIT_INTERVENTIONS_PER_CLUSTER * 2 {
        return None;
    }
    let mut farthest = None;
    for left in 0..observations.len() {
        for right in (left + 1)..observations.len() {
            let score = embedding::cosine_similarity(
                &observations[left].embedding,
                &observations[right].embedding,
            );
            if farthest
                .as_ref()
                .map(|(_, _, current)| score < *current)
                .unwrap_or(true)
            {
                farthest = Some((left, right, score));
            }
        }
    }
    let (left_seed, right_seed, seed_similarity) = farthest?;
    if seed_similarity > SPLIT_BETWEEN_CLUSTER_MAX {
        return None;
    }
    let mut left_centroid = observations[left_seed].embedding.clone();
    let mut right_centroid = observations[right_seed].embedding.clone();
    let mut assignments = vec![0usize; observations.len()];
    for _ in 0..4 {
        for (index, observation) in observations.iter().enumerate() {
            let left = embedding::cosine_similarity(&observation.embedding, &left_centroid);
            let right = embedding::cosine_similarity(&observation.embedding, &right_centroid);
            assignments[index] = usize::from(right > left);
        }
        let left_vectors = observations
            .iter()
            .zip(&assignments)
            .filter(|(_, assignment)| **assignment == 0)
            .map(|(observation, _)| observation.embedding.clone());
        let right_vectors = observations
            .iter()
            .zip(&assignments)
            .filter(|(_, assignment)| **assignment == 1)
            .map(|(observation, _)| observation.embedding.clone());
        let next_left = average_embeddings(left_vectors);
        let next_right = average_embeddings(right_vectors);
        if next_left.is_empty() || next_right.is_empty() {
            return None;
        }
        left_centroid = next_left;
        right_centroid = next_right;
    }
    let clusters = [0usize, 1usize]
        .into_iter()
        .map(|cluster| {
            observations
                .iter()
                .enumerate()
                .filter(|(index, _)| assignments[*index] == cluster)
                .map(|(index, _)| index)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for cluster in &clusters {
        if cluster.len() < MIN_SPLIT_INTERVENTIONS_PER_CLUSTER {
            return None;
        }
        let duration = cluster
            .iter()
            .map(|index| observations[*index].selected_duration_ms)
            .sum::<u64>();
        if duration < MIN_SPLIT_SPEECH_MS_PER_CLUSTER {
            return None;
        }
        let vectors = cluster
            .iter()
            .map(|index| observations[*index].embedding.clone())
            .collect::<Vec<_>>();
        if mean_pairwise_similarity(&vectors) < SPLIT_WITHIN_CLUSTER_THRESHOLD {
            return None;
        }
    }
    if embedding::cosine_similarity(&left_centroid, &right_centroid) > SPLIT_BETWEEN_CLUSTER_MAX {
        return None;
    }
    Some(
        clusters
            .into_iter()
            .map(|cluster| {
                cluster
                    .into_iter()
                    .map(|index| observations[index].segment_index)
                    .collect()
            })
            .collect(),
    )
}

fn model_unavailable_voice_reason(
    speech_intervals: Option<&[vad::SpeechInterval]>,
) -> &'static str {
    if speech_intervals.is_none() {
        "the local VAD model was unavailable, so Recall did not create a global voice profile"
    } else {
        "the local ECAPA model was unavailable, so Recall did not create a global voice profile"
    }
}

fn persist_model_unavailable_voice_groups(
    audio: &AudioClip,
    segments: &[TranscriptSegment],
    ordered_speakers: &[String],
    session_id: &str,
    db: &Db,
    speech_intervals: Option<&[vad::SpeechInterval]>,
) -> Result<Vec<MeetingLocalPreviewPersistence>, String> {
    let reason = model_unavailable_voice_reason(speech_intervals);
    let mut assignments = HashMap::new();
    let mut meeting_local_previews = HashMap::new();
    for diarized_speaker in ordered_speakers
        .iter()
        .filter(|label| label.as_str() != "unknown")
    {
        if let Some(speech_intervals) = speech_intervals {
            if let Some(window) =
                clean_sample_windows(audio, segments, diarized_speaker, speech_intervals)
                    .windows
                    .into_iter()
                    .next()
            {
                meeting_local_previews.insert(diarized_speaker.clone(), window.pcm);
            }
        }
        let group_id = db.insert_session_voice_group(&SessionVoiceGroupSave {
            session_id,
            provider_speaker_label: diarized_speaker,
            cluster_index: 0,
            resulting_speaker_id: None,
            status: "meeting_local_model_unavailable",
            centroid: None,
            selected_duration_ms: 0,
            selected_window_count: 0,
            consistency_score: None,
            model_version: None,
        })?;
        let provider_speakers = vec![diarized_speaker.clone()];
        db.insert_voice_match_decision(&VoiceMatchDecisionSave {
            session_id,
            provider_speakers: &provider_speakers,
            resulting_speaker_id: None,
            best_speaker_id: None,
            runner_up_speaker_id: None,
            best_score: None,
            runner_up_score: None,
            support_count: 0,
            selected_duration_ms: 0,
            selected_window_count: 0,
            consistency_score: None,
            model_version: EMBEDDING_VERSION,
            decision: VoiceMatchKind::Skipped.as_str(),
            reason,
        })?;
        assignments.insert(
            diarized_speaker.clone(),
            VoiceGroupAssignment {
                speaker_id: None,
                display_label: diarized_speaker.clone(),
                group_id,
            },
        );
    }
    for segment in segments {
        let assignment = assignments.get(&segment.speaker);
        db.insert_segment_with_provenance(
            session_id,
            segment.start_ms as i64,
            segment.end_ms as i64,
            None,
            assignment
                .map(|value| value.display_label.as_str())
                .or(Some("Unknown speaker")),
            (segment.speaker != "unknown").then_some(segment.speaker.as_str()),
            assignment.map(|value| value.group_id.as_str()),
            segment.text.trim(),
        )?;
    }

    let mut preview_results = Vec::new();
    for diarized_speaker in ordered_speakers
        .iter()
        .filter(|label| label.as_str() != "unknown")
    {
        let Some(preview_pcm) = meeting_local_previews.remove(diarized_speaker) else {
            continue;
        };
        let assignment = assignments
            .get(diarized_speaker)
            .expect("every non-unknown diarized speaker has a meeting-local group");
        let result = encode_wav_base64(&preview_pcm, audio.sample_rate).and_then(|sample| {
            db.upsert_voice_group_sample(&assignment.group_id, &sample, audio.sample_rate)
        });
        preview_results.push(MeetingLocalPreviewPersistence {
            diarized_speaker: diarized_speaker.clone(),
            result,
        });
    }
    Ok(preview_results)
}

fn process_segments(
    audio: &AudioClip,
    segments: &[TranscriptSegment],
    session_id: &str,
    db: &Db,
    embedder: Option<&embedding::Embedder>,
    speech_intervals: Option<&[vad::SpeechInterval]>,
    progress: (&tauri::AppHandle, &str),
) -> Result<(), String> {
    let (app_handle, run_id) = progress;
    let known = db.list_embeddings(EMBEDDING_VERSION)?;
    let mut ordered_speakers = Vec::new();
    let mut seen = HashSet::new();
    for segment in segments {
        if seen.insert(segment.speaker.clone()) {
            ordered_speakers.push(segment.speaker.clone());
        }
    }
    let mut observations = Vec::new();
    let mut intervention_observations_by_speaker: HashMap<
        String,
        Vec<InterventionVoiceObservation>,
    > = HashMap::new();
    let mut skipped_reasons: HashMap<String, String> = HashMap::new();
    let mut meeting_local_previews: HashMap<String, Vec<f32>> = HashMap::new();

    if embedder.is_none() || speech_intervals.is_none() {
        let reason = model_unavailable_voice_reason(speech_intervals);
        let preview_results = persist_model_unavailable_voice_groups(
            audio,
            segments,
            &ordered_speakers,
            session_id,
            db,
            speech_intervals,
        )?;
        for preview_result in preview_results {
            let diarized_speaker = preview_result.diarized_speaker;
            let result = preview_result.result;
            match result {
                Ok(()) => emit_progress(
                    app_handle,
                    "voiceprint:meeting-local-sample:stored",
                    Some(format!(
                        "Stored a VAD-confirmed meeting-local preview for {diarized_speaker} because the ECAPA model was unavailable"
                    )),
                    Some(run_id),
                ),
                Err(error) => emit_progress(
                    app_handle,
                    "voiceprint:warning",
                    Some(format!(
                        "{diarized_speaker}: could not retain the meeting-local preview: {error}"
                    )),
                    Some(run_id),
                ),
            }
        }
        emit_progress(
            app_handle,
            "voiceprint:skipped",
            Some(reason.into()),
            Some(run_id),
        );
        return Ok(());
    }
    let embedder = embedder.expect("checked above");
    let speech_intervals = speech_intervals.expect("checked above");

    for diarized_speaker in &ordered_speakers {
        if diarized_speaker == "unknown" {
            emit_progress(
                app_handle,
                "voiceprint:skipped",
                Some("Provider audio without a speaker label remains Unknown speaker".into()),
                Some(run_id),
            );
            continue;
        }
        let window_set = clean_sample_windows(audio, segments, diarized_speaker, speech_intervals);
        if window_set.windows.is_empty() {
            let reason = format!(
                "no VAD-confirmed speech excerpt of at least {:.1} seconds ({} overlapping interventions, {} short interventions, {} without speech, {} short speech runs)",
                MIN_SPEAKER_MS as f64 / 1_000.0,
                window_set.overlapping_segments,
                window_set.short_segments,
                window_set.no_speech_segments,
                window_set.short_speech_intervals,
            );
            skipped_reasons.insert(diarized_speaker.clone(), reason.clone());
            emit_progress(
                app_handle,
                "voiceprint:skipped",
                Some(format!("{diarized_speaker}: {reason}; kept meeting-local")),
                Some(run_id),
            );
            continue;
        }

        let mut embedded_windows = Vec::new();
        let mut candidate_windows = window_set.windows.into_iter().peekable();
        let mut trusted_batch = None;
        for batch_index in 0..MAX_SAMPLE_CANDIDATE_BATCHES {
            while candidate_windows
                .peek()
                .map(|window| window.candidate_batch == batch_index)
                .unwrap_or(false)
            {
                let window = candidate_windows.next().expect("peeked candidate window");
                meeting_local_previews
                    .entry(diarized_speaker.clone())
                    .or_insert_with(|| window.pcm.clone());
                match embedder.embed(&window.pcm, audio.sample_rate) {
                    Ok(embedding) => embedded_windows.push((window, embedding)),
                    Err(error) => emit_progress(
                        app_handle,
                        "voiceprint:warning",
                        Some(format!(
                            "{diarized_speaker}: rejected one candidate excerpt: {error}"
                        )),
                        Some(run_id),
                    ),
                }
            }
            trusted_batch = first_trusted_sample_batch(&embedded_windows);
            if trusted_batch.is_some() {
                break;
            }
        }

        // Split review must see both the trusted majority and any outlier
        // interventions from every batch that was evaluated. Later unneeded
        // batches remain unembedded once a trusted batch has been found.
        let all_window_indices = (0..embedded_windows.len()).collect::<Vec<_>>();
        let intervention_observations =
            intervention_observations(&embedded_windows, &all_window_indices);
        if !intervention_observations.is_empty() {
            intervention_observations_by_speaker
                .insert(diarized_speaker.clone(), intervention_observations.clone());
        }
        let Some(trusted_batch) = trusted_batch else {
            let reason =
                "no bounded candidate batch had a strict internally consistent majority for a trusted voiceprint"
                    .to_string();
            skipped_reasons.insert(diarized_speaker.clone(), reason.clone());
            emit_progress(
                app_handle,
                "voiceprint:skipped",
                Some(format!("{diarized_speaker}: {reason}; kept meeting-local")),
                Some(run_id),
            );
            continue;
        };
        let consistent_indices = trusted_batch.window_indices;

        let target_samples = ((audio.sample_rate as u64 * TARGET_SPEAKER_MS) / 1_000) as usize;
        let mut selected_samples_total = 0usize;
        let mut selected_windows = 0usize;
        let mut selected_ms = 0u64;
        let mut selected_vectors = Vec::new();
        let mut selected_indices = Vec::new();
        for index in consistent_indices.iter().copied() {
            let window = &embedded_windows[index].0;
            let remaining = target_samples.saturating_sub(selected_samples_total);
            if remaining == 0 {
                break;
            }
            let selected_samples = remaining.min(window.pcm.len());
            selected_samples_total += selected_samples;
            selected_vectors.push(embedded_windows[index].1.clone());
            selected_indices.push(index);
            selected_windows += 1;
            selected_ms += if audio.sample_rate == 0 {
                0
            } else {
                (selected_samples as u64 * 1_000) / audio.sample_rate as u64
            };
        }
        let sample_duration_ms = if audio.sample_rate == 0 {
            0
        } else {
            (selected_samples_total as u64 * 1_000) / audio.sample_rate as u64
        };
        if sample_duration_ms < MIN_SPEAKER_MS {
            emit_progress(
                app_handle,
                "voiceprint:skipped",
                Some(format!(
                    "{diarized_speaker}: consistent clean speech was shorter than {:.1} seconds; keeping the provider voice for manual review",
                    MIN_SPEAKER_MS as f64 / 1_000.0,
                )),
                Some(run_id),
            );
            continue;
        }
        let embedding = average_embeddings(selected_vectors.iter().cloned());
        if embedding.is_empty() {
            emit_progress(
                app_handle,
                "voiceprint:warning",
                Some(format!(
                    "{diarized_speaker}: clean excerpts did not produce a usable centroid"
                )),
                Some(run_id),
            );
            continue;
        }
        let consistency_score = mean_pairwise_similarity(&selected_vectors);
        let preview_pcm = selected_indices
            .iter()
            .map(|index| &embedded_windows[*index].0)
            .max_by_key(|window| window.end_ms.saturating_sub(window.start_ms))
            .map(|window| window.pcm.clone())
            .unwrap_or_default();
        let consistency_rejections = trusted_batch
            .candidate_count
            .saturating_sub(consistent_indices.len());
        let fallback_batch_note = if trusted_batch.batch_index == 0 {
            String::new()
        } else {
            format!(
                " using fallback batch {}",
                trusted_batch.batch_index.saturating_add(1)
            )
        };
        emit_progress(
            app_handle,
            "voiceprint:sample:selected",
            Some(format!(
                "{diarized_speaker}: selected {selected_windows} clean central excerpt{} ({:.1}s){fallback_batch_note}; rejected {} inconsistent, {} overlapping, {} short, and {} silent candidate{}",
                if selected_windows == 1 { "" } else { "s" },
                sample_duration_ms.min(selected_ms) as f64 / 1_000.0,
                consistency_rejections,
                window_set.overlapping_segments,
                window_set.short_segments,
                window_set.no_speech_segments + window_set.short_speech_intervals,
                if consistency_rejections
                    + window_set.overlapping_segments
                    + window_set.short_segments
                    + window_set.no_speech_segments
                    + window_set.short_speech_intervals
                    == 1
                {
                    ""
                } else {
                    "s"
                },
            )),
            Some(run_id),
        );

        observations.push(VoiceObservation {
            diarized_speaker: diarized_speaker.clone(),
            pcm: preview_pcm,
            embedding,
            clean_window_count: selected_windows,
            selected_duration_ms: sample_duration_ms.min(selected_ms),
            consistency_score,
        });
    }

    let groups = group_voice_observations(&observations);
    let mut candidates = groups
        .iter()
        .map(|group| classify_speaker_match(&group.embedding, &known))
        .collect::<Vec<_>>();
    resolve_unique_profile_matches(&mut candidates);
    let mut mapping: HashMap<String, VoiceGroupAssignment> = HashMap::new();

    for (group, candidate) in groups.into_iter().zip(candidates) {
        let representative_index = group
            .observation_indices
            .iter()
            .copied()
            .max_by_key(|index| {
                (
                    observations[*index].clean_window_count,
                    observations[*index].pcm.len(),
                )
            })
            .ok_or_else(|| "Voice observation group was unexpectedly empty".to_string())?;
        let representative = &observations[representative_index];
        let diarized_speakers = group
            .observation_indices
            .iter()
            .map(|index| observations[*index].diarized_speaker.clone())
            .collect::<Vec<_>>();
        let diarized_label = diarized_speakers.join(" + ");
        let (speaker_id, label, is_new) = match candidate.kind {
            VoiceMatchKind::Automatic => {
                let matched = candidate
                    .best
                    .as_ref()
                    .ok_or_else(|| "Automatic voice match had no identity".to_string())?;
                emit_progress(
                    app_handle,
                    "voiceprint:matched",
                    Some(format!(
                        "{diarized_label} → {} automatically ({:.3}; {}; reference left unchanged)",
                        matched.label, matched.score, candidate.reason
                    )),
                    Some(run_id),
                );
                (matched.speaker_id.clone(), matched.label.clone(), false)
            }
            VoiceMatchKind::Suggested | VoiceMatchKind::New => {
                let label = db.next_voice_label()?;
                let speaker_id = db.insert_speaker(Some(&label))?;
                let (stage, detail) = if candidate.kind == VoiceMatchKind::Suggested {
                    let likely = candidate
                        .best
                        .as_ref()
                        .ok_or_else(|| "Voice suggestion had no identity".to_string())?;
                    (
                        "voiceprint:suggested",
                        format!(
                            "{diarized_label} → {label}; likely {} at {:.3}, kept for one-click review ({})",
                            likely.label, likely.score, candidate.reason
                        ),
                    )
                } else {
                    (
                        "voiceprint:new",
                        format!("{diarized_label} → {label} ({})", candidate.reason),
                    )
                };
                emit_progress(app_handle, stage, Some(detail), Some(run_id));
                (speaker_id, label, true)
            }
            VoiceMatchKind::Skipped => {
                return Err("A skipped voice observation reached the matching stage".into());
            }
        };

        // New provisional voices establish a reference. Automatic matches are
        // intentionally not fed back into the reference library: only a human
        // naming or assigning a provisional profile can expand a known person.
        if is_new {
            db.insert_embedding(&speaker_id, session_id, &group.embedding, EMBEDDING_VERSION)?;
        }
        if is_new && !representative.pcm.is_empty() {
            let sample = encode_wav_base64(&representative.pcm, audio.sample_rate)?;
            db.insert_sample(&speaker_id, &sample, audio.sample_rate)?;
            emit_progress(
                app_handle,
                "voiceprint:sample:stored",
                Some(format!("Stored a temporary preview for {label}")),
                Some(run_id),
            );
        }
        let selected_window_count = group
            .observation_indices
            .iter()
            .map(|index| observations[*index].clean_window_count)
            .sum::<usize>();
        let selected_duration_ms = group
            .observation_indices
            .iter()
            .map(|index| observations[*index].selected_duration_ms)
            .sum::<u64>();
        let observation_consistency = group
            .observation_indices
            .iter()
            .map(|index| observations[*index].consistency_score)
            .fold(1.0_f32, f32::min);
        let group_vectors = group
            .observation_indices
            .iter()
            .map(|index| observations[*index].embedding.clone())
            .collect::<Vec<_>>();
        let consistency_score =
            observation_consistency.min(mean_pairwise_similarity(&group_vectors));
        db.insert_voice_match_decision(&VoiceMatchDecisionSave {
            session_id,
            provider_speakers: &diarized_speakers,
            resulting_speaker_id: Some(&speaker_id),
            best_speaker_id: candidate
                .best
                .as_ref()
                .map(|match_| match_.speaker_id.as_str()),
            runner_up_speaker_id: candidate
                .runner_up
                .as_ref()
                .map(|match_| match_.speaker_id.as_str()),
            best_score: candidate.best.as_ref().map(|match_| match_.score),
            runner_up_score: candidate.runner_up.as_ref().map(|match_| match_.score),
            support_count: candidate
                .best
                .as_ref()
                .map(|match_| match_.support_count)
                .unwrap_or(0),
            selected_duration_ms,
            selected_window_count,
            consistency_score: Some(consistency_score),
            model_version: EMBEDDING_VERSION,
            decision: candidate.kind.as_str(),
            reason: &candidate.reason,
        })?;
        if diarized_speakers.len() > 1 {
            emit_progress(
                app_handle,
                "voiceprint:labels:coalesced",
                Some(format!(
                    "Combined {} provider speaker labels because their clean voiceprints agreed at {:.2} or higher",
                    diarized_speakers.len(),
                    SAME_VOICE_SPLIT_THRESHOLD,
                )),
                Some(run_id),
            );
        }
        let group_status = match candidate.kind {
            VoiceMatchKind::Automatic => "automatic",
            VoiceMatchKind::Suggested => "provisional_suggested",
            VoiceMatchKind::New => "provisional_new",
            VoiceMatchKind::Skipped => "meeting_local",
        };
        for observation_index in &group.observation_indices {
            let observation = &observations[*observation_index];
            let voice_group_id = db.insert_session_voice_group(&SessionVoiceGroupSave {
                session_id,
                provider_speaker_label: &observation.diarized_speaker,
                cluster_index: 0,
                resulting_speaker_id: Some(&speaker_id),
                status: group_status,
                centroid: Some(&observation.embedding),
                selected_duration_ms: observation.selected_duration_ms,
                selected_window_count: observation.clean_window_count,
                consistency_score: Some(observation.consistency_score),
                model_version: Some(EMBEDDING_VERSION),
            })?;
            mapping.insert(
                observation.diarized_speaker.clone(),
                VoiceGroupAssignment {
                    speaker_id: Some(speaker_id.clone()),
                    display_label: label.clone(),
                    group_id: voice_group_id,
                },
            );
        }
    }

    for diarized_speaker in ordered_speakers {
        if diarized_speaker == "unknown" || mapping.contains_key(&diarized_speaker) {
            continue;
        }
        let reason = skipped_reasons
            .get(&diarized_speaker)
            .cloned()
            .unwrap_or_else(|| "no safe VAD-confirmed ECAPA observation was available".into());
        let group_id = db.insert_session_voice_group(&SessionVoiceGroupSave {
            session_id,
            provider_speaker_label: &diarized_speaker,
            cluster_index: 0,
            resulting_speaker_id: None,
            status: "meeting_local_no_safe_speech",
            centroid: None,
            selected_duration_ms: 0,
            selected_window_count: 0,
            consistency_score: None,
            model_version: Some(EMBEDDING_VERSION),
        })?;
        emit_progress(
            app_handle,
            "voiceprint:meeting-local",
            Some(format!(
                "{diarized_speaker}: no global VOICE profile was created; {reason}"
            )),
            Some(run_id),
        );
        let provider_speakers = vec![diarized_speaker.clone()];
        db.insert_voice_match_decision(&VoiceMatchDecisionSave {
            session_id,
            provider_speakers: &provider_speakers,
            resulting_speaker_id: None,
            best_speaker_id: None,
            runner_up_speaker_id: None,
            best_score: None,
            runner_up_score: None,
            support_count: 0,
            selected_duration_ms: 0,
            selected_window_count: 0,
            consistency_score: None,
            model_version: EMBEDDING_VERSION,
            decision: VoiceMatchKind::Skipped.as_str(),
            reason: &reason,
        })?;
        mapping.insert(
            diarized_speaker.clone(),
            VoiceGroupAssignment {
                speaker_id: None,
                display_label: diarized_speaker,
                group_id,
            },
        );
    }

    let mut segment_ids = HashMap::new();
    for (segment_index, segment) in segments.iter().enumerate() {
        let mapped = mapping.get(&segment.speaker);
        let segment_id = db.insert_segment_with_provenance(
            session_id,
            segment.start_ms as i64,
            segment.end_ms as i64,
            mapped.and_then(|value| value.speaker_id.as_deref()),
            mapped
                .map(|value| value.display_label.as_str())
                .or(Some("Unknown speaker")),
            (segment.speaker != "unknown").then_some(segment.speaker.as_str()),
            mapped.map(|value| value.group_id.as_str()),
            segment.text.trim(),
        )?;
        segment_ids.insert(segment_index, segment_id);
    }

    for (diarized_speaker, preview_pcm) in meeting_local_previews {
        let Some(assignment) = mapping.get(&diarized_speaker) else {
            continue;
        };
        if assignment.speaker_id.is_some() || preview_pcm.is_empty() {
            continue;
        }
        match encode_wav_base64(&preview_pcm, audio.sample_rate).and_then(|sample| {
            db.upsert_voice_group_sample(&assignment.group_id, &sample, audio.sample_rate)
        }) {
            Ok(()) => emit_progress(
                app_handle,
                "voiceprint:meeting-local-sample:stored",
                Some(format!(
                    "Stored a meeting-local preview for {diarized_speaker}"
                )),
                Some(run_id),
            ),
            Err(error) => emit_progress(
                app_handle,
                "voiceprint:warning",
                Some(format!(
                    "{diarized_speaker}: could not retain the meeting-local preview: {error}"
                )),
                Some(run_id),
            ),
        }
    }

    for (diarized_speaker, interventions) in &intervention_observations_by_speaker {
        let Some(assignment) = mapping.get(diarized_speaker) else {
            continue;
        };
        for intervention in interventions {
            let Some(segment_id) = segment_ids.get(&intervention.segment_index) else {
                continue;
            };
            db.insert_voice_observation(&VoiceObservationSave {
                voice_group_id: &assignment.group_id,
                session_id,
                segment_id,
                start_ms: intervention.start_ms,
                end_ms: intervention.end_ms,
                vector: &intervention.embedding,
                model_version: EMBEDDING_VERSION,
                speech_duration_ms: intervention.selected_duration_ms,
                consistency_score: intervention.consistency_score,
            })?;
        }
        if let Some(clusters) = suggested_split_clusters(interventions) {
            let persisted_clusters = clusters
                .into_iter()
                .map(|cluster| {
                    cluster
                        .into_iter()
                        .filter_map(|segment_index| segment_ids.get(&segment_index).cloned())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            db.set_voice_group_split_suggestion(&assignment.group_id, &persisted_clusters)?;
            emit_progress(
                app_handle,
                "voiceprint:split:suggested",
                Some(format!(
                    "{} may contain multiple people; review the suggested intervention split",
                    diarized_speaker
                )),
                Some(run_id),
            );
        }
    }
    Ok(())
}

fn is_provisional_label(label: &str) -> bool {
    label
        .strip_prefix("VOICE")
        .map(|suffix| !suffix.is_empty() && suffix.chars().all(|value| value.is_ascii_digit()))
        .unwrap_or(false)
}

fn is_matchable_person_label(label: &str) -> bool {
    let trimmed = label.trim();
    !trimmed.is_empty()
        && !is_provisional_label(trimmed)
        && !trimmed.eq_ignore_ascii_case("unknown speaker")
        && !trimmed.eq_ignore_ascii_case("unnamed voice")
}

fn ranked_identity_matches(
    query: &[f32],
    known: &[StoredEmbedding],
) -> (Vec<IdentityMatch>, usize) {
    let mut normalized_profiles: HashMap<String, HashSet<String>> = HashMap::new();
    for candidate in known {
        let Some(label) = candidate.speaker_label.as_deref() else {
            continue;
        };
        if is_provisional_label(label) {
            continue;
        }
        if !is_matchable_person_label(label) {
            continue;
        }
        normalized_profiles
            .entry(db::normalized_person_name(label))
            .or_default()
            .insert(candidate.speaker_id.clone());
    }
    let conflicted_profiles = normalized_profiles
        .values()
        .filter(|profiles| profiles.len() > 1)
        .flat_map(|profiles| profiles.iter().cloned())
        .collect::<HashSet<_>>();

    let mut by_speaker: HashMap<&str, IdentityMatch> = HashMap::new();
    for candidate in known {
        let Some(label) = candidate.speaker_label.as_deref() else {
            continue;
        };
        if !is_matchable_person_label(label)
            || conflicted_profiles.contains(candidate.speaker_id.as_str())
        {
            continue;
        }
        let score = embedding::cosine_similarity(query, &candidate.vector);
        let entry = by_speaker
            .entry(candidate.speaker_id.as_str())
            .or_insert_with(|| IdentityMatch {
                speaker_id: candidate.speaker_id.clone(),
                label: label.to_string(),
                score,
                support_count: 0,
            });
        if score > entry.score {
            entry.score = score;
        }
        if score >= MATCH_THRESHOLD {
            entry.support_count += 1;
        }
    }
    let mut ranked = by_speaker.into_values().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.score.total_cmp(&left.score));
    (ranked, conflicted_profiles.len())
}

fn classify_speaker_match(query: &[f32], known: &[StoredEmbedding]) -> VoiceMatchCandidate {
    let (ranked, conflicted_profile_count) = ranked_identity_matches(query, known);
    let duplicate_note = if conflicted_profile_count == 0 {
        String::new()
    } else {
        format!(
            "; excluded {conflicted_profile_count} profile{} in unresolved duplicate-name groups",
            if conflicted_profile_count == 1 {
                ""
            } else {
                "s"
            }
        )
    };
    let Some(best) = ranked.first().cloned() else {
        return VoiceMatchCandidate {
            kind: VoiceMatchKind::New,
            best: None,
            runner_up: None,
            reason: format!("no eligible named voiceprint was available{duplicate_note}"),
        };
    };
    let runner_up = ranked.get(1).cloned();
    let runner_up_score = runner_up.as_ref().map(|match_| match_.score);
    if best.score < MATCH_THRESHOLD {
        return VoiceMatchCandidate {
            kind: VoiceMatchKind::New,
            reason: format!(
                "best named identity {} scored {:.3}, below {:.2}{duplicate_note}",
                best.label, best.score, MATCH_THRESHOLD
            ),
            best: Some(best),
            runner_up,
        };
    }

    let different_person_lead = best.score - runner_up_score.unwrap_or(-1.0);
    let strong_single_match =
        best.score >= STRONG_MATCH_THRESHOLD && different_person_lead >= STRONG_MATCH_MARGIN;
    let multi_reference_consensus =
        best.support_count >= 2 && runner_up_score.unwrap_or(-1.0) < MATCH_THRESHOLD;
    if strong_single_match || multi_reference_consensus {
        let reason = if strong_single_match {
            format!(
                "{} scored {:.3} with a {:.3} lead over the next different identity{}",
                best.label, best.score, different_person_lead, duplicate_note
            )
        } else {
            format!(
                "{} had {} agreeing references at or above {:.2}; every different identity stayed below {:.2}{}",
                best.label,
                best.support_count,
                MATCH_THRESHOLD,
                MATCH_THRESHOLD,
                duplicate_note
            )
        };
        VoiceMatchCandidate {
            kind: VoiceMatchKind::Automatic,
            best: Some(best),
            runner_up,
            reason,
        }
    } else {
        let runner_up_description = runner_up
            .as_ref()
            .map(|match_| format!("{} at {:.3}", match_.label, match_.score))
            .unwrap_or_else(|| "no different named identity".into());
        VoiceMatchCandidate {
            kind: VoiceMatchKind::Suggested,
            reason: format!(
                "{} scored {:.3} ({} agreeing reference{}); runner-up was {runner_up_description}. The evidence is strong enough to suggest but not assign automatically{duplicate_note}",
                best.label,
                best.score,
                best.support_count,
                if best.support_count == 1 { "" } else { "s" },
            ),
            best: Some(best),
            runner_up,
        }
    }
}

fn resolve_unique_profile_matches(candidates: &mut [VoiceMatchCandidate]) {
    let mut claims: HashMap<String, Vec<(usize, f32)>> = HashMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.kind == VoiceMatchKind::Automatic {
            let candidate = candidate
                .best
                .as_ref()
                .expect("automatic match must have a best identity");
            claims
                .entry(candidate.speaker_id.clone())
                .or_default()
                .push((index, candidate.score));
        }
    }

    for mut profile_claims in claims.into_values() {
        if profile_claims.len() < 2 {
            continue;
        }
        profile_claims.sort_by(|left, right| right.1.total_cmp(&left.1));
        let (_, best_score) = profile_claims[0];
        let runner_up_score = profile_claims[1].1;
        if best_score - runner_up_score >= PROFILE_CLAIM_MARGIN {
            for (index, _) in profile_claims.into_iter().skip(1) {
                candidates[index].kind = VoiceMatchKind::Suggested;
                candidates[index].reason.push_str(
                    "; another provider voice made a clearly stronger claim to this person in the same recording",
                );
            }
        } else {
            for (index, _) in profile_claims {
                candidates[index].kind = VoiceMatchKind::Suggested;
                candidates[index].reason.push_str(
                    "; multiple provider voices made close claims to this person in the same recording",
                );
            }
        }
    }
}

fn encode_wav_base64(pcm: &[f32], sample_rate: u32) -> Result<String, String> {
    use std::io::Cursor;
    let mut buffer = Vec::new();
    let cursor = Cursor::new(&mut buffer);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(cursor, spec).map_err(|error| error.to_string())?;
    for sample in pcm {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .map_err(|error| error.to_string())?;
    }
    writer.finalize().map_err(|error| error.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(buffer))
}

#[tauri::command]
fn save_soniox_key(api_key: String, app_state: State<AppState>) -> Result<(), String> {
    app_state.save_soniox_key(&api_key)
}

#[tauri::command]
fn delete_soniox_key(app_state: State<AppState>) -> Result<(), String> {
    app_state.delete_soniox_key()
}

#[tauri::command]
fn soniox_key_status(app_state: State<AppState>) -> bool {
    app_state.has_soniox_key()
}

#[tauri::command]
fn save_openai_key(api_key: String, app_state: State<AppState>) -> Result<(), String> {
    app_state.save_openai_key(&api_key)
}

#[tauri::command]
fn delete_openai_key(app_state: State<AppState>) -> Result<(), String> {
    app_state.delete_openai_key()
}

#[tauri::command]
fn openai_key_status(app_state: State<AppState>) -> bool {
    app_state.has_openai_key()
}

#[tauri::command]
fn get_preferences(app_state: State<AppState>) -> Result<AppConfig, String> {
    app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned".to_string())
        .map(|config| config.clone())
}

#[tauri::command]
fn list_translation_languages() -> Vec<soniox::TranslationLanguage> {
    soniox::supported_translation_languages()
}

#[tauri::command]
fn save_preferences(
    preferences: PreferenceUpdate,
    app_state: State<AppState>,
) -> Result<(), String> {
    let PreferenceUpdate {
        selected_input_device,
        language_hints,
        live_transcription,
        openai_model,
        preferred_language,
        no_translation_languages,
    } = preferences;
    let openai_model = openai_model.trim();
    if openai_model.is_empty() {
        return Err("LLM model cannot be empty".into());
    }
    let normalized_hints = language_hints
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let mut config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?;
    let requested_preferred = preferred_language
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-")
        .split('-')
        .next()
        .unwrap_or_default()
        .to_string();
    let preferred_language = match soniox::normalize_translation_language(&requested_preferred) {
        Some(language) => language,
        None if !requested_preferred.is_empty()
            && requested_preferred == config.preferred_language =>
        {
            requested_preferred
        }
        None => return Err("Choose a supported preferred language".into()),
    };
    let mut excluded_languages = no_translation_languages
        .into_iter()
        .filter_map(|value| soniox::normalize_translation_language(&value))
        .filter(|value| value != &preferred_language)
        .collect::<Vec<_>>();
    excluded_languages.sort();
    excluded_languages.dedup();
    config.selected_input_device = selected_input_device.filter(|value| !value.trim().is_empty());
    config.language_hints = normalized_hints;
    config.live_transcription = live_transcription;
    config.openai_model = openai_model.to_string();
    config.preferred_language = preferred_language;
    config.no_translation_languages = excluded_languages;
    config.save(&app_state.config_path)
}

#[tauri::command]
fn complete_onboarding(version: String, app_state: State<AppState>) -> Result<(), String> {
    if version != ONBOARDING_VERSION {
        return Err("Unsupported onboarding version".into());
    }
    let mut config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?;
    config.onboarding_version = Some(version);
    config.save(&app_state.config_path)
}

#[tauri::command]
fn unlock_db(password: String, app_state: State<AppState>) -> Result<(), String> {
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?
        .clone();
    if !config.encryption_enabled {
        return Err("Local database encryption is not enabled".into());
    }
    let salt = Db::load_existing_salt(app_state.db_path()).unwrap_or(None);
    app_state.unlock_db(Crypto::new(Some(&password), salt))
}

#[tauri::command]
fn enable_encryption(_password: String, _app_state: State<AppState>) -> Result<(), String> {
    Err(
        "Encryption migration is not implemented safely yet; existing data was left unchanged"
            .into(),
    )
}

#[tauri::command]
fn app_status(
    app_state: State<AppState>,
    manager: State<RecordingManager>,
) -> Result<AppStatus, String> {
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?
        .clone();
    let db_open = app_state
        .db
        .lock()
        .map_err(|_| "Database lock poisoned")?
        .is_some();
    let current_recording = manager.current_stt_context();
    Ok(AppStatus {
        encryption_enabled: config.encryption_enabled,
        db_open,
        needs_password: config.encryption_enabled && !db_open,
        recording: manager.is_recording(),
        soniox_key_configured: app_state.has_soniox_key(),
        openai_key_configured: app_state.has_openai_key(),
        speaker_model_available: app_state.model_path.is_file(),
        selected_input_device: config.selected_input_device,
        language_hints: config.language_hints,
        live_transcription: config.live_transcription,
        current_stt_context: current_recording
            .as_ref()
            .map(|(context, _)| context.clone()),
        live_recording_active: current_recording
            .as_ref()
            .is_some_and(|(_, live_active)| *live_active),
    })
}

#[tauri::command]
fn list_sessions(app_state: State<AppState>) -> Result<Vec<SessionSummary>, String> {
    app_state.db_handle()?.list_session_summaries()
}

#[tauri::command]
fn search_session_ids(query: String, app_state: State<AppState>) -> Result<Vec<String>, String> {
    app_state.db_handle()?.search_session_ids(&query)
}

#[tauri::command]
fn list_segments(
    session_id: String,
    app_state: State<AppState>,
) -> Result<Vec<SegmentRecord>, String> {
    app_state.db_handle()?.list_segments(&session_id)
}

#[tauri::command]
fn update_transcript(
    session_id: String,
    transcript: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    app_state
        .db_handle()?
        .update_session_transcript(&session_id, &transcript)
}

#[tauri::command]
fn update_session_title(
    session_id: String,
    title: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    app_state
        .db_handle()?
        .update_session_title(&session_id, &title)
}

#[tauri::command]
fn update_segment_text(
    segment_id: String,
    session_id: String,
    text: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    let db = app_state.db_handle()?;
    db.update_segment_text(&session_id, &segment_id, &text)?;
    refresh_session_transcript(&db, &session_id)
}

#[tauri::command]
fn assign_segment_speaker(
    segment_id: String,
    session_id: String,
    speaker_id: Option<String>,
    app_state: State<AppState>,
) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let sessions = vec![session_id.clone()];
    claim_identity_sessions(app_state.inner(), &sessions)?;
    let result = db
        .assign_segment_speaker(&session_id, &segment_id, speaker_id.as_deref())
        .and_then(|_| refresh_session_transcript(&db, &session_id));
    release_identity_sessions(app_state.inner(), &sessions);
    result
}

#[tauri::command]
fn delete_session(session_id: String, app_state: State<AppState>) -> Result<usize, String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    let db = app_state.db_handle()?;
    if let Some(job) = db.processing_job(&session_id)? {
        if matches!(job.status.as_str(), "queued" | "processing") {
            return Err(
                "This conversation is still being processed. Wait for it to finish or fail before deleting it."
                    .into(),
            );
        }
        remove_managed_audio(Path::new(&job.audio_path), app_state.inner())?;
    }
    db.delete_session(&session_id)
}

fn recap_snapshot(db: &Db, session_id: &str) -> Result<RecapSnapshot, String> {
    let source = db.recap_source_snapshot(session_id)?;
    let mut seen_unresolved = HashSet::new();
    let unresolved_profiles = source
        .segments
        .iter()
        .filter_map(|segment| {
            let unresolved = segment.speaker_id.is_none()
                || segment.speaker_label.trim().is_empty()
                || segment
                    .speaker_label
                    .eq_ignore_ascii_case("Unknown speaker")
                || is_provisional_label(&segment.speaker_label);
            (unresolved && seen_unresolved.insert(segment.speaker_label.clone()))
                .then(|| segment.speaker_label.clone())
        })
        .collect::<Vec<_>>();
    Ok(RecapSnapshot {
        meeting_created_at: source.meeting_created_at,
        segments: source.segments,
        agenda: source.agenda,
        source_fingerprint: source.source_fingerprint,
        unresolved_profiles,
    })
}

fn standard_recap_prompts(
    db: &Db,
    variable_context: &RecapPromptVariableContext,
) -> Result<StandardRecapPrompts, String> {
    let mut executive_summary = None;
    let mut full_summary = None;
    let mut actions = None;
    for recap_type in db.list_recap_types()? {
        match recap_type.id.as_str() {
            BUILTIN_EXECUTIVE_SUMMARY_ID => executive_summary = Some(recap_type.prompt),
            BUILTIN_FULL_SUMMARY_ID => full_summary = Some(recap_type.prompt),
            BUILTIN_ACTIONS_ID => actions = Some(recap_type.prompt),
            _ => {}
        }
    }
    let templates = StandardRecapPrompts {
        executive_summary: executive_summary
            .ok_or_else(|| "The Executive summary recap type is missing".to_string())?,
        full_summary: full_summary
            .ok_or_else(|| "The Full summary recap type is missing".to_string())?,
        actions: actions.ok_or_else(|| "The Actions recap type is missing".to_string())?,
    };
    Ok(StandardRecapPrompts {
        executive_summary: expand_recap_prompt(&templates.executive_summary, variable_context),
        full_summary: expand_recap_prompt(&templates.full_summary, variable_context),
        actions: expand_recap_prompt(&templates.actions, variable_context),
    })
}

fn recap_prompt_variable_context(snapshot: &RecapSnapshot) -> RecapPromptVariableContext {
    RecapPromptVariableContext::from_desktop_local(snapshot.meeting_created_at)
}

fn recap_snapshot_from(
    db: &Db,
    session: &Session,
    stored_segments: &[SegmentRecord],
) -> Result<RecapSnapshot, String> {
    let mut segments = stored_segments
        .iter()
        .filter(|segment| !segment.text.trim().is_empty())
        .map(|segment| RecapSourceSegment {
            id: segment.id.clone(),
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            speaker_id: segment.speaker_id.clone(),
            speaker_label: segment
                .speaker_label
                .clone()
                .filter(|label| !label.trim().is_empty())
                .unwrap_or_else(|| "Unknown speaker".to_string()),
            text: segment.text.clone(),
        })
        .collect::<Vec<_>>();
    if segments.is_empty() && !session.transcript.trim().is_empty() {
        segments.push(RecapSourceSegment {
            id: format!("legacy-{}", session.id),
            start_ms: 0,
            end_ms: session.duration_ms,
            speaker_id: None,
            speaker_label: "Unknown speaker".into(),
            text: session.transcript.clone(),
        });
    }
    if segments.is_empty() {
        return Err("This conversation has no transcript to recap".into());
    }
    let mut seen_unresolved = HashSet::new();
    let unresolved_profiles = segments
        .iter()
        .filter_map(|segment| {
            let unresolved = segment.speaker_id.is_none()
                || segment.speaker_label.trim().is_empty()
                || segment
                    .speaker_label
                    .eq_ignore_ascii_case("Unknown speaker")
                || is_provisional_label(&segment.speaker_label);
            (unresolved && seen_unresolved.insert(segment.speaker_label.clone()))
                .then(|| segment.speaker_label.clone())
        })
        .collect::<Vec<_>>();
    let agenda = db.load_agenda(&session.id)?;
    let agenda_fingerprint = agenda.as_ref().map(|agenda| AgendaFingerprint {
        source_kind: &agenda.source_kind,
        filename: &agenda.filename,
        mime_type: &agenda.mime_type,
        content: &agenda.content,
    });
    let source_fingerprint = recap::source_fingerprint(&segments, agenda_fingerprint)?;
    Ok(RecapSnapshot {
        meeting_created_at: session.created_at,
        segments,
        agenda,
        source_fingerprint,
        unresolved_profiles,
    })
}

fn recap_state_view(app_state: &AppState, session_id: &str) -> Result<RecapStateView, String> {
    let db = app_state.db_handle()?;
    let session = db
        .get_session(session_id)?
        .ok_or_else(|| "Conversation not found".to_string())?;
    let segments = db.list_segments(session_id)?;
    recap_state_view_from(app_state, &db, &session, &segments)
}

fn recap_state_view_from(
    app_state: &AppState,
    db: &Db,
    session: &Session,
    segments: &[SegmentRecord],
) -> Result<RecapStateView, String> {
    let session_id = session.id.as_str();
    let snapshot = recap_snapshot_from(db, session, segments)?;
    let mut recap = db.load_recap(session_id)?;
    if let Some(saved) = recap.as_mut() {
        if saved.source_fingerprint != snapshot.source_fingerprint
            && saved.schema_version != recap::SCHEMA_VERSION
        {
            let config = app_state
                .config
                .lock()
                .map_err(|_| "Configuration lock poisoned".to_string())?
                .clone();
            let agenda_fingerprint = snapshot.agenda.as_ref().map(|agenda| AgendaFingerprint {
                source_kind: &agenda.source_kind,
                filename: &agenda.filename,
                mime_type: &agenda.mime_type,
                content: &agenda.content,
            });
            let legacy_fingerprint = recap::legacy_source_fingerprint(
                &snapshot.segments,
                agenda_fingerprint,
                &config.no_translation_languages,
            )?;
            if saved.source_fingerprint == legacy_fingerprint {
                db.update_recap_source_fingerprint(session_id, &snapshot.source_fingerprint)?;
                saved.source_fingerprint = snapshot.source_fingerprint.clone();
            }
        }
    }
    let stale = recap
        .as_ref()
        .map(|recap| recap.source_fingerprint != snapshot.source_fingerprint)
        .unwrap_or(false);
    let mut custom_recaps = db
        .load_custom_recaps(session_id)?
        .into_iter()
        .map(|saved| CustomRecapStateView {
            stale: saved.source_fingerprint != snapshot.source_fingerprint,
            recap_type_id: saved.recap_type_id,
            name: saved.name_snapshot,
            content_markdown: saved.content_markdown,
            target_language: saved.target_language,
            model: saved.model,
            source_fingerprint: saved.source_fingerprint,
            input_tokens: saved.input_tokens,
            output_tokens: saved.output_tokens,
            generated_at: saved.generated_at,
        })
        .collect::<Vec<_>>();
    custom_recaps.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left.recap_type_id.cmp(&right.recap_type_id))
    });
    let in_flight = app_state
        .recap_in_flight
        .lock()
        .map_err(|_| "Recap lock poisoned".to_string())?
        .contains(session_id);
    Ok(RecapStateView {
        agenda: snapshot.agenda.as_ref().map(AgendaRecord::metadata),
        recap,
        custom_recaps,
        current_fingerprint: snapshot.source_fingerprint,
        stale,
        unresolved_profiles: snapshot.unresolved_profiles,
        in_flight,
    })
}

#[tauri::command]
fn load_conversation(
    session_id: String,
    app_state: State<AppState>,
) -> Result<ConversationPayload, String> {
    let db = app_state.db_handle()?;
    let session = db
        .get_session(&session_id)?
        .ok_or_else(|| "Conversation not found".to_string())?;
    let segments = db.list_segments(&session_id)?;
    let voice_groups = db.list_session_voice_groups(&session_id)?;
    let recap_state = recap_state_view_from(app_state.inner(), &db, &session, &segments)?;
    let imported_artifact = db.load_imported_session_artifact(&session_id)?;
    Ok(ConversationPayload {
        session,
        segments,
        voice_groups,
        recap_state,
        imported_artifact,
    })
}

#[tauri::command]
fn get_recap_state(
    session_id: String,
    app_state: State<AppState>,
) -> Result<RecapStateView, String> {
    recap_state_view(app_state.inner(), &session_id)
}

#[tauri::command]
fn list_recap_types(
    include_prompts: Option<bool>,
    app_state: State<AppState>,
) -> Result<Vec<RecapTypeView>, String> {
    let include_prompts = include_prompts.unwrap_or(false);
    Ok(app_state
        .db_handle()?
        .list_recap_types()?
        .into_iter()
        .map(|value| RecapTypeView::from_record(value, include_prompts))
        .collect())
}

#[tauri::command]
fn list_recap_prompt_variables() -> Vec<RecapPromptVariableDefinition> {
    recap_prompt_variables::recap_prompt_variable_definitions()
}

#[tauri::command]
fn create_recap_type(
    name: String,
    prompt: String,
    app_state: State<AppState>,
) -> Result<RecapTypeView, String> {
    app_state
        .db_handle()?
        .create_recap_type(&name, &prompt)
        .map(|value| RecapTypeView::from_record(value, true))
}

#[tauri::command]
fn update_recap_type(
    recap_type_id: String,
    name: String,
    prompt: String,
    app_state: State<AppState>,
) -> Result<RecapTypeView, String> {
    app_state
        .db_handle()?
        .update_recap_type(&recap_type_id, &name, &prompt)
        .map(|value| RecapTypeView::from_record(value, true))
}

#[tauri::command]
fn delete_recap_type(recap_type_id: String, app_state: State<AppState>) -> Result<(), String> {
    app_state.db_handle()?.delete_recap_type(&recap_type_id)?;
    Ok(())
}

#[tauri::command]
fn restore_recap_type_default(
    recap_type_id: String,
    app_state: State<AppState>,
) -> Result<RecapTypeView, String> {
    app_state
        .db_handle()?
        .restore_recap_type_default(&recap_type_id)
        .map(|value| RecapTypeView::from_record(value, true))
}

#[tauri::command]
fn save_agenda_text(
    session_id: String,
    text: String,
    app_state: State<AppState>,
) -> Result<AgendaMetadata, String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    if text.trim().is_empty() {
        return Err("Paste some agenda text first".into());
    }
    if text.len() >= MAX_AGENDA_BYTES {
        return Err("Agenda text must be smaller than 50 MB".into());
    }
    app_state
        .db_handle()?
        .upsert_agenda(
            &session_id,
            "text",
            "Pasted agenda.txt",
            "text/plain",
            text.as_bytes(),
        )
        .map(|agenda| agenda.metadata())
}

#[tauri::command]
fn choose_agenda_file(
    session_id: String,
    app_state: State<AppState>,
) -> Result<Option<AgendaMetadata>, String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    let path = rfd::FileDialog::new()
        .set_title("Choose a meeting agenda")
        .add_filter(
            "Agenda documents",
            &[
                "pdf", "doc", "docx", "rtf", "odt", "txt", "md", "json", "html", "htm", "xml",
                "ppt", "pptx", "csv", "xls", "xlsx",
            ],
        )
        .pick_file();
    let Some(path) = path else {
        return Ok(None);
    };
    let mime_type = agenda_mime_type(&path).ok_or_else(|| {
        "That agenda file type is not supported. Choose PDF, DOC/DOCX, RTF, ODT, text, HTML/XML, PowerPoint, or a spreadsheet."
            .to_string()
    })?;
    let metadata = std::fs::metadata(&path)
        .map_err(|error| format!("Could not inspect the agenda file: {error}"))?;
    if metadata.len() >= MAX_AGENDA_BYTES as u64 {
        return Err("Agenda files must be smaller than 50 MB".into());
    }
    let content =
        std::fs::read(&path).map_err(|error| format!("Could not read the agenda file: {error}"))?;
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "The agenda filename is not valid Unicode".to_string())?;
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    app_state
        .db_handle()?
        .upsert_agenda(&session_id, "file", filename, mime_type, &content)
        .map(|agenda| Some(agenda.metadata()))
}

#[tauri::command]
fn remove_agenda(session_id: String, app_state: State<AppState>) -> Result<bool, String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    app_state.db_handle()?.delete_agenda(&session_id)
}

fn agenda_mime_type(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("pdf") => Some("application/pdf"),
        Some("doc") => Some("application/msword"),
        Some("docx") => {
            Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }
        Some("rtf") => Some("application/rtf"),
        Some("odt") => Some("application/vnd.oasis.opendocument.text"),
        Some("txt") => Some("text/plain"),
        Some("md") => Some("text/markdown"),
        Some("json") => Some("application/json"),
        Some("html") | Some("htm") => Some("text/html"),
        Some("xml") => Some("application/xml"),
        Some("ppt") => Some("application/vnd.ms-powerpoint"),
        Some("pptx") => {
            Some("application/vnd.openxmlformats-officedocument.presentationml.presentation")
        }
        Some("csv") => Some("text/csv"),
        Some("xls") => Some("application/vnd.ms-excel"),
        Some("xlsx") => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        _ => None,
    }
}

fn jamie_preview_for_path(
    path: &Path,
    supplied_draft: Option<JamieImportDraft>,
    app_state: &AppState,
) -> Result<JamieImportPreview, String> {
    let archive = jamie_import::parse_jamie_export(path)?;
    let db = app_state.db_handle()?;
    let known_people = db.jamie_known_people()?;
    let initial = jamie_import::initial_import_draft(path, &archive, &known_people);
    let draft_path =
        jamie_import::import_draft_path(&app_state.data_dir, &archive.metadata.source_sha256);
    let saved = match supplied_draft {
        Some(draft) => Some(draft),
        None => jamie_import::load_import_draft(&draft_path)?,
    };
    let mut draft = saved
        .map(|saved| jamie_import::merge_saved_draft(initial.clone(), saved))
        .unwrap_or(initial);
    draft.updated_at = chrono::Utc::now();
    jamie_import::save_import_draft(&draft_path, &draft)?;
    let existing = db.imported_meeting_fingerprints("Jamie")?;
    Ok(jamie_import::build_import_preview(
        &archive,
        &draft,
        &known_people,
        &existing,
    ))
}

#[tauri::command]
fn choose_jamie_export() -> Result<Option<String>, String> {
    let path = rfd::FileDialog::new()
        .set_title("Choose a Jamie meeting export")
        .add_filter("Jamie text export", &["txt"])
        .pick_file();
    let Some(path) = path else {
        return Ok(None);
    };
    path.into_os_string()
        .into_string()
        .map(Some)
        .map_err(|_| "The selected Jamie export path is not valid Unicode.".to_string())
}

#[tauri::command]
async fn inspect_jamie_export(
    source_path: String,
    app_state: State<'_, AppState>,
) -> Result<JamieImportPreview, String> {
    let path = PathBuf::from(source_path);
    if !path.is_file() {
        return Err("The selected Jamie export is no longer available.".into());
    }
    let state = app_state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || jamie_preview_for_path(&path, None, &state))
        .await
        .map_err(|error| format!("Jamie import inspection stopped unexpectedly: {error}"))?
}

#[tauri::command]
async fn resume_jamie_import(
    app_state: State<'_, AppState>,
) -> Result<Option<JamieImportPreview>, String> {
    let state = app_state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let directory = state.data_dir.join("imports");
        if !directory.is_dir() {
            return Ok(None);
        }
        let mut candidates = std::fs::read_dir(&directory)
            .map_err(|error| format!("Could not inspect saved import drafts: {error}"))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|value| value.to_str())
                    .map(|value| value.starts_with("jamie-") && value.ends_with(".json"))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|path| {
            std::fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
        });
        let Some(draft_path) = candidates.pop() else {
            return Ok(None);
        };
        let Some(draft) = jamie_import::load_import_draft(&draft_path)? else {
            return Ok(None);
        };
        let source = PathBuf::from(&draft.source_path);
        if !source.is_file() {
            return Err(format!(
                "The source file for this saved Jamie import is no longer available at {}",
                source.display()
            ));
        }
        jamie_preview_for_path(&source, Some(draft), &state).map(Some)
    })
    .await
    .map_err(|error| format!("Jamie import resume stopped unexpectedly: {error}"))?
}

#[tauri::command]
fn save_jamie_import_draft(
    draft: JamieImportDraft,
    app_state: State<AppState>,
) -> Result<(), String> {
    let source = PathBuf::from(&draft.source_path);
    if !source.is_file() {
        return Err("The selected Jamie export is no longer available.".into());
    }
    if draft.source_sha256.len() != 64
        || !draft
            .source_sha256
            .bytes()
            .all(|value| value.is_ascii_hexdigit())
    {
        return Err("The Jamie import draft has an invalid source fingerprint.".into());
    }
    let draft_path = jamie_import::import_draft_path(&app_state.data_dir, &draft.source_sha256);
    jamie_import::save_import_draft(&draft_path, &draft)
}

#[tauri::command]
async fn run_jamie_import(
    draft: JamieImportDraft,
    app_state: State<'_, AppState>,
) -> Result<db::JamieImportResult, String> {
    let state = app_state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let source = PathBuf::from(&draft.source_path);
        let archive = jamie_import::parse_jamie_export(&source)?;
        if archive.metadata.source_sha256 != draft.source_sha256 {
            return Err(
                "The Jamie export changed after review. Reopen it and review the new content."
                    .into(),
            );
        }
        let db = state.db_handle()?;
        let result = db.import_jamie_archive(&archive, &draft)?;
        let draft_path = jamie_import::import_draft_path(&state.data_dir, &draft.source_sha256);
        if draft_path.is_file() {
            std::fs::remove_file(&draft_path)
                .map_err(|error| format!("Import succeeded but draft cleanup failed: {error}"))?;
        }
        Ok(result)
    })
    .await
    .map_err(|error| format!("Jamie import stopped unexpectedly: {error}"))?
}

#[tauri::command]
async fn rollback_jamie_import(
    import_id: String,
    app_state: State<'_, AppState>,
) -> Result<db::JamieRollbackResult, String> {
    let state = app_state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let db = state.db_handle()?;
        let session_ids = db.import_batch_session_ids(&import_id)?;
        ensure_sessions_not_recapping(&state, &session_ids)?;
        db.rollback_import(&import_id)
    })
    .await
    .map_err(|error| format!("Jamie rollback stopped unexpectedly: {error}"))?
}

#[tauri::command]
fn list_import_batches(app_state: State<AppState>) -> Result<Vec<db::ImportBatchSummary>, String> {
    app_state.db_handle()?.list_import_batches()
}

#[tauri::command]
fn get_imported_session_artifact(
    session_id: String,
    app_state: State<AppState>,
) -> Result<Option<db::ImportedSessionArtifact>, String> {
    app_state
        .db_handle()?
        .load_imported_session_artifact(&session_id)
}

fn emit_recap_progress(app_handle: &tauri::AppHandle, session_id: &str, stage: &str, detail: &str) {
    emit_recap_progress_for(app_handle, session_id, None, None, stage, detail);
}

fn emit_recap_progress_for(
    app_handle: &tauri::AppHandle,
    session_id: &str,
    recap_type_id: Option<&str>,
    recap_type_name: Option<&str>,
    stage: &str,
    detail: &str,
) {
    let type_detail = recap_type_name
        .map(|name| format!(" custom={name:?}"))
        .unwrap_or_default();
    eprintln!("[recap {session_id}{type_detail}] {stage}: {detail}");
    let _ = app_handle.emit(
        "recap:progress",
        RecapProgressEvent {
            session_id: session_id.to_string(),
            stage: stage.to_string(),
            detail: detail.to_string(),
            recap_type_id: recap_type_id.map(str::to_string),
            recap_type_name: recap_type_name.map(str::to_string),
        },
    );
}

async fn generate_recap_inner(
    session_id: &str,
    allow_unresolved: bool,
    app_state: &AppState,
    app_handle: &tauri::AppHandle,
) -> Result<(), String> {
    emit_recap_progress(
        app_handle,
        session_id,
        "prepare",
        "Preparing transcript and agenda",
    );
    let db = app_state.db_handle()?;
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned".to_string())?
        .clone();
    let model = config.openai_model.trim().to_string();
    if model.is_empty() {
        return Err("Configure an LLM model in Settings before creating a recap".into());
    }
    let api_key = app_state.load_openai_key()?;
    let snapshot = recap_snapshot(&db, session_id)?;
    let variable_context = recap_prompt_variable_context(&snapshot);
    let standard_prompts = standard_recap_prompts(&db, &variable_context)?;
    if !allow_unresolved && !snapshot.unresolved_profiles.is_empty() {
        return Err(format!(
            "Name or assign {} unresolved voice profile{} before recapping, or explicitly choose Recap anyway",
            snapshot.unresolved_profiles.len(),
            if snapshot.unresolved_profiles.len() == 1 {
                ""
            } else {
                "s"
            }
        ));
    }
    emit_recap_progress(
        app_handle,
        session_id,
        "llm:start",
        "Starting the on-demand LLM recap run",
    );
    let response = openai::generate_recap(
        openai::RecapRequest {
            api_key: &api_key,
            model: &model,
            segments: &snapshot.segments,
            agenda: snapshot.agenda.as_ref(),
            preferred_language: &config.preferred_language,
            no_translation_languages: &config.no_translation_languages,
            standard_prompts: &standard_prompts,
        },
        |stage, detail| emit_recap_progress(app_handle, session_id, stage, detail),
    )
    .await?;
    emit_recap_progress(
        app_handle,
        session_id,
        "llm:done",
        "LLM recap requests complete",
    );
    for warning in &response.warnings {
        emit_recap_progress(app_handle, session_id, "warning", warning);
    }
    emit_recap_progress(
        app_handle,
        session_id,
        "validate",
        "Validating structured recap",
    );
    emit_recap_progress(app_handle, session_id, "save", "Saving recap locally");
    db.save_recap_and_title_if_source_matches(RecapSave {
        session_id,
        title: &response.payload.meeting_title,
        model: &model,
        prompt_version: recap::PROMPT_VERSION,
        schema_version: recap::SCHEMA_VERSION,
        source_fingerprint: &snapshot.source_fingerprint,
        payload: &response.payload,
        input_tokens: response.input_tokens,
        output_tokens: response.output_tokens,
    })?;
    let persisted = db
        .load_recap(session_id)?
        .ok_or_else(|| "The recap save completed but could not be read back".to_string())?;
    if persisted.source_fingerprint != snapshot.source_fingerprint {
        return Err("The saved recap failed its source-integrity check".into());
    }
    emit_recap_progress(app_handle, session_id, "complete", "Recap ready");
    Ok(())
}

fn claim_recap_session(app_state: &AppState, session_id: &str) -> Result<(), String> {
    let maintenance = app_state
        .maintenance_in_flight
        .lock()
        .map_err(|_| "Maintenance lock poisoned".to_string())?;
    if *maintenance {
        return Err(
            "Voice recognition maintenance is running. Start the recap when it finishes.".into(),
        );
    }
    let mut in_flight = app_state
        .recap_in_flight
        .lock()
        .map_err(|_| "Recap lock poisoned".to_string())?;
    let identity_in_flight = app_state
        .identity_in_flight
        .lock()
        .map_err(|_| "Identity lock poisoned".to_string())?;
    if identity_in_flight.contains(session_id) {
        return Err(
            "This conversation's people or voices are being changed. Run the recap after that operation finishes."
                .into(),
        );
    }
    if !in_flight.insert(session_id.to_string()) {
        return Err("A recap is already being generated for this conversation".into());
    }
    Ok(())
}

fn release_recap_session(app_state: &AppState, session_id: &str) {
    if let Ok(mut in_flight) = app_state.recap_in_flight.lock() {
        in_flight.remove(session_id);
    }
}

#[tauri::command]
async fn generate_recap(
    session_id: String,
    allow_unresolved: bool,
    app_state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<RecapStateView, String> {
    let app_state = app_state.inner().clone();
    claim_recap_session(&app_state, &session_id)?;
    let result = generate_recap_inner(&session_id, allow_unresolved, &app_state, &app_handle).await;
    release_recap_session(&app_state, &session_id);
    match result {
        Ok(()) => recap_state_view(&app_state, &session_id),
        Err(error) => {
            emit_recap_progress(&app_handle, &session_id, "error", &error);
            Err(error)
        }
    }
}

async fn generate_custom_recap_inner(
    session_id: &str,
    recap_type: &RecapType,
    allow_unresolved: bool,
    app_state: &AppState,
    app_handle: &tauri::AppHandle,
) -> Result<(), String> {
    let emit = |stage: &str, detail: &str| {
        emit_recap_progress_for(
            app_handle,
            session_id,
            Some(&recap_type.id),
            Some(&recap_type.name),
            stage,
            detail,
        )
    };
    emit("prepare", "Preparing transcript and agenda");
    let db = app_state.db_handle()?;
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned".to_string())?
        .clone();
    let model = config.openai_model.trim().to_string();
    if model.is_empty() {
        return Err("Configure an LLM model in Settings before creating a recap".into());
    }
    let api_key = app_state.load_openai_key()?;
    let snapshot = recap_snapshot(&db, session_id)?;
    let variable_context = recap_prompt_variable_context(&snapshot);
    let expanded_prompt = expand_recap_prompt(&recap_type.prompt, &variable_context);
    if !allow_unresolved && !snapshot.unresolved_profiles.is_empty() {
        return Err(format!(
            "Name or assign {} unresolved voice profile{} before recapping, or explicitly choose Recap anyway",
            snapshot.unresolved_profiles.len(),
            if snapshot.unresolved_profiles.len() == 1 {
                ""
            } else {
                "s"
            }
        ));
    }
    emit("llm:start", "Starting the on-demand custom recap run");
    let response = openai::generate_custom_recap(
        openai::CustomRecapRequest {
            api_key: &api_key,
            model: &model,
            segments: &snapshot.segments,
            agenda: snapshot.agenda.as_ref(),
            preferred_language: &config.preferred_language,
            prompt: &expanded_prompt,
        },
        |stage, detail| emit(stage, detail),
    )
    .await?;
    emit("llm:done", "LLM custom recap request complete");
    emit("validate", "Validating the custom recap and source");
    emit("save", "Saving custom recap locally");
    save_custom_recap_if_source_matches(
        &db,
        session_id,
        recap_type,
        &expanded_prompt,
        &model,
        &snapshot.source_fingerprint,
        &response,
    )?;
    emit("complete", "Custom recap ready");
    Ok(())
}

fn save_custom_recap_if_source_matches(
    db: &Db,
    session_id: &str,
    recap_type: &RecapType,
    expanded_prompt: &str,
    model: &str,
    expected_source_fingerprint: &str,
    response: &openai::CustomRecapResponse,
) -> Result<(), String> {
    db.save_custom_recap_if_source_matches(CustomRecapSave {
        session_id,
        recap_type_id: &recap_type.id,
        name_snapshot: &recap_type.name,
        prompt_snapshot: expanded_prompt,
        content_markdown: &response.content_markdown,
        target_language: &response.target_language,
        model,
        source_fingerprint: expected_source_fingerprint,
        input_tokens: response.input_tokens,
        output_tokens: response.output_tokens,
    })?;
    let persisted = db
        .load_custom_recap(session_id, &recap_type.id)?
        .ok_or_else(|| "The custom recap save completed but could not be read back".to_string())?;
    if persisted.source_fingerprint != expected_source_fingerprint {
        return Err("The saved custom recap failed its source-integrity check".into());
    }
    Ok(())
}

#[tauri::command]
async fn generate_custom_recap(
    session_id: String,
    recap_type_id: String,
    allow_unresolved: bool,
    app_state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<RecapStateView, String> {
    let app_state = app_state.inner().clone();
    let recap_type = app_state
        .db_handle()?
        .load_recap_type(&recap_type_id)?
        .ok_or_else(|| "Recap type not found".to_string())?;
    if recap_type.kind != RECAP_TYPE_KIND_CUSTOM {
        return Err("Only custom recap types can be run from the split menu".into());
    }
    claim_recap_session(&app_state, &session_id)?;
    let result = generate_custom_recap_inner(
        &session_id,
        &recap_type,
        allow_unresolved,
        &app_state,
        &app_handle,
    )
    .await;
    release_recap_session(&app_state, &session_id);
    match result {
        Ok(()) => recap_state_view(&app_state, &session_id),
        Err(error) => {
            emit_recap_progress_for(
                &app_handle,
                &session_id,
                Some(&recap_type.id),
                Some(&recap_type.name),
                "error",
                &error,
            );
            Err(error)
        }
    }
}

#[tauri::command]
fn list_speakers(app_state: State<AppState>) -> Result<Vec<Speaker>, String> {
    app_state.db_handle()?.list_speakers()
}

#[tauri::command]
fn list_speakers_with_stats(app_state: State<AppState>) -> Result<Vec<db::SpeakerStats>, String> {
    app_state.db_handle()?.list_speakers_with_stats()
}

#[tauri::command]
fn list_identity_profiles(
    search: String,
    status: String,
    page: usize,
    page_size: usize,
    app_state: State<AppState>,
) -> Result<db::IdentityProfilePage, String> {
    app_state
        .db_handle()?
        .list_identity_profiles(&search, &status, page, page_size)
}

#[tauri::command]
fn list_unassigned_identities(
    search: String,
    status: String,
    page: usize,
    page_size: usize,
    app_state: State<AppState>,
) -> Result<db::UnassignedIdentityPage, String> {
    app_state
        .db_handle()?
        .list_unassigned_identities(&search, &status, page, page_size)
}

#[tauri::command]
fn preview_identity_consolidation(
    request: db::IdentityConsolidationRequest,
    app_state: State<AppState>,
) -> Result<db::IdentityConsolidationPreview, String> {
    app_state
        .db_handle()?
        .preview_identity_consolidation(&request)
}

#[tauri::command]
async fn consolidate_identities(
    request: db::IdentityConsolidationRequest,
    expected_affected_session_ids: Vec<String>,
    expected_impact_revision: String,
    app_state: State<'_, AppState>,
) -> Result<db::IdentityConsolidationResult, String> {
    let app_state = app_state.inner().clone();
    let db = app_state.db_handle()?;
    let preview = db.preview_identity_consolidation(&request)?;
    if preview.affected_session_ids != expected_affected_session_ids
        || preview.impact_revision != expected_impact_revision
    {
        return Err(
            "The people, voices, recaps, or affected conversations changed after the impact preview. Review the operation again."
                .into(),
        );
    }
    let affected_session_ids = preview.affected_session_ids.clone();
    claim_identity_sessions(&app_state, &affected_session_ids)?;
    let expected_session_ids = expected_affected_session_ids;
    let expected_revision = expected_impact_revision;
    let result = tokio::task::spawn_blocking(move || {
        db.consolidate_identities(&request, &expected_session_ids, &expected_revision)
    })
    .await
    .map_err(|error| format!("People and voices operation stopped unexpectedly: {error}"))
    .and_then(|result| result);
    release_identity_sessions(&app_state, &affected_session_ids);
    result
}

#[tauri::command]
fn list_voice_match_decisions(
    session_id: String,
    app_state: State<AppState>,
) -> Result<Vec<db::VoiceMatchDecision>, String> {
    app_state
        .db_handle()?
        .list_voice_match_decisions(&session_id)
}

#[tauri::command]
fn list_session_ids_for_speaker(
    speaker_id: String,
    app_state: State<AppState>,
) -> Result<Vec<String>, String> {
    app_state
        .db_handle()?
        .session_ids_for_speakers(&[speaker_id.as_str()])
}

#[tauri::command]
fn list_session_ids_for_speakers(
    speaker_ids: Vec<String>,
    app_state: State<AppState>,
) -> Result<Vec<String>, String> {
    let speaker_ids = speaker_ids.iter().map(String::as_str).collect::<Vec<_>>();
    app_state
        .db_handle()?
        .session_ids_for_speakers(&speaker_ids)
}

#[tauri::command]
async fn split_voice_group(
    voice_group_id: String,
    selected_segment_ids: Vec<String>,
    app_state: State<'_, AppState>,
) -> Result<VoiceGroupSplitResult, String> {
    let state = app_state.inner().clone();
    let db = state.db_handle()?;
    let session_id = db
        .voice_group_session_id(&voice_group_id)?
        .ok_or_else(|| "Meeting voice group not found".to_string())?;
    let sessions = vec![session_id];
    claim_identity_sessions(&state, &sessions)?;
    let result = match tauri::async_runtime::spawn_blocking(move || {
        db.split_voice_group(&voice_group_id, &selected_segment_ids)
    })
    .await
    {
        Ok(result) => result,
        Err(error) => Err(format!("Voice split stopped unexpectedly: {error}")),
    };
    release_identity_sessions(&state, &sessions);
    result
}

#[tauri::command]
fn dismiss_voice_group_split(
    voice_group_id: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let session_id = db
        .voice_group_session_id(&voice_group_id)?
        .ok_or_else(|| "Meeting voice group not found".to_string())?;
    let sessions = vec![session_id];
    claim_identity_sessions(app_state.inner(), &sessions)?;
    let result = db.dismiss_voice_group_split(&voice_group_id);
    release_identity_sessions(app_state.inner(), &sessions);
    result
}

fn voice_recognition_reset_blockers(
    manager: &RecordingManager,
    app_state: &AppState,
    db: &Db,
    include_maintenance: bool,
) -> Result<Vec<String>, String> {
    let mut blockers = Vec::new();
    if include_maintenance
        && *app_state
            .maintenance_in_flight
            .lock()
            .map_err(|_| "Maintenance lock poisoned".to_string())?
    {
        blockers.push("Another voice-recognition maintenance operation is running".into());
    }
    if manager.is_recording() {
        blockers.push("A recording is active".into());
    }
    let processing = db.active_processing_job_count()?;
    if processing > 0 {
        blockers.push(format!(
            "{processing} conversation{} still being processed",
            if processing == 1 { " is" } else { "s are" }
        ));
    }
    let recap_count = app_state
        .recap_in_flight
        .lock()
        .map_err(|_| "Recap lock poisoned".to_string())?
        .len();
    if recap_count > 0 {
        blockers.push(format!(
            "{recap_count} recap{} running",
            if recap_count == 1 { " is" } else { "s are" }
        ));
    }
    let identity_count = app_state
        .identity_in_flight
        .lock()
        .map_err(|_| "Identity lock poisoned".to_string())?
        .len();
    if identity_count > 0 {
        blockers.push(format!(
            "People or voices are being changed in {identity_count} conversation{}",
            if identity_count == 1 { "" } else { "s" }
        ));
    }
    Ok(blockers)
}

#[tauri::command]
fn preview_voice_recognition_reset(
    manager: State<RecordingManager>,
    app_state: State<AppState>,
) -> Result<VoiceRecognitionResetReadiness, String> {
    let db = app_state.db_handle()?;
    let preview = db.preview_voice_recognition_reset()?;
    let blockers = voice_recognition_reset_blockers(&manager, app_state.inner(), &db, true)?;
    Ok(VoiceRecognitionResetReadiness {
        preview,
        can_reset: blockers.is_empty(),
        blockers,
    })
}

#[tauri::command]
async fn reset_voice_recognition(
    manager: State<'_, RecordingManager>,
    app_state: State<'_, AppState>,
) -> Result<VoiceRecognitionResetResult, String> {
    let state = app_state.inner().clone();
    {
        let mut maintenance = state
            .maintenance_in_flight
            .lock()
            .map_err(|_| "Maintenance lock poisoned".to_string())?;
        if *maintenance {
            return Err("Voice recognition maintenance is already running".into());
        }
        *maintenance = true;
    }
    let result = match state.db_handle() {
        Err(error) => Err(error),
        Ok(db) => match voice_recognition_reset_blockers(&manager, &state, &db, false) {
            Err(error) => Err(error),
            Ok(blockers) if !blockers.is_empty() => Err(format!(
                "Voice recognition cannot be reset yet: {}.",
                blockers.join("; ")
            )),
            Ok(_) => {
                let reset_state = state.clone();
                match tauri::async_runtime::spawn_blocking(move || {
                    reset_state.db_handle()?.reset_voice_recognition_data()
                })
                .await
                {
                    Ok(result) => result,
                    Err(error) => Err(format!(
                        "Voice recognition reset stopped unexpectedly: {error}"
                    )),
                }
            }
        },
    };
    if let Ok(mut maintenance) = state.maintenance_in_flight.lock() {
        *maintenance = false;
    }
    result
}

#[tauri::command]
fn create_profile_for_unknown_segments(
    session_id: String,
    app_state: State<AppState>,
) -> Result<String, String> {
    let db = app_state.db_handle()?;
    let sessions = vec![session_id.clone()];
    claim_identity_sessions(app_state.inner(), &sessions)?;
    let result = db
        .create_speaker_for_unattributed_segments(&session_id)
        .and_then(|(_, label, _)| {
            refresh_session_transcript(&db, &session_id)?;
            Ok(label)
        });
    release_identity_sessions(app_state.inner(), &sessions);
    result
}

#[tauri::command]
fn rename_speaker(
    speaker_id: String,
    new_label: String,
    app_state: State<AppState>,
) -> Result<db::RenameSpeakerResult, String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[speaker_id.as_str()])?;
    claim_identity_sessions(app_state.inner(), &sessions)?;
    let result = db.rename_speaker(&speaker_id, &new_label);
    release_identity_sessions(app_state.inner(), &sessions);
    result
}

#[tauri::command]
fn delete_speaker(speaker_id: String, app_state: State<AppState>) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[speaker_id.as_str()])?;
    claim_identity_sessions(app_state.inner(), &sessions)?;
    let result = db.delete_speaker(&speaker_id);
    release_identity_sessions(app_state.inner(), &sessions);
    result
}

#[tauri::command]
fn get_speaker_samples(
    speaker_id: String,
    app_state: State<AppState>,
) -> Result<Vec<db::SpeakerSample>, String> {
    app_state.db_handle()?.list_samples(&speaker_id)
}

#[tauri::command]
fn get_voice_group_sample(
    voice_group_id: String,
    app_state: State<AppState>,
) -> Result<Option<db::VoiceGroupSample>, String> {
    app_state
        .db_handle()?
        .get_voice_group_sample(&voice_group_id)
}

#[tauri::command]
fn merge_speakers(
    target_id: String,
    source_id: String,
    replace_embeddings: bool,
    app_state: State<AppState>,
) -> Result<db::SpeakerMergeResult, String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[source_id.as_str(), target_id.as_str()])?;
    claim_identity_sessions(app_state.inner(), &sessions)?;
    let result = db.merge_speakers(&source_id, &target_id, replace_embeddings);
    release_identity_sessions(app_state.inner(), &sessions);
    result
}

#[tauri::command]
fn accept_voice_match_suggestion(
    source_id: String,
    target_id: String,
    app_state: State<AppState>,
) -> Result<db::SuggestionAcceptance, String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[source_id.as_str(), target_id.as_str()])?;
    claim_identity_sessions(app_state.inner(), &sessions)?;
    let result = db.accept_voice_match_suggestion(&source_id, &target_id);
    release_identity_sessions(app_state.inner(), &sessions);
    result
}

fn is_allowed_external_url(url: &str) -> bool {
    ALLOWED_EXTERNAL_URLS.contains(&url)
}

#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    if !is_allowed_external_url(&url) {
        return Err("Recall refused to open an unapproved external URL".into());
    }

    #[cfg(target_os = "macos")]
    {
        Command::new("/usr/bin/open")
            .arg(&url)
            .spawn()
            .map(|_| ())
            .map_err(|error| format!("Could not open the default browser: {error}"))
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("Opening setup links is currently supported only on macOS".into())
    }
}

fn build_tray(app: &mut tauri::App) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, MenuId::new("open"), "Open Recall", true, None::<&str>)?;
    let start = MenuItem::with_id(
        app,
        MenuId::new("start"),
        "Start recording",
        true,
        None::<&str>,
    )?;
    let stop = MenuItem::with_id(
        app,
        MenuId::new("stop"),
        "Stop recording",
        true,
        None::<&str>,
    )?;
    let quit = MenuItem::with_id(app, MenuId::new("quit"), "Quit", true, None::<&str>)?;
    let menu = MenuBuilder::new(app)
        .item(&open)
        .separator()
        .item(&start)
        .item(&stop)
        .separator()
        .item(&quit)
        .build()?;
    let icon = Image::new(&[30, 60, 120, 255], 1, 1);
    TrayIconBuilder::new()
        .icon(icon)
        .menu(&menu)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "open" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "start" => {
                let manager = app.state::<RecordingManager>();
                let state = app.state::<AppState>();
                if let Err(error) = start_recording_impl(&manager, &state, app.clone(), None) {
                    let _ = app.emit("recording:error", error);
                }
            }
            "stop" => {
                let manager = app.state::<RecordingManager>();
                match manager.stop() {
                    Ok(stopped) => {
                        let _ = app.emit("recording:stopped", stopped.clone());
                        let state = app.state::<AppState>().inner().clone();
                        match queue_transcription(
                            stopped.path,
                            stopped.stt_context,
                            state,
                            app.clone(),
                        ) {
                            Ok(queued) => {
                                let _ = app.emit("transcription:queued", queued);
                            }
                            Err(error) => {
                                let _ = app.emit("recording:error", error);
                            }
                        }
                    }
                    Err(error) => {
                        let _ = app.emit("recording:error", error);
                    }
                }
            }
            "quit" => std::process::exit(0),
            _ => {}
        })
        .build(app)?;
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            list_input_devices,
            start_recording,
            stop_recording,
            update_live_context,
            transcribe_file_async,
            retry_processing,
            discard_retained_audio,
            get_progress,
            get_live_transcription,
            save_soniox_key,
            delete_soniox_key,
            soniox_key_status,
            save_openai_key,
            delete_openai_key,
            openai_key_status,
            get_preferences,
            list_translation_languages,
            save_preferences,
            complete_onboarding,
            unlock_db,
            enable_encryption,
            app_status,
            list_sessions,
            search_session_ids,
            load_conversation,
            list_segments,
            update_transcript,
            update_session_title,
            update_segment_text,
            assign_segment_speaker,
            delete_session,
            get_recap_state,
            list_recap_types,
            list_recap_prompt_variables,
            create_recap_type,
            update_recap_type,
            delete_recap_type,
            restore_recap_type_default,
            save_agenda_text,
            choose_agenda_file,
            remove_agenda,
            choose_jamie_export,
            inspect_jamie_export,
            resume_jamie_import,
            save_jamie_import_draft,
            run_jamie_import,
            rollback_jamie_import,
            list_import_batches,
            get_imported_session_artifact,
            generate_recap,
            generate_custom_recap,
            list_speakers,
            list_speakers_with_stats,
            list_identity_profiles,
            list_unassigned_identities,
            preview_identity_consolidation,
            consolidate_identities,
            list_voice_match_decisions,
            list_session_ids_for_speaker,
            list_session_ids_for_speakers,
            split_voice_group,
            dismiss_voice_group_split,
            preview_voice_recognition_reset,
            reset_voice_recognition,
            create_profile_for_unknown_segments,
            rename_speaker,
            delete_speaker,
            get_speaker_samples,
            get_voice_group_sample,
            merge_speakers,
            accept_voice_match_suggestion,
            open_external_url,
        ])
        .manage(RecordingManager::default())
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::env::temp_dir().join("recall"));
            std::fs::create_dir_all(&data_dir).ok();
            let resource_model = app
                .path()
                .resolve("models/spkrec-ecapa-voxceleb.onnx", BaseDirectory::Resource)
                .ok()
                .filter(|path| path.is_file());
            let development_model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../models/spkrec-ecapa-voxceleb.onnx");
            let model_path = resource_model.unwrap_or(development_model);
            let app_state = AppState::new(data_dir, model_path);
            {
                let config = app_state.config.lock().unwrap().clone();
                if !config.encryption_enabled {
                    let _ = app_state.open_db(Crypto::new(None, None));
                }
            }
            app.manage(app_state);
            build_tray(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Recall");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meeting_stt_context_is_normalized_and_validated_per_recording() {
        let context = MeetingSttContext {
            language_hints: vec![" EN-us ".into(), "bn".into(), "en-US".into()],
            expected_speakers: Some(5),
        }
        .normalized()
        .unwrap();

        assert_eq!(context.language_hints, vec!["en", "bn"]);
        assert_eq!(context.expected_speakers, Some(5));
        assert!(MeetingSttContext {
            language_hints: Vec::new(),
            expected_speakers: Some(16),
        }
        .normalized()
        .is_err());
        assert_eq!(
            MeetingSttContext {
                language_hints: vec!["jp".into(), "ja-JP".into()],
                expected_speakers: None,
            }
            .normalized()
            .unwrap()
            .language_hints,
            vec!["ja"]
        );
        assert!(MeetingSttContext {
            language_hints: vec!["not-a-language".into()],
            expected_speakers: None,
        }
        .normalized()
        .is_err());
    }

    #[test]
    fn selected_legacy_session_uses_its_cached_transcript_without_segments() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let session_id = db
            .insert_session(
                "Legacy conversation",
                "Unknown speaker: cached legacy transcript",
                4_000,
            )
            .unwrap();
        let session = db.get_session(&session_id).unwrap().unwrap();

        let snapshot = recap_snapshot_from(&db, &session, &[]).unwrap();

        assert_eq!(snapshot.segments.len(), 1);
        assert_eq!(snapshot.segments[0].id, format!("legacy-{session_id}"));
        assert_eq!(
            snapshot.segments[0].text,
            "Unknown speaker: cached legacy transcript"
        );
        assert_eq!(snapshot.meeting_created_at, session.created_at);
        assert_eq!(snapshot.unresolved_profiles, vec!["Unknown speaker"]);
    }

    #[test]
    fn standard_recap_variables_expand_for_the_run_without_mutating_templates() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let template =
            "Summarize the meeting on {{meeting_datetime}}. Keep {{future_variable}} literal.";
        db.update_recap_type(BUILTIN_EXECUTIVE_SUMMARY_ID, "Executive summary", template)
            .unwrap();
        let context = RecapPromptVariableContext::from_fixed_offset(
            chrono::DateTime::parse_from_rfc3339("2026-09-01T07:30:45Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            chrono::FixedOffset::east_opt(2 * 60 * 60).unwrap(),
        );

        let prompts = standard_recap_prompts(&db, &context).unwrap();

        assert_eq!(
            prompts.executive_summary,
            "Summarize the meeting on 2026/09/01 09:30 UTC+02:00. Keep {{future_variable}} literal."
        );
        assert_eq!(
            db.load_recap_type(BUILTIN_EXECUTIVE_SUMMARY_ID)
                .unwrap()
                .unwrap()
                .prompt,
            template
        );
    }

    #[test]
    fn recap_prompt_context_uses_the_selected_sessions_persisted_timestamp() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let session_id = db
            .insert_session("Daily", "Alice: Status update", 4_000)
            .unwrap();
        let session = db.get_session(&session_id).unwrap().unwrap();
        let snapshot = recap_snapshot(&db, &session_id).unwrap();

        let expanded = expand_recap_prompt(
            "{{meeting_datetime}}",
            &recap_prompt_variable_context(&snapshot),
        );
        let local_timestamp = session.created_at.with_timezone(&chrono::Local);

        assert_eq!(
            expanded,
            format!(
                "{} UTC{}",
                local_timestamp.format("%Y/%m/%d %H:%M"),
                local_timestamp.format("%:z")
            )
        );
    }

    #[test]
    fn an_existing_speaker_without_a_label_still_requires_participant_review() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let session_id = db.insert_session("Meeting", "", 4_000).unwrap();
        let speaker_id = db.insert_speaker(None).unwrap();
        db.insert_segment(
            &session_id,
            0,
            4_000,
            Some(&speaker_id),
            None,
            "Unattributed transcript",
        )
        .unwrap();

        let snapshot = recap_snapshot(&db, &session_id).unwrap();

        assert_eq!(snapshot.unresolved_profiles, vec!["Unknown speaker"]);
    }

    #[test]
    fn changed_custom_recap_source_rejects_replacement_and_preserves_title_and_result() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let session_id = db
            .insert_session("Original title", "Alice: Original transcript", 4_000)
            .unwrap();
        let recap_type = db
            .create_recap_type("Risk review", "Identify material risks")
            .unwrap();
        let original_snapshot = recap_snapshot(&db, &session_id).unwrap();
        db.save_custom_recap(CustomRecapSave {
            session_id: &session_id,
            recap_type_id: &recap_type.id,
            name_snapshot: &recap_type.name,
            prompt_snapshot: &recap_type.prompt,
            content_markdown: "# Previous result",
            target_language: "en",
            model: "test-model",
            source_fingerprint: &original_snapshot.source_fingerprint,
            input_tokens: 10,
            output_tokens: 5,
        })
        .unwrap();
        db.update_session_transcript(&session_id, "Alice: Changed transcript")
            .unwrap();

        let error = save_custom_recap_if_source_matches(
            &db,
            &session_id,
            &recap_type,
            &recap_type.prompt,
            "test-model",
            &original_snapshot.source_fingerprint,
            &openai::CustomRecapResponse {
                target_language: "en".into(),
                content_markdown: "# Replacement result".into(),
                input_tokens: 20,
                output_tokens: 10,
            },
        )
        .unwrap_err();

        assert!(error.contains("changed while the LLM provider was working"));
        let preserved = db
            .load_custom_recap(&session_id, &recap_type.id)
            .unwrap()
            .unwrap();
        assert_eq!(preserved.content_markdown, "# Previous result");
        assert_eq!(
            db.get_session(&session_id).unwrap().unwrap().title,
            "Original title"
        );
    }

    #[test]
    fn custom_recap_snapshot_persists_the_expanded_prompt_without_mutating_the_template() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let session_id = db
            .insert_session("Daily", "Alice: Yesterday I fixed the issue.", 4_000)
            .unwrap();
        let template = "Heading date: {{meeting_date}} at {{meeting_time}}. {{future_variable}}";
        let recap_type = db.create_recap_type("Daily", template).unwrap();
        let source = recap_snapshot(&db, &session_id).unwrap();
        let context = RecapPromptVariableContext::from_fixed_offset(
            chrono::DateTime::parse_from_rfc3339("2026-09-01T07:30:45Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            chrono::FixedOffset::east_opt(2 * 60 * 60).unwrap(),
        );
        let expanded_prompt = expand_recap_prompt(&recap_type.prompt, &context);

        save_custom_recap_if_source_matches(
            &db,
            &session_id,
            &recap_type,
            &expanded_prompt,
            "test-model",
            &source.source_fingerprint,
            &openai::CustomRecapResponse {
                target_language: "en".into(),
                content_markdown: "# Daily".into(),
                input_tokens: 20,
                output_tokens: 10,
            },
        )
        .unwrap();

        let persisted = db
            .load_custom_recap(&session_id, &recap_type.id)
            .unwrap()
            .unwrap();
        assert_eq!(
            persisted.prompt_snapshot,
            "Heading date: 2026/09/01 at 09:30. {{future_variable}}"
        );
        assert_eq!(
            db.load_recap_type(&recap_type.id).unwrap().unwrap().prompt,
            template
        );
    }

    fn write_test_wav(path: &Path, seconds: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..(16_000 * seconds) {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn stored_embedding(
        id: &str,
        speaker_id: &str,
        label: &str,
        vector: Vec<f32>,
    ) -> StoredEmbedding {
        StoredEmbedding {
            id: id.into(),
            speaker_id: speaker_id.into(),
            speaker_label: Some(label.into()),
            vector,
            source_session_id: "x".into(),
            created_at: chrono::Utc::now(),
            model_version: EMBEDDING_VERSION.into(),
        }
    }

    fn embedded_sample_window(
        candidate_batch: usize,
        segment_index: usize,
        vector: Vec<f32>,
    ) -> (SampleWindow, Vec<f32>) {
        (
            SampleWindow {
                start_ms: 0,
                end_ms: SAMPLE_WINDOW_MS,
                segment_index,
                candidate_batch,
                pcm: Vec::new(),
            },
            vector,
        )
    }

    #[test]
    fn setup_links_are_restricted_to_the_documented_provider_and_source_pages() {
        for url in ALLOWED_EXTERNAL_URLS {
            assert!(is_allowed_external_url(url));
        }
        assert!(!is_allowed_external_url("javascript:alert(1)"));
        assert!(!is_allowed_external_url(
            "https://console.soniox.com.evil.example"
        ));
        assert!(!is_allowed_external_url(
            "https://platform.openai.com/api-keys?next=evil"
        ));
    }

    #[test]
    fn completed_recording_is_verified_in_private_recovery_storage() {
        let root = std::env::temp_dir().join(format!("recall-audio-persist-{}", Uuid::new_v4()));
        let source = root.join("temporary.wav");
        fs::create_dir_all(&root).unwrap();
        write_test_wav(&source, 2);
        let state = AppState::new(root.join("data"), root.join("missing-model.onnx"));

        let retained = persist_recording_audio(&source, &state, "session-1").unwrap();

        assert!(!source.exists());
        assert!(retained.is_file());
        assert_eq!(wav_duration_ms(&retained).unwrap(), 2_000);
        assert!(retained.starts_with(state.data_dir.join("processing")));
        remove_managed_audio(&retained, &state).unwrap();
        assert!(!retained.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cleanup_refuses_to_delete_audio_outside_recovery_storage() {
        let root = std::env::temp_dir().join(format!("recall-audio-scope-{}", Uuid::new_v4()));
        let outside = root.join("outside.wav");
        let state = AppState::new(root.join("data"), root.join("missing-model.onnx"));
        fs::create_dir_all(state.data_dir.join("processing")).unwrap();
        write_test_wav(&outside, 1);

        let error = remove_managed_audio(&outside, &state).unwrap_err();

        assert!(error.contains("outside its recovery directory"));
        assert!(outside.is_file());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn startup_surfaces_an_orphaned_recovery_wav_for_retry() {
        let root = std::env::temp_dir().join(format!("recall-orphan-recovery-{}", Uuid::new_v4()));
        let data_dir = root.join("data");
        let processing_dir = data_dir.join("processing");
        let session_id = Uuid::new_v4().to_string();
        let retained = processing_dir.join(format!("{session_id}.wav"));
        fs::create_dir_all(&processing_dir).unwrap();
        write_test_wav(&retained, 3);
        let state = AppState::new(data_dir, root.join("missing-model.onnx"));

        state.open_db(Crypto::new(None, None)).unwrap();

        let sessions = state.db_handle().unwrap().list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, session_id);
        assert_eq!(sessions[0].title, "Recovered recording");
        assert_eq!(sessions[0].processing_status.as_deref(), Some("failed"));
        assert!(sessions[0].recoverable_audio);
        assert_eq!(sessions[0].duration_ms, 3_000);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn onboarding_version_is_explicitly_versioned() {
        assert_eq!(ONBOARDING_VERSION, "1");
        assert_eq!(AppConfig::default().onboarding_version, None);
    }

    #[test]
    fn recap_lock_is_scoped_to_the_conversation_being_processed() {
        let root = std::env::temp_dir().join(format!("recall-recap-lock-{}", Uuid::new_v4()));
        let state = AppState::new(root.clone(), root.join("missing-model.onnx"));
        state.recap_in_flight.lock().unwrap().insert("busy".into());

        assert!(ensure_session_not_recapping(&state, "busy").is_err());
        assert!(ensure_session_not_recapping(&state, "available").is_ok());
    }

    #[test]
    fn identity_lock_blocks_overlapping_recaps_and_not_unrelated_conversations() {
        let root = std::env::temp_dir().join(format!("recall-identity-lock-{}", Uuid::new_v4()));
        let state = AppState::new(root.clone(), root.join("missing-model.onnx"));
        state
            .recap_in_flight
            .lock()
            .unwrap()
            .insert("recapping".into());

        assert!(claim_identity_sessions(&state, &["recapping".into()]).is_err());
        assert!(claim_identity_sessions(&state, &["identity-change".into()]).is_ok());
        assert!(state
            .identity_in_flight
            .lock()
            .unwrap()
            .contains("identity-change"));
        assert!(claim_identity_sessions(&state, &["identity-change".into()]).is_err());
        assert!(claim_identity_sessions(&state, &["unrelated".into()]).is_ok());
        release_identity_sessions(&state, &["identity-change".into(), "unrelated".into()]);
        assert!(state.identity_in_flight.lock().unwrap().is_empty());
    }

    #[test]
    fn adjacent_interventions_from_same_speaker_are_merged() {
        let segments = vec![
            TranscriptSegment {
                speaker: "speaker_1".into(),
                start_ms: 0,
                end_ms: 500,
                text: "Hello".into(),
            },
            TranscriptSegment {
                speaker: "speaker_1".into(),
                start_ms: 600,
                end_ms: 900,
                text: "again".into(),
            },
            TranscriptSegment {
                speaker: "speaker_2".into(),
                start_ms: 1_000,
                end_ms: 1_200,
                text: "Hi".into(),
            },
        ];
        let merged = merge_segments(&segments);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, "Hello again");
    }

    #[test]
    fn strong_identity_match_is_automatic() {
        let known = vec![
            stored_embedding("e1", "s1", "Alice", vec![1.0, 0.0]),
            stored_embedding("e2", "s2", "Bob", vec![0.0, 1.0]),
        ];
        let decision = classify_speaker_match(&[1.0, 0.0], &known);
        assert_eq!(decision.kind, VoiceMatchKind::Automatic);
        assert_eq!(decision.best.unwrap().speaker_id, "s1");
    }

    #[test]
    fn consistent_window_centroid_is_l2_normalized() {
        let centroid =
            average_embeddings(vec![vec![1.0, 0.0], vec![0.8, 0.6], vec![0.8, -0.6]].into_iter());
        let norm = centroid
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(centroid[0] > 0.99);
        assert!(centroid[1].abs() < 1e-6);
    }

    #[test]
    fn ambiguous_known_voice_becomes_a_reviewable_suggestion() {
        let known = vec![
            stored_embedding("e1", "s1", "Michael", vec![0.9555, 0.294_990_7]),
            stored_embedding("e2", "s2", "Dmitrii", vec![0.9403, 0.340_346_5]),
        ];
        let decision = classify_speaker_match(&[1.0, 0.0], &known);
        assert_eq!(decision.kind, VoiceMatchKind::Suggested);
        assert_eq!(decision.best.as_ref().unwrap().speaker_id, "s1");
        assert_eq!(decision.runner_up.as_ref().unwrap().speaker_id, "s2");
    }

    #[test]
    fn below_threshold_voice_stays_new() {
        let known = vec![
            stored_embedding("e1", "s1", "Vasily", vec![0.9318, 0.363_003_3]),
            stored_embedding("e2", "s2", "Michael", vec![0.8863, 0.463_111_5]),
        ];
        assert_eq!(
            classify_speaker_match(&[1.0, 0.0], &known).kind,
            VoiceMatchKind::New
        );
    }

    #[test]
    fn two_agreeing_references_can_make_a_consensus_match() {
        let known = vec![
            stored_embedding("e1", "s1", "Alice", vec![0.96, 0.28]),
            stored_embedding("e2", "s1", "Alice", vec![0.95, -0.312_249_9]),
            stored_embedding("e3", "s2", "Bob", vec![0.93, 0.367_559_5]),
        ];
        let decision = classify_speaker_match(&[1.0, 0.0], &known);
        assert_eq!(decision.kind, VoiceMatchKind::Automatic);
        assert_eq!(decision.best.unwrap().support_count, 2);
    }

    #[test]
    fn voice_samples_use_centered_windows_away_from_intervention_edges() {
        let audio = AudioClip {
            samples: vec![1.0; 5_700],
            sample_rate: 1_000,
        };
        let segments = vec![TranscriptSegment {
            speaker: "speaker_1".into(),
            start_ms: 0,
            end_ms: 5_700,
            text: "A long intervention".into(),
        }];

        let selected = clean_sample_windows(
            &audio,
            &segments,
            "speaker_1",
            &[vad::SpeechInterval {
                start_ms: 0,
                end_ms: 5_700,
            }],
        );

        assert_eq!(selected.windows.len(), 1);
        assert_eq!(selected.windows[0].start_ms, 850);
        assert_eq!(selected.windows[0].end_ms, 4_850);
        assert_eq!(selected.windows[0].pcm.len(), 4_000);
    }

    #[test]
    fn unavailable_embedder_keeps_vad_confirmed_meeting_local_previews() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let session_id = db.insert_session("Unavailable ECAPA", "", 12_000).unwrap();
        let audio = AudioClip {
            samples: vec![0.25; 12_000],
            sample_rate: 1_000,
        };
        let segments = vec![
            TranscriptSegment {
                speaker: "speaker_1".into(),
                start_ms: 0,
                end_ms: 5_000,
                text: "First speaker".into(),
            },
            TranscriptSegment {
                speaker: "unknown".into(),
                start_ms: 5_200,
                end_ms: 5_600,
                text: "Unattributed sound".into(),
            },
            TranscriptSegment {
                speaker: "speaker_2".into(),
                start_ms: 6_000,
                end_ms: 11_000,
                text: "Second speaker".into(),
            },
        ];
        let ordered_speakers = vec!["speaker_1".into(), "unknown".into(), "speaker_2".into()];
        let speech_intervals = vec![
            vad::SpeechInterval {
                start_ms: 0,
                end_ms: 5_000,
            },
            vad::SpeechInterval {
                start_ms: 6_000,
                end_ms: 11_000,
            },
        ];
        let reason =
            "the local ECAPA model was unavailable, so Recall did not create a global voice profile";

        let preview_results = persist_model_unavailable_voice_groups(
            &audio,
            &segments,
            &ordered_speakers,
            &session_id,
            &db,
            Some(&speech_intervals),
        )
        .unwrap();

        assert_eq!(
            preview_results
                .iter()
                .map(|result| result.diarized_speaker.as_str())
                .collect::<Vec<_>>(),
            vec!["speaker_1", "speaker_2"]
        );
        assert!(preview_results.iter().all(|result| result.result.is_ok()));
        let groups = db.list_session_voice_groups(&session_id).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups
                .iter()
                .map(|group| group.provider_speaker_label.as_str())
                .collect::<Vec<_>>(),
            vec!["speaker_1", "speaker_2"]
        );
        assert!(groups.iter().all(|group| {
            group.status == "meeting_local_model_unavailable"
                && group.resulting_speaker_id.is_none()
                && group.has_preview_sample
                && group.model_version.is_none()
        }));
        for group in &groups {
            let preview = db
                .get_voice_group_sample(&group.id)
                .unwrap()
                .expect("each diarized speaker should retain one preview");
            assert_eq!(preview.sample_rate, audio.sample_rate);
            assert!(!preview.sample_b64.is_empty());
        }
        let saved_segments = db.list_segments(&session_id).unwrap();
        assert_eq!(saved_segments.len(), 3);
        assert!(saved_segments
            .iter()
            .all(|segment| segment.speaker_id.is_none()));
        assert_eq!(
            saved_segments[0].speaker_label.as_deref(),
            Some("speaker_1")
        );
        assert!(saved_segments[0].voice_group_id.is_some());
        assert_eq!(
            saved_segments[1].speaker_label.as_deref(),
            Some("Unknown speaker")
        );
        assert!(saved_segments[1].provider_speaker_label.is_none());
        assert!(saved_segments[1].voice_group_id.is_none());
        assert_eq!(
            saved_segments[2].speaker_label.as_deref(),
            Some("speaker_2")
        );
        assert!(saved_segments[2].voice_group_id.is_some());
        assert!(db.list_speakers().unwrap().is_empty());
        assert!(db.list_embeddings(EMBEDDING_VERSION).unwrap().is_empty());
        let decisions = db.list_voice_match_decisions(&session_id).unwrap();
        assert_eq!(decisions.len(), 2);
        assert!(decisions.iter().all(|decision| {
            decision.decision == VoiceMatchKind::Skipped.as_str()
                && decision.resulting_speaker_id.is_none()
                && decision.reason == reason
        }));
    }

    #[test]
    fn unavailable_embedder_does_not_invent_a_preview_without_vad_speech() {
        let db = Db::open(":memory:", Crypto::new(None, None)).unwrap();
        let session_id = db
            .insert_session("No VAD-confirmed speech", "", 5_000)
            .unwrap();
        let audio = AudioClip {
            samples: vec![0.25; 5_000],
            sample_rate: 1_000,
        };
        let segments = vec![TranscriptSegment {
            speaker: "speaker_1".into(),
            start_ms: 0,
            end_ms: 5_000,
            text: "Provider-labelled noise".into(),
        }];
        let ordered_speakers = vec!["speaker_1".into()];

        let preview_results = persist_model_unavailable_voice_groups(
            &audio,
            &segments,
            &ordered_speakers,
            &session_id,
            &db,
            Some(&[]),
        )
        .unwrap();

        assert!(preview_results.is_empty());
        let group = db.list_session_voice_groups(&session_id).unwrap().remove(0);
        assert_eq!(group.status, "meeting_local_model_unavailable");
        assert!(!group.has_preview_sample);
        assert!(db.get_voice_group_sample(&group.id).unwrap().is_none());
        assert!(db.list_speakers().unwrap().is_empty());
    }

    #[test]
    fn bounded_voice_candidate_ranges_cap_a_pathologically_long_intervention() {
        let full_windows = MAX_SAMPLE_WINDOWS_PER_SPEAKER as u64 + 10_000;
        let end_ms = full_windows * SAMPLE_WINDOW_MS;
        let midpoint = end_ms / 2;

        let (ranges, short_intervals) = bounded_centered_sample_ranges(
            &[vad::SpeechInterval {
                start_ms: 0,
                end_ms,
            }],
            midpoint,
        );

        assert_eq!(short_intervals, 0);
        assert_eq!(ranges.len(), MAX_SAMPLE_WINDOWS_PER_SPEAKER);
        assert!(ranges
            .iter()
            .all(|(start, end)| end - start == SAMPLE_WINDOW_MS));
        assert!(ranges.iter().any(|(start, end)| {
            (start + ((end - start) / 2)).abs_diff(midpoint) <= SAMPLE_WINDOW_MS / 2
        }));
    }

    #[test]
    fn first_voice_candidate_round_covers_distinct_interventions() {
        let audio = AudioClip {
            samples: vec![1.0; 56_000],
            sample_rate: 1_000,
        };
        let segments = vec![
            TranscriptSegment {
                speaker: "speaker_1".into(),
                start_ms: 0,
                end_ms: 16_000,
                text: "First long intervention".into(),
            },
            TranscriptSegment {
                speaker: "speaker_1".into(),
                start_ms: 20_000,
                end_ms: 36_000,
                text: "Second long intervention".into(),
            },
            TranscriptSegment {
                speaker: "speaker_1".into(),
                start_ms: 40_000,
                end_ms: 56_000,
                text: "Third long intervention".into(),
            },
        ];

        let selected = clean_sample_windows(
            &audio,
            &segments,
            "speaker_1",
            &[vad::SpeechInterval {
                start_ms: 0,
                end_ms: 56_000,
            }],
        );

        let first_round = selected
            .windows
            .iter()
            .take(segments.len())
            .map(|window| window.segment_index)
            .collect::<HashSet<_>>();
        assert_eq!(first_round.len(), segments.len());
        assert!(selected
            .windows
            .iter()
            .take(segments.len())
            .all(|window| window.candidate_batch == 0));
        assert!(selected.windows.len() <= MAX_SAMPLE_WINDOWS_PER_SPEAKER);
    }

    #[test]
    fn voice_samples_reject_interventions_overlapping_another_provider_speaker() {
        let audio = AudioClip {
            samples: vec![1.0; 7_000],
            sample_rate: 1_000,
        };
        let segments = vec![
            TranscriptSegment {
                speaker: "speaker_1".into(),
                start_ms: 0,
                end_ms: 6_000,
                text: "First".into(),
            },
            TranscriptSegment {
                speaker: "speaker_2".into(),
                start_ms: 3_000,
                end_ms: 5_000,
                text: "Overlap".into(),
            },
        ];

        let selected = clean_sample_windows(
            &audio,
            &segments,
            "speaker_1",
            &[vad::SpeechInterval {
                start_ms: 0,
                end_ms: 7_000,
            }],
        );

        assert!(selected.windows.is_empty());
        assert_eq!(selected.overlapping_segments, 1);
    }

    #[test]
    fn mixed_voice_windows_require_a_consistent_majority() {
        let vectors = vec![vec![1.0, 0.0], vec![0.95, 0.312_249_9], vec![0.0, 1.0]];
        assert_eq!(dominant_consistent_indices(&vectors), vec![0, 1]);
        assert!(dominant_consistent_indices(&[vec![1.0, 0.0], vec![0.0, 1.0]]).is_empty());
    }

    #[test]
    fn later_voice_candidate_batch_can_recover_after_an_inconsistent_first_batch() {
        let mut embedded = Vec::new();
        for index in 0..SAMPLE_WINDOWS_PER_CANDIDATE_BATCH {
            embedded.push(embedded_sample_window(
                0,
                index,
                if index % 2 == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                },
            ));
        }
        for index in 0..SAMPLE_WINDOWS_PER_CANDIDATE_BATCH {
            embedded.push(embedded_sample_window(
                1,
                SAMPLE_WINDOWS_PER_CANDIDATE_BATCH + index,
                if index < 5 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                },
            ));
        }

        let trusted = first_trusted_sample_batch(&embedded).expect("second batch should recover");

        assert_eq!(trusted.batch_index, 1);
        assert_eq!(trusted.candidate_count, SAMPLE_WINDOWS_PER_CANDIDATE_BATCH);
        assert_eq!(trusted.window_indices.len(), 5);
        assert!(trusted.window_indices.iter().all(|index| *index >= 8));
    }

    #[test]
    fn bounded_voice_candidate_batches_never_trust_a_batch_without_a_majority() {
        let mut embedded = Vec::new();
        for batch in 0..MAX_SAMPLE_CANDIDATE_BATCHES {
            for index in 0..SAMPLE_WINDOWS_PER_CANDIDATE_BATCH {
                embedded.push(embedded_sample_window(
                    batch,
                    batch * SAMPLE_WINDOWS_PER_CANDIDATE_BATCH + index,
                    if index % 2 == 0 {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    },
                ));
            }
        }
        embedded.push(embedded_sample_window(
            MAX_SAMPLE_CANDIDATE_BATCHES,
            MAX_SAMPLE_WINDOWS_PER_SPEAKER,
            vec![1.0, 0.0],
        ));

        assert!(first_trusted_sample_batch(&embedded).is_none());
    }

    #[test]
    fn balanced_intervention_clusters_remain_reviewable_when_no_global_majority_exists() {
        let observations = vec![
            InterventionVoiceObservation {
                segment_index: 0,
                start_ms: 0,
                end_ms: 3_000,
                embedding: vec![1.0, 0.0],
                selected_duration_ms: 3_000,
                consistency_score: 1.0,
            },
            InterventionVoiceObservation {
                segment_index: 1,
                start_ms: 4_000,
                end_ms: 7_000,
                embedding: vec![0.999, 0.044_710_2],
                selected_duration_ms: 3_000,
                consistency_score: 1.0,
            },
            InterventionVoiceObservation {
                segment_index: 2,
                start_ms: 8_000,
                end_ms: 11_000,
                embedding: vec![0.0, 1.0],
                selected_duration_ms: 3_000,
                consistency_score: 1.0,
            },
            InterventionVoiceObservation {
                segment_index: 3,
                start_ms: 12_000,
                end_ms: 15_000,
                embedding: vec![0.044_710_2, 0.999],
                selected_duration_ms: 3_000,
                consistency_score: 1.0,
            },
        ];

        let clusters = suggested_split_clusters(&observations).expect("reviewable split");
        assert_eq!(clusters.len(), 2);
        assert!(clusters.contains(&vec![0, 1]));
        assert!(clusters.contains(&vec![2, 3]));
        assert!(dominant_consistent_indices(
            &observations
                .iter()
                .map(|observation| observation.embedding.clone())
                .collect::<Vec<_>>()
        )
        .is_empty());
    }

    #[test]
    fn only_near_identical_clean_voiceprints_coalesce_split_provider_labels() {
        let observations = vec![
            VoiceObservation {
                diarized_speaker: "speaker_1".into(),
                pcm: vec![0.1; 10],
                embedding: vec![1.0, 0.0],
                clean_window_count: 2,
                selected_duration_ms: 8_000,
                consistency_score: 1.0,
            },
            VoiceObservation {
                diarized_speaker: "speaker_3".into(),
                pcm: vec![0.1; 10],
                embedding: vec![0.999, 0.044_710_2],
                clean_window_count: 2,
                selected_duration_ms: 8_000,
                consistency_score: 1.0,
            },
            VoiceObservation {
                diarized_speaker: "speaker_2".into(),
                pcm: vec![0.1; 10],
                embedding: vec![0.0, 1.0],
                clean_window_count: 2,
                selected_duration_ms: 8_000,
                consistency_score: 1.0,
            },
        ];

        let groups = group_voice_observations(&observations);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].observation_indices, vec![0, 1]);
        assert_eq!(groups[1].observation_indices, vec![2]);
    }

    #[test]
    fn provisional_profiles_never_match_automatically() {
        let known = vec![stored_embedding("e1", "s1", "VOICE9", vec![1.0, 0.0])];
        assert_eq!(
            classify_speaker_match(&[1.0, 0.0], &known).kind,
            VoiceMatchKind::New
        );
    }

    #[test]
    fn duplicate_normalized_names_are_quarantined_from_matching() {
        let known = vec![
            stored_embedding("e1", "s1", "Michael Vartanyan", vec![1.0, 0.0]),
            stored_embedding("e2", "s2", " michael  vartanyan ", vec![0.99, 0.1]),
        ];
        let decision = classify_speaker_match(&[1.0, 0.0], &known);
        assert_eq!(decision.kind, VoiceMatchKind::New);
        assert!(decision.reason.contains("duplicate-name"));
    }

    #[test]
    fn one_named_profile_cannot_claim_multiple_diarized_voices() {
        let mut candidates = vec![
            VoiceMatchCandidate {
                kind: VoiceMatchKind::Automatic,
                best: Some(IdentityMatch {
                    speaker_id: "s1".into(),
                    label: "Alice".into(),
                    score: 0.99,
                    support_count: 1,
                }),
                runner_up: None,
                reason: "strong".into(),
            },
            VoiceMatchCandidate {
                kind: VoiceMatchKind::Automatic,
                best: Some(IdentityMatch {
                    speaker_id: "s1".into(),
                    label: "Alice".into(),
                    score: 0.90,
                    support_count: 1,
                }),
                runner_up: None,
                reason: "strong".into(),
            },
        ];
        resolve_unique_profile_matches(&mut candidates);
        assert_eq!(candidates[0].kind, VoiceMatchKind::Automatic);
        assert_eq!(candidates[1].kind, VoiceMatchKind::Suggested);
    }

    #[test]
    fn close_competing_claims_are_all_rejected() {
        let mut candidates = vec![
            VoiceMatchCandidate {
                kind: VoiceMatchKind::Automatic,
                best: Some(IdentityMatch {
                    speaker_id: "s1".into(),
                    label: "Alice".into(),
                    score: 0.99,
                    support_count: 1,
                }),
                runner_up: None,
                reason: "strong".into(),
            },
            VoiceMatchCandidate {
                kind: VoiceMatchKind::Automatic,
                best: Some(IdentityMatch {
                    speaker_id: "s1".into(),
                    label: "Alice".into(),
                    score: 0.95,
                    support_count: 1,
                }),
                runner_up: None,
                reason: "strong".into(),
            },
        ];
        resolve_unique_profile_matches(&mut candidates);
        assert!(candidates
            .iter()
            .all(|candidate| candidate.kind == VoiceMatchKind::Suggested));
    }
}

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
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
use serde::Serialize;
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
mod openai;
mod recap;
mod soniox;
mod state;

use config::AppConfig;
use db::{
    AgendaMetadata, AgendaRecord, Crypto, Db, RecapRecord, RecapSave, SegmentRecord, Session,
    Speaker, StoredEmbedding,
};
use embedding::EMBEDDING_VERSION;
use recap::{AgendaFingerprint, RecapSourceSegment};
use soniox::{LiveAudioMessage, LiveTranscriptEvent, TranscriptSegment};
use state::AppState;

const TARGET_SPEAKER_MS: u64 = 12_000;
const MIN_SPEAKER_MS: u64 = 3_000;
const SAMPLE_EDGE_TRIM_MS: u64 = 350;
const SAMPLE_WINDOW_MS: u64 = 4_000;
const SAMPLE_OVERLAP_TOLERANCE_MS: u64 = 200;
const MAX_SAMPLE_WINDOWS_PER_SPEAKER: usize = 8;
const MIN_SAMPLE_RMS: f32 = 0.002;
const SAMPLE_CONSISTENCY_THRESHOLD: f32 = 0.90;
const SAME_VOICE_SPLIT_THRESHOLD: f32 = 0.97;
const MATCH_THRESHOLD: f32 = 0.94;
const MATCH_MARGIN: f32 = 0.08;
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
struct SpeakerMatch {
    speaker_id: String,
    label: String,
    score: f32,
}

#[derive(Debug)]
struct VoiceObservation {
    diarized_speaker: String,
    pcm: Vec<f32>,
    embedding: Vec<f32>,
    clean_window_count: usize,
}

#[derive(Debug, Clone)]
struct SampleWindow {
    start_ms: u64,
    end_ms: u64,
    pcm: Vec<f32>,
}

#[derive(Debug)]
struct VoiceObservationGroup {
    observation_indices: Vec<usize>,
    embedding: Vec<f32>,
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
}

#[derive(Debug, Serialize)]
struct RecapStateView {
    agenda: Option<AgendaMetadata>,
    recap: Option<RecapRecord>,
    current_fingerprint: String,
    stale: bool,
    unresolved_profiles: Vec<String>,
    in_flight: bool,
}

struct RecapSnapshot {
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
}

#[derive(Default)]
struct RecordingManager {
    current: Mutex<Option<Recorder>>,
}

impl RecordingManager {
    fn start(
        &self,
        requested_device: Option<&str>,
        live: Option<(String, Vec<String>)>,
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

        let live_tx = live.map(|(api_key, language_hints)| {
            let (tx, rx) = tokio_mpsc::unbounded_channel();
            let handle = app_handle.clone();
            tauri::async_runtime::spawn(async move {
                if let Err(error) =
                    soniox::run_realtime(api_key, language_hints, sample_rate, rx, handle.clone())
                        .await
                {
                    soniox::emit_realtime_error(&handle, error);
                }
            });
            tx
        });
        let live_started = live_tx.is_some();

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
                    let live_tx = live_tx.clone();
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
                    let live_tx = live_tx.clone();
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
                    let live_tx = live_tx.clone();
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
            if let Some(live_tx) = live_tx {
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
        });
        Ok(RecordingStarted {
            path: output_for_result.to_string_lossy().to_string(),
            device_name,
            sample_rate,
            live_started,
        })
    }

    fn stop(&self) -> Result<PathBuf, String> {
        let mut guard = self.current.lock().map_err(|_| "Recording lock poisoned")?;
        let mut recorder = guard
            .take()
            .ok_or_else(|| "There is no active recording".to_string())?;
        if let Some(tx) = recorder.stop_tx.take() {
            let _ = tx.send(());
        }
        recorder
            .handle
            .take()
            .ok_or_else(|| "Recording worker is missing".to_string())?
            .join()
            .map_err(|_| "Recording worker stopped unexpectedly".to_string())?
    }

    fn is_recording(&self) -> bool {
        self.current
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
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
    let api_key = app_state.load_soniox_key()?;
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?
        .clone();
    let requested = input_device.or(config.selected_input_device.clone());
    app_state.reset_live_transcript(config.live_transcription)?;
    let live = config
        .live_transcription
        .then_some((api_key, config.language_hints.clone()));
    let started = manager.start(requested.as_deref(), live, app_handle.clone())?;
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
) -> Result<String, String> {
    let path = manager.stop()?;
    let path_string = path.to_string_lossy().to_string();
    let _ = app_handle.emit("recording:stopped", path_string.clone());
    Ok(path_string)
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
    let language_hints = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?
        .language_hints
        .clone();
    let result = soniox::transcribe_file(
        Path::new(path),
        &api_key,
        &language_hints,
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
            app_handle,
            run_id,
        )?;
    } else {
        process_segments(&audio, &segments, session_id, &db, None, app_handle, run_id)?;
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
    state: AppState,
    app_handle: tauri::AppHandle,
    run_id: String,
) {
    tauri::async_runtime::spawn_blocking(move || {
        let result = transcribe_file_inner(
            &path,
            &session_id,
            &draft_transcript,
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
    state: AppState,
    app_handle: tauri::AppHandle,
) -> Result<QueuedTranscription, String> {
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
            if let Err(database_error) = db.create_processing_session(
                &session_id,
                &run_id,
                &title,
                &draft_transcript,
                0,
                &expected_path.to_string_lossy(),
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
    if let Err(error) = db.create_processing_session(
        &session_id,
        &run_id,
        &title,
        &draft_transcript,
        duration_ms,
        &retained_path.to_string_lossy(),
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
    spawn_transcription_worker(
        retained_path.to_string_lossy().to_string(),
        session_id.clone(),
        draft_transcript,
        state,
        app_handle,
        run_id.clone(),
    );
    Ok(QueuedTranscription { run_id, session_id })
}

#[tauri::command]
fn transcribe_file_async(
    path: String,
    app_state: State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<QueuedTranscription, String> {
    queue_transcription(path, app_state.inner().clone(), app_handle)
}

#[tauri::command]
fn retry_processing(
    session_id: String,
    app_state: State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<QueuedTranscription, String> {
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
        .list_sessions()?
        .into_iter()
        .find(|candidate| candidate.id == session_id)
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
    spawn_transcription_worker(
        job.audio_path,
        session_id.clone(),
        session.transcript,
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

fn ensure_session_not_recapping(app_state: &AppState, session_id: &str) -> Result<(), String> {
    ensure_sessions_not_recapping(app_state, &[session_id.to_string()])
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
    silent_windows: usize,
}

fn pcm_rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    (pcm.iter().map(|sample| sample * sample).sum::<f32>() / pcm.len() as f32).sqrt()
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

fn clean_sample_windows(
    audio: &AudioClip,
    segments: &[TranscriptSegment],
    diarized_speaker: &str,
) -> SampleWindowSet {
    let mut speaker_segments = segments
        .iter()
        .filter(|segment| segment.speaker == diarized_speaker && segment.end_ms > segment.start_ms)
        .collect::<Vec<_>>();
    speaker_segments
        .sort_by(|left, right| (right.end_ms - right.start_ms).cmp(&(left.end_ms - left.start_ms)));
    let mut result = SampleWindowSet::default();

    for segment in speaker_segments {
        if result.windows.len() >= MAX_SAMPLE_WINDOWS_PER_SPEAKER {
            break;
        }
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

        let full_windows = safe_duration / SAMPLE_WINDOW_MS;
        let mut ranges = if full_windows == 0 {
            vec![(safe_start, safe_end)]
        } else {
            let used = full_windows * SAMPLE_WINDOW_MS;
            let offset = (safe_duration - used) / 2;
            (0..full_windows)
                .map(|index| {
                    let start = safe_start + offset + (index * SAMPLE_WINDOW_MS);
                    (start, start + SAMPLE_WINDOW_MS)
                })
                .collect::<Vec<_>>()
        };
        let segment_midpoint = segment.start_ms + (duration_ms / 2);
        ranges.sort_by_key(|(start, end)| (start + ((end - start) / 2)).abs_diff(segment_midpoint));

        for (start_ms, end_ms) in ranges {
            if result.windows.len() >= MAX_SAMPLE_WINDOWS_PER_SPEAKER {
                break;
            }
            let Some(pcm) = sample_range(audio, start_ms, end_ms) else {
                continue;
            };
            if pcm_rms(&pcm) < MIN_SAMPLE_RMS {
                result.silent_windows += 1;
                continue;
            }
            result.windows.push(SampleWindow {
                start_ms,
                end_ms,
                pcm,
            });
        }
    }
    result
}

fn dominant_consistent_indices(vectors: &[Vec<f32>]) -> Vec<usize> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let mut best_members = Vec::new();
    let mut best_similarity = f32::NEG_INFINITY;
    for (index, vector) in vectors.iter().enumerate() {
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
        if index + 1 == vectors.len()
            && vectors.len() > 1
            && best_members.len() * 2 <= vectors.len()
        {
            return Vec::new();
        }
    }
    best_members
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

fn group_voice_observations(observations: &[VoiceObservation]) -> Vec<VoiceObservationGroup> {
    let mut groups: Vec<VoiceObservationGroup> = Vec::new();
    for (index, observation) in observations.iter().enumerate() {
        let compatible_group = groups.iter().position(|group| {
            group.observation_indices.iter().all(|member| {
                embedding::cosine_similarity(
                    &observation.embedding,
                    &observations[*member].embedding,
                ) >= SAME_VOICE_SPLIT_THRESHOLD
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

fn process_segments(
    audio: &AudioClip,
    segments: &[TranscriptSegment],
    session_id: &str,
    db: &Db,
    embedder: Option<&embedding::Embedder>,
    app_handle: &tauri::AppHandle,
    run_id: &str,
) -> Result<(), String> {
    let known = db.list_embeddings(EMBEDDING_VERSION)?;
    let mut ordered_speakers = Vec::new();
    let mut seen = HashSet::new();
    for segment in segments {
        if seen.insert(segment.speaker.clone()) {
            ordered_speakers.push(segment.speaker.clone());
        }
    }
    let mut observations = Vec::new();
    let mut fallback_previews: HashMap<String, Vec<f32>> = HashMap::new();

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
        let window_set = clean_sample_windows(audio, segments, diarized_speaker);
        if window_set.windows.is_empty() {
            emit_progress(
                app_handle,
                "voiceprint:skipped",
                Some(format!(
                    "{diarized_speaker}: no clean central excerpt of at least {:.1} seconds; keeping the provider voice for manual review ({} overlapping, {} short, {} silent candidates)",
                    MIN_SPEAKER_MS as f64 / 1_000.0,
                    window_set.overlapping_segments,
                    window_set.short_segments,
                    window_set.silent_windows,
                )),
                Some(run_id),
            );
            continue;
        }
        if let Some(window) = window_set.windows.first() {
            fallback_previews.insert(diarized_speaker.clone(), window.pcm.clone());
        }
        let Some(embedder) = embedder else {
            emit_progress(
                app_handle,
                "voiceprint:warning",
                Some(format!(
                    "{diarized_speaker}: local ECAPA model is unavailable; keeping the provider voice for manual review"
                )),
                Some(run_id),
            );
            continue;
        };

        let mut embedded_windows = Vec::new();
        for window in window_set.windows {
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
        let vectors = embedded_windows
            .iter()
            .map(|(_, embedding)| embedding.clone())
            .collect::<Vec<_>>();
        let consistent_indices = dominant_consistent_indices(&vectors);
        if consistent_indices.is_empty() {
            emit_progress(
                app_handle,
                "voiceprint:skipped",
                Some(format!(
                    "{diarized_speaker}: candidate excerpts were not acoustically consistent; keeping the provider voice for manual review without a trusted voiceprint"
                )),
                Some(run_id),
            );
            continue;
        }

        let target_samples = ((audio.sample_rate as u64 * TARGET_SPEAKER_MS) / 1_000) as usize;
        let mut pcm = Vec::with_capacity(target_samples);
        let mut selected_windows = 0usize;
        let mut selected_ms = 0u64;
        for index in consistent_indices.iter().copied() {
            let window = &embedded_windows[index].0;
            let remaining = target_samples.saturating_sub(pcm.len());
            if remaining == 0 {
                break;
            }
            pcm.extend_from_slice(&window.pcm[..remaining.min(window.pcm.len())]);
            selected_windows += 1;
            selected_ms += window.end_ms.saturating_sub(window.start_ms);
        }
        let sample_duration_ms = if audio.sample_rate == 0 {
            0
        } else {
            (pcm.len() as u64 * 1_000) / audio.sample_rate as u64
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
        let embedding = if selected_windows == 1 {
            embedded_windows[consistent_indices[0]].1.clone()
        } else {
            match embedder.embed(&pcm, audio.sample_rate) {
                Ok(value) => value,
                Err(error) => {
                    emit_progress(
                        app_handle,
                        "voiceprint:warning",
                        Some(format!("{diarized_speaker}: {error}")),
                        Some(run_id),
                    );
                    continue;
                }
            }
        };
        let consistency_rejections = embedded_windows.len() - consistent_indices.len();
        emit_progress(
            app_handle,
            "voiceprint:sample:selected",
            Some(format!(
                "{diarized_speaker}: selected {selected_windows} clean central excerpt{} ({:.1}s); rejected {} inconsistent, {} overlapping, {} short, and {} silent candidate{}",
                if selected_windows == 1 { "" } else { "s" },
                sample_duration_ms.min(selected_ms) as f64 / 1_000.0,
                consistency_rejections,
                window_set.overlapping_segments,
                window_set.short_segments,
                window_set.silent_windows,
                if consistency_rejections
                    + window_set.overlapping_segments
                    + window_set.short_segments
                    + window_set.silent_windows
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
            pcm,
            embedding,
            clean_window_count: selected_windows,
        });
    }

    let groups = group_voice_observations(&observations);
    let candidates = groups
        .iter()
        .map(|group| best_speaker_match(&group.embedding, &known))
        .collect::<Vec<_>>();
    let resolved_matches = resolve_unique_profile_matches(&candidates);
    let mut mapping: HashMap<String, (String, String)> = HashMap::new();

    for (group, (candidate, matched)) in groups
        .into_iter()
        .zip(candidates.into_iter().zip(resolved_matches))
    {
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
        let (speaker_id, label, is_new) = if let Some(matched) = matched {
            emit_progress(
                app_handle,
                "voiceprint:matched",
                Some(format!(
                    "{diarized_label} → {} ({:.2}; reference left unchanged)",
                    matched.label, matched.score
                )),
                Some(run_id),
            );
            (matched.speaker_id, matched.label, false)
        } else {
            let label = db.next_voice_label()?;
            let speaker_id = db.insert_speaker(Some(&label))?;
            let reason = candidate
                .map(|candidate| {
                    format!(
                        "the {:.2} claim for {} was not unique within this recording",
                        candidate.score, candidate.label
                    )
                })
                .unwrap_or_else(|| {
                    format!("no named profile reached {MATCH_THRESHOLD:.2} unambiguously")
                });
            emit_progress(
                app_handle,
                "voiceprint:new",
                Some(format!("{diarized_label} → {label} ({reason})")),
                Some(run_id),
            );
            (speaker_id, label, true)
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
        for diarized_speaker in diarized_speakers {
            mapping.insert(diarized_speaker, (speaker_id.clone(), label.clone()));
        }
    }

    for diarized_speaker in ordered_speakers {
        if diarized_speaker == "unknown" || mapping.contains_key(&diarized_speaker) {
            continue;
        }
        let label = db.next_voice_label()?;
        let speaker_id = db.insert_speaker(Some(&label))?;
        if let Some(pcm) = fallback_previews.get(&diarized_speaker) {
            let sample = encode_wav_base64(pcm, audio.sample_rate)?;
            db.insert_sample(&speaker_id, &sample, audio.sample_rate)?;
            emit_progress(
                app_handle,
                "voiceprint:sample:stored",
                Some(format!(
                    "Stored a temporary preview for {label}; no trusted voiceprint was created"
                )),
                Some(run_id),
            );
        }
        emit_progress(
            app_handle,
            "voiceprint:new",
            Some(format!(
                "{diarized_speaker} → {label} for manual review; no safe automatic match was available"
            )),
            Some(run_id),
        );
        mapping.insert(diarized_speaker, (speaker_id, label));
    }

    for segment in segments {
        let mapped = mapping.get(&segment.speaker);
        db.insert_segment(
            session_id,
            segment.start_ms as i64,
            segment.end_ms as i64,
            mapped.map(|value| value.0.as_str()),
            mapped
                .map(|value| value.1.as_str())
                .or(Some("Unknown speaker")),
            segment.text.trim(),
        )?;
    }
    Ok(())
}

fn is_provisional_label(label: &str) -> bool {
    label
        .strip_prefix("VOICE")
        .map(|suffix| !suffix.is_empty() && suffix.chars().all(|value| value.is_ascii_digit()))
        .unwrap_or(false)
}

fn best_speaker_match(query: &[f32], known: &[StoredEmbedding]) -> Option<SpeakerMatch> {
    let mut by_speaker: HashMap<&str, (&StoredEmbedding, f32)> = HashMap::new();
    for candidate in known {
        let Some(label) = candidate.speaker_label.as_deref() else {
            continue;
        };
        if is_provisional_label(label) {
            continue;
        }
        let score = embedding::cosine_similarity(query, &candidate.vector);
        let entry = by_speaker
            .entry(candidate.speaker_id.as_str())
            .or_insert((candidate, score));
        if score > entry.1 {
            *entry = (candidate, score);
        }
    }
    let mut ranked = by_speaker.into_values().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let (best, score) = ranked.first().copied()?;
    let second = ranked.get(1).map(|value| value.1).unwrap_or(-1.0);
    if score < MATCH_THRESHOLD || score - second < MATCH_MARGIN {
        return None;
    }
    Some(SpeakerMatch {
        speaker_id: best.speaker_id.clone(),
        label: best
            .speaker_label
            .clone()
            .unwrap_or_else(|| "Unnamed speaker".into()),
        score,
    })
}

fn resolve_unique_profile_matches(
    candidates: &[Option<SpeakerMatch>],
) -> Vec<Option<SpeakerMatch>> {
    let mut claims: HashMap<&str, Vec<(usize, f32)>> = HashMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if let Some(candidate) = candidate {
            claims
                .entry(candidate.speaker_id.as_str())
                .or_default()
                .push((index, candidate.score));
        }
    }

    let mut accepted = HashSet::new();
    for mut profile_claims in claims.into_values() {
        profile_claims.sort_by(|left, right| right.1.total_cmp(&left.1));
        let (best_index, best_score) = profile_claims[0];
        let runner_up = profile_claims.get(1).map(|value| value.1);
        if runner_up
            .map(|score| best_score - score >= PROFILE_CLAIM_MARGIN)
            .unwrap_or(true)
        {
            accepted.insert(best_index);
        }
    }

    candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            if accepted.contains(&index) {
                candidate.clone()
            } else {
                None
            }
        })
        .collect()
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
fn save_preferences(
    selected_input_device: Option<String>,
    language_hints: Vec<String>,
    live_transcription: bool,
    openai_model: String,
    no_translation_languages: Vec<String>,
    app_state: State<AppState>,
) -> Result<(), String> {
    let openai_model = openai_model.trim();
    if openai_model.is_empty() {
        return Err("LLM model cannot be empty".into());
    }
    let normalized_hints = language_hints
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let mut excluded_languages = no_translation_languages
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty() && value != "en")
        .collect::<Vec<_>>();
    excluded_languages.sort();
    excluded_languages.dedup();
    let mut config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?;
    config.selected_input_device = selected_input_device.filter(|value| !value.trim().is_empty());
    config.language_hints = normalized_hints;
    config.live_transcription = live_transcription;
    config.openai_model = openai_model.to_string();
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
    })
}

#[tauri::command]
fn list_sessions(app_state: State<AppState>) -> Result<Vec<Session>, String> {
    app_state.db_handle()?.list_sessions()
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
    db.update_segment_text(&segment_id, &text)?;
    refresh_session_transcript(&db, &session_id)
}

#[tauri::command]
fn assign_segment_speaker(
    segment_id: String,
    session_id: String,
    speaker_id: Option<String>,
    app_state: State<AppState>,
) -> Result<(), String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    let db = app_state.db_handle()?;
    db.assign_segment_speaker(&segment_id, speaker_id.as_deref())?;
    refresh_session_transcript(&db, &session_id)
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

fn recap_snapshot(db: &Db, config: &AppConfig, session_id: &str) -> Result<RecapSnapshot, String> {
    let session = db
        .list_sessions()?
        .into_iter()
        .find(|session| session.id == session_id)
        .ok_or_else(|| "Conversation not found".to_string())?;
    let stored_segments = db.list_segments(session_id)?;
    let mut segments = stored_segments
        .into_iter()
        .filter(|segment| !segment.text.trim().is_empty())
        .map(|segment| RecapSourceSegment {
            id: segment.id,
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            speaker_id: segment.speaker_id,
            speaker_label: segment
                .speaker_label
                .filter(|label| !label.trim().is_empty())
                .unwrap_or_else(|| "Unknown speaker".to_string()),
            text: segment.text,
        })
        .collect::<Vec<_>>();
    if segments.is_empty() && !session.transcript.trim().is_empty() {
        segments.push(RecapSourceSegment {
            id: format!("legacy-{session_id}"),
            start_ms: 0,
            end_ms: session.duration_ms,
            speaker_id: None,
            speaker_label: "Unknown speaker".into(),
            text: session.transcript,
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
                || is_provisional_label(&segment.speaker_label);
            (unresolved && seen_unresolved.insert(segment.speaker_label.clone()))
                .then(|| segment.speaker_label.clone())
        })
        .collect::<Vec<_>>();
    let agenda = db.load_agenda(session_id)?;
    let agenda_fingerprint = agenda.as_ref().map(|agenda| AgendaFingerprint {
        source_kind: &agenda.source_kind,
        filename: &agenda.filename,
        mime_type: &agenda.mime_type,
        content: &agenda.content,
    });
    let source_fingerprint = recap::source_fingerprint(
        &segments,
        agenda_fingerprint,
        &config.no_translation_languages,
    )?;
    Ok(RecapSnapshot {
        segments,
        agenda,
        source_fingerprint,
        unresolved_profiles,
    })
}

fn recap_state_view(app_state: &AppState, session_id: &str) -> Result<RecapStateView, String> {
    let db = app_state.db_handle()?;
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned".to_string())?
        .clone();
    let snapshot = recap_snapshot(&db, &config, session_id)?;
    let recap = db.load_recap(session_id)?;
    let stale = recap
        .as_ref()
        .map(|recap| recap.source_fingerprint != snapshot.source_fingerprint)
        .unwrap_or(false);
    let in_flight = app_state
        .recap_in_flight
        .lock()
        .map_err(|_| "Recap lock poisoned".to_string())?
        .contains(session_id);
    Ok(RecapStateView {
        agenda: snapshot.agenda.as_ref().map(AgendaRecord::metadata),
        recap,
        current_fingerprint: snapshot.source_fingerprint,
        stale,
        unresolved_profiles: snapshot.unresolved_profiles,
        in_flight,
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

fn emit_recap_progress(app_handle: &tauri::AppHandle, session_id: &str, stage: &str, detail: &str) {
    eprintln!("[recap {session_id}] {stage}: {detail}");
    let _ = app_handle.emit(
        "recap:progress",
        RecapProgressEvent {
            session_id: session_id.to_string(),
            stage: stage.to_string(),
            detail: detail.to_string(),
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
    let snapshot = recap_snapshot(&db, &config, session_id)?;
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
            no_translation_languages: &config.no_translation_languages,
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
    let current_config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned".to_string())?
        .clone();
    let current_snapshot = recap_snapshot(&db, &current_config, session_id)?;
    if current_snapshot.source_fingerprint != snapshot.source_fingerprint {
        return Err(
            "The transcript, speakers, agenda, or translation policy changed while the LLM provider was working. Nothing was replaced; run Recap again."
                .into(),
        );
    }
    emit_recap_progress(app_handle, session_id, "save", "Saving recap locally");
    db.save_recap_and_title(RecapSave {
        session_id,
        title: &response.payload.meeting_title_english,
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

#[tauri::command]
async fn generate_recap(
    session_id: String,
    allow_unresolved: bool,
    app_state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<RecapStateView, String> {
    let app_state = app_state.inner().clone();
    {
        let mut in_flight = app_state
            .recap_in_flight
            .lock()
            .map_err(|_| "Recap lock poisoned".to_string())?;
        if !in_flight.insert(session_id.clone()) {
            return Err("A recap is already being generated for this conversation".into());
        }
    }
    let result = generate_recap_inner(&session_id, allow_unresolved, &app_state, &app_handle).await;
    if let Ok(mut in_flight) = app_state.recap_in_flight.lock() {
        in_flight.remove(&session_id);
    }
    match result {
        Ok(()) => recap_state_view(&app_state, &session_id),
        Err(error) => {
            emit_recap_progress(&app_handle, &session_id, "error", &error);
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
fn create_profile_for_unknown_segments(
    session_id: String,
    app_state: State<AppState>,
) -> Result<String, String> {
    ensure_session_not_recapping(app_state.inner(), &session_id)?;
    let db = app_state.db_handle()?;
    let (_, label, _) = db.create_speaker_for_unattributed_segments(&session_id)?;
    refresh_session_transcript(&db, &session_id)?;
    Ok(label)
}

#[tauri::command]
fn rename_speaker(
    speaker_id: String,
    new_label: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[speaker_id.as_str()])?;
    ensure_sessions_not_recapping(app_state.inner(), &sessions)?;
    db.rename_speaker(&speaker_id, &new_label)?;
    for session_id in sessions {
        refresh_session_transcript(&db, &session_id)?;
    }
    Ok(())
}

#[tauri::command]
fn delete_speaker(speaker_id: String, app_state: State<AppState>) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[speaker_id.as_str()])?;
    ensure_sessions_not_recapping(app_state.inner(), &sessions)?;
    db.delete_speaker(&speaker_id)?;
    for session_id in sessions {
        refresh_session_transcript(&db, &session_id)?;
    }
    Ok(())
}

#[tauri::command]
fn get_speaker_samples(
    speaker_id: String,
    app_state: State<AppState>,
) -> Result<Vec<db::SpeakerSample>, String> {
    app_state.db_handle()?.list_samples(&speaker_id)
}

#[tauri::command]
fn merge_speakers(
    target_id: String,
    source_id: String,
    replace_embeddings: bool,
    app_state: State<AppState>,
) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[source_id.as_str(), target_id.as_str()])?;
    ensure_sessions_not_recapping(app_state.inner(), &sessions)?;
    db.merge_speakers(&source_id, &target_id, replace_embeddings)?;
    for session_id in sessions {
        refresh_session_transcript(&db, &session_id)?;
    }
    Ok(())
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
                    Ok(path) => {
                        let path = path.to_string_lossy().to_string();
                        let _ = app.emit("recording:stopped", path.clone());
                        let state = app.state::<AppState>().inner().clone();
                        match queue_transcription(path, state, app.clone()) {
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
            save_preferences,
            complete_onboarding,
            unlock_db,
            enable_encryption,
            app_status,
            list_sessions,
            list_segments,
            update_transcript,
            update_session_title,
            update_segment_text,
            assign_segment_speaker,
            delete_session,
            get_recap_state,
            save_agenda_text,
            choose_agenda_file,
            remove_agenda,
            generate_recap,
            list_speakers,
            list_speakers_with_stats,
            list_session_ids_for_speaker,
            list_session_ids_for_speakers,
            create_profile_for_unknown_segments,
            rename_speaker,
            delete_speaker,
            get_speaker_samples,
            merge_speakers,
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
    fn match_requires_threshold_and_margin() {
        let known = vec![
            stored_embedding("e1", "s1", "Alice", vec![1.0, 0.0]),
            stored_embedding("e2", "s2", "Bob", vec![0.0, 1.0]),
        ];
        let matched = best_speaker_match(&[1.0, 0.0], &known).unwrap();
        assert_eq!(matched.speaker_id, "s1");
        assert!(best_speaker_match(&[0.71, 0.70], &known).is_none());
    }

    #[test]
    fn strict_ecapa_threshold_accepts_high_confidence_repeat_voice() {
        let known = vec![stored_embedding("e1", "s1", "Alice", vec![1.0, 0.0])];
        assert!(best_speaker_match(&[0.95, 0.312_249_9], &known).is_some());
        assert!(best_speaker_match(&[0.93, 0.367_559_5], &known).is_none());
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

        let selected = clean_sample_windows(&audio, &segments, "speaker_1");

        assert_eq!(selected.windows.len(), 1);
        assert_eq!(selected.windows[0].start_ms, 850);
        assert_eq!(selected.windows[0].end_ms, 4_850);
        assert_eq!(selected.windows[0].pcm.len(), 4_000);
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

        let selected = clean_sample_windows(&audio, &segments, "speaker_1");

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
    fn only_near_identical_clean_voiceprints_coalesce_split_provider_labels() {
        let observations = vec![
            VoiceObservation {
                diarized_speaker: "speaker_1".into(),
                pcm: vec![0.1; 10],
                embedding: vec![1.0, 0.0],
                clean_window_count: 1,
            },
            VoiceObservation {
                diarized_speaker: "speaker_3".into(),
                pcm: vec![0.1; 10],
                embedding: vec![0.98, 0.198_997_5],
                clean_window_count: 1,
            },
            VoiceObservation {
                diarized_speaker: "speaker_2".into(),
                pcm: vec![0.1; 10],
                embedding: vec![0.0, 1.0],
                clean_window_count: 1,
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
        assert!(best_speaker_match(&[1.0, 0.0], &known).is_none());
    }

    #[test]
    fn one_named_profile_cannot_claim_multiple_diarized_voices() {
        let candidates = vec![
            Some(SpeakerMatch {
                speaker_id: "s1".into(),
                label: "Alice".into(),
                score: 0.96,
            }),
            Some(SpeakerMatch {
                speaker_id: "s1".into(),
                label: "Alice".into(),
                score: 0.88,
            }),
        ];
        let resolved = resolve_unique_profile_matches(&candidates);
        assert!(resolved[0].is_some());
        assert!(resolved[1].is_none());
    }

    #[test]
    fn close_competing_claims_are_all_rejected() {
        let candidates = vec![
            Some(SpeakerMatch {
                speaker_id: "s1".into(),
                label: "Alice".into(),
                score: 0.95,
            }),
            Some(SpeakerMatch {
                speaker_id: "s1".into(),
                label: "Alice".into(),
                score: 0.92,
            }),
        ];
        assert!(resolve_unique_profile_matches(&candidates)
            .iter()
            .all(Option::is_none));
    }
}

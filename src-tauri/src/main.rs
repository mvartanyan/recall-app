#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::{HashMap, HashSet},
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
const MATCH_THRESHOLD: f32 = 0.90;
const MATCH_MARGIN: f32 = 0.06;
const PROFILE_CLAIM_MARGIN: f32 = 0.05;
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

fn transcribe_file_inner(
    path: &str,
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
    let initial_display = build_display_transcript(&segments, &result.transcript);
    let title = make_conversation_title(&result.transcript);
    let session_id = db
        .insert_session(&title, &initial_display, audio.duration_ms() as i64)
        .map_err(|error| format!("Could not save conversation: {error}"))?;

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
            &session_id,
            &db,
            embedder.as_ref(),
            app_handle,
            run_id,
        )?;
    } else {
        process_segments(
            &audio,
            &segments,
            &session_id,
            &db,
            None,
            app_handle,
            run_id,
        )?;
    }
    emit_progress(
        app_handle,
        "voiceprints:done",
        Some("Speaker attribution finished".into()),
        Some(run_id),
    );
    let saved_segments = db.list_segments(&session_id)?;
    let final_display = build_saved_transcript(&saved_segments, &result.transcript);
    db.update_session_transcript(&session_id, &final_display)?;
    emit_progress(
        app_handle,
        "transcription:done",
        Some("Conversation saved locally".into()),
        Some(run_id),
    );
    Ok(session_id)
}

fn queue_transcription(path: String, state: AppState, app_handle: tauri::AppHandle) -> String {
    let run_id = Uuid::new_v4().to_string();
    if let Ok(mut progress) = state.progress.lock() {
        progress.entry(run_id.clone()).or_default();
    }
    emit_progress(
        &app_handle,
        "queued",
        Some("Final transcription queued".into()),
        Some(&run_id),
    );
    let worker_run_id = run_id.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let result = transcribe_file_inner(&path, &state, &app_handle, &worker_run_id);
        if let Err(error) = std::fs::remove_file(&path) {
            if Path::new(&path).exists() {
                emit_progress(
                    &app_handle,
                    "audio:cleanup:warning",
                    Some(format!("Could not delete temporary recording: {error}")),
                    Some(&worker_run_id),
                );
            }
        } else {
            emit_progress(
                &app_handle,
                "audio:cleanup:done",
                Some("Temporary recording deleted".into()),
                Some(&worker_run_id),
            );
        }
        match result {
            Ok(session_id) => emit_progress(
                &app_handle,
                "complete",
                Some(session_id),
                Some(&worker_run_id),
            ),
            Err(error) => emit_progress(&app_handle, "error", Some(error), Some(&worker_run_id)),
        }
    });
    run_id
}

#[tauri::command]
fn transcribe_file_async(
    path: String,
    app_state: State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    if !Path::new(&path).is_file() {
        return Err("Recording file does not exist".into());
    }
    Ok(queue_transcription(
        path,
        app_state.inner().clone(),
        app_handle,
    ))
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

fn build_display_transcript(segments: &[TranscriptSegment], fallback: &str) -> String {
    let lines = segments
        .iter()
        .filter(|segment| !segment.text.trim().is_empty())
        .map(|segment| format!("{}: {}", segment.speaker, segment.text.trim()))
        .collect::<Vec<_>>();
    if lines.is_empty() {
        fallback.trim().to_string()
    } else {
        lines.join("\n")
    }
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

fn collect_audio_by_speaker(
    audio: &AudioClip,
    segments: &[TranscriptSegment],
) -> HashMap<String, Vec<f32>> {
    let mut buckets = HashMap::new();
    let target_samples = ((audio.sample_rate as u64 * TARGET_SPEAKER_MS) / 1_000) as usize;
    for segment in segments {
        let start = ((segment.start_ms as u128 * audio.sample_rate as u128) / 1_000) as usize;
        let end = ((segment.end_ms as u128 * audio.sample_rate as u128) / 1_000) as usize;
        let start = start.min(audio.samples.len());
        let end = end.min(audio.samples.len());
        if end <= start {
            continue;
        }
        let bucket: &mut Vec<f32> = buckets.entry(segment.speaker.clone()).or_default();
        let remaining = target_samples.saturating_sub(bucket.len());
        if remaining > 0 {
            bucket.extend_from_slice(&audio.samples[start..(start + remaining.min(end - start))]);
        }
    }
    buckets
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
    let buckets = collect_audio_by_speaker(audio, segments);
    let known = db.list_embeddings(EMBEDDING_VERSION)?;
    let mut ordered_speakers = Vec::new();
    let mut seen = HashSet::new();
    for segment in segments {
        if seen.insert(segment.speaker.clone()) {
            ordered_speakers.push(segment.speaker.clone());
        }
    }
    let mut observations = Vec::new();

    for diarized_speaker in ordered_speakers {
        let pcm = buckets.get(&diarized_speaker).cloned().unwrap_or_default();
        let sample_duration_ms = if audio.sample_rate == 0 {
            0
        } else {
            (pcm.len() as u64 * 1_000) / audio.sample_rate as u64
        };
        let embedding = match (
            embedder,
            pcm.is_empty(),
            sample_duration_ms >= MIN_SPEAKER_MS,
        ) {
            (Some(embedder), false, true) => match embedder.embed(&pcm, audio.sample_rate) {
                Ok(value) => Some(value),
                Err(error) => {
                    emit_progress(
                        app_handle,
                        "voiceprint:warning",
                        Some(format!("{diarized_speaker}: {error}")),
                        Some(run_id),
                    );
                    None
                }
            },
            (None, _, _) => {
                emit_progress(
                    app_handle,
                    "voiceprint:warning",
                    Some(format!(
                        "{diarized_speaker}: local ECAPA model is unavailable; leaving this intervention unattributed"
                    )),
                    Some(run_id),
                );
                None
            }
            (_, _, false) => {
                emit_progress(
                    app_handle,
                    "voiceprint:warning",
                    Some(format!(
                        "{diarized_speaker}: only {:.1} seconds of speech; at least {:.1} seconds is required for a voiceprint",
                        sample_duration_ms as f64 / 1_000.0,
                        MIN_SPEAKER_MS as f64 / 1_000.0,
                    )),
                    Some(run_id),
                );
                None
            }
            _ => None,
        };

        let Some(embedding) = embedding else {
            emit_progress(
                app_handle,
                "voiceprint:skipped",
                Some(format!(
                    "{diarized_speaker}: no persistent voice profile was created"
                )),
                Some(run_id),
            );
            continue;
        };

        observations.push(VoiceObservation {
            diarized_speaker,
            pcm,
            embedding,
        });
    }

    let candidates = observations
        .iter()
        .map(|observation| best_speaker_match(&observation.embedding, &known))
        .collect::<Vec<_>>();
    let resolved_matches = resolve_unique_profile_matches(&candidates);
    let mut mapping: HashMap<String, (String, String)> = HashMap::new();

    for (observation, (candidate, matched)) in observations
        .into_iter()
        .zip(candidates.into_iter().zip(resolved_matches))
    {
        let VoiceObservation {
            diarized_speaker,
            pcm,
            embedding,
        } = observation;
        let (speaker_id, label, is_new) = if let Some(matched) = matched {
            emit_progress(
                app_handle,
                "voiceprint:matched",
                Some(format!(
                    "{diarized_speaker} → {} ({:.2}; reference left unchanged)",
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
                Some(format!("{diarized_speaker} → {label} ({reason})")),
                Some(run_id),
            );
            (speaker_id, label, true)
        };

        // New provisional voices establish a reference. Automatic matches are
        // intentionally not fed back into the reference library: only a human
        // naming or assigning a provisional profile can expand a known person.
        if is_new {
            db.insert_embedding(&speaker_id, session_id, &embedding, EMBEDDING_VERSION)?;
        }
        if is_new && !pcm.is_empty() {
            let sample = encode_wav_base64(&pcm, audio.sample_rate)?;
            db.insert_sample(&speaker_id, &sample, audio.sample_rate)?;
            emit_progress(
                app_handle,
                "voiceprint:sample:stored",
                Some(format!("Stored a temporary preview for {label}")),
                Some(run_id),
            );
        }
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
    let db = app_state.db_handle()?;
    db.assign_segment_speaker(&segment_id, speaker_id.as_deref())?;
    refresh_session_transcript(&db, &session_id)
}

#[tauri::command]
fn delete_session(session_id: String, app_state: State<AppState>) -> Result<usize, String> {
    app_state.db_handle()?.delete_session(&session_id)
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
    app_state
        .db_handle()?
        .upsert_agenda(&session_id, "file", filename, mime_type, &content)
        .map(|agenda| Some(agenda.metadata()))
}

#[tauri::command]
fn remove_agenda(session_id: String, app_state: State<AppState>) -> Result<bool, String> {
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
        "llm",
        "Waiting for the LLM provider",
    );
    let response = openai::generate_recap(openai::RecapRequest {
        api_key: &api_key,
        model: &model,
        segments: &snapshot.segments,
        agenda: snapshot.agenda.as_ref(),
        no_translation_languages: &config.no_translation_languages,
    })
    .await?;
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
fn rename_speaker(
    speaker_id: String,
    new_label: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    let db = app_state.db_handle()?;
    let sessions = db.session_ids_for_speakers(&[speaker_id.as_str()])?;
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
                        let run_id = queue_transcription(path, state, app.clone());
                        let _ = app.emit("transcription:queued", run_id);
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
    fn onboarding_version_is_explicitly_versioned() {
        assert_eq!(ONBOARDING_VERSION, "1");
        assert_eq!(AppConfig::default().onboarding_version, None);
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
        assert!(best_speaker_match(&[0.91, 0.414_608_24], &known).is_some());
        assert!(best_speaker_match(&[0.89, 0.455_960_5], &known).is_none());
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

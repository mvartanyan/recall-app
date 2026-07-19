#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{mpsc, Arc, Mutex},
    thread::{self, JoinHandle},
    time::Duration,
};

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
mod db;
mod embedding;
mod keychain;
mod soniox;
mod state;

use config::AppConfig;
use db::{Crypto, Db, SegmentRecord, Session, Speaker, StoredEmbedding};
use embedding::EMBEDDING_VERSION;
use soniox::{LiveAudioMessage, TranscriptSegment};
use state::AppState;

const TARGET_SPEAKER_MS: u64 = 12_000;
const MATCH_THRESHOLD: f32 = 0.90;
const MATCH_MARGIN: f32 = 0.04;

#[derive(Debug, Serialize, Clone)]
struct ProgressEvent {
    event_id: String,
    stage: String,
    detail: Option<String>,
    run_id: Option<String>,
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
    speaker_model_available: bool,
    selected_input_device: Option<String>,
    language_hints: Vec<String>,
    live_transcription: bool,
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
    if meter_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 8 == 0 {
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
    let api_key = keychain::load_api_key()?;
    let config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?
        .clone();
    let requested = input_device.or(config.selected_input_device.clone());
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
    let api_key = keychain::load_api_key()?;
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
        Some("Extracting and matching local voiceprints".into()),
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
    let mut mapping: HashMap<String, (String, String)> = HashMap::new();

    for diarized_speaker in ordered_speakers {
        let pcm = buckets.get(&diarized_speaker).cloned().unwrap_or_default();
        let embedding = match (embedder, pcm.is_empty()) {
            (Some(embedder), false) => match embedder.embed(&pcm, audio.sample_rate) {
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
            _ => None,
        };

        let matched = embedding
            .as_ref()
            .and_then(|query| best_speaker_match(query, &known));
        let (speaker_id, label, is_new) = if let Some((speaker_id, label, score)) = matched {
            emit_progress(
                app_handle,
                "voiceprint:matched",
                Some(format!("{diarized_speaker} → {label} ({score:.2})")),
                Some(run_id),
            );
            (speaker_id, label, false)
        } else {
            let label = db.next_voice_label()?;
            let speaker_id = db.insert_speaker(Some(&label))?;
            emit_progress(
                app_handle,
                "voiceprint:new",
                Some(format!("{diarized_speaker} → {label}")),
                Some(run_id),
            );
            (speaker_id, label, true)
        };

        if let Some(embedding) = embedding.as_ref() {
            db.insert_embedding(&speaker_id, session_id, embedding, EMBEDDING_VERSION)?;
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

fn best_speaker_match(query: &[f32], known: &[StoredEmbedding]) -> Option<(String, String, f32)> {
    let mut by_speaker: HashMap<&str, (&StoredEmbedding, f32)> = HashMap::new();
    for candidate in known {
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
    Some((
        best.speaker_id.clone(),
        best.speaker_label
            .clone()
            .unwrap_or_else(|| "Unnamed speaker".into()),
        score,
    ))
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
fn save_soniox_key(api_key: String) -> Result<(), String> {
    keychain::save_api_key(&api_key)
}

#[tauri::command]
fn delete_soniox_key() -> Result<(), String> {
    keychain::delete_api_key()
}

#[tauri::command]
fn soniox_key_status() -> bool {
    keychain::has_api_key()
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
    app_state: State<AppState>,
) -> Result<(), String> {
    let mut config = app_state
        .config
        .lock()
        .map_err(|_| "Configuration lock poisoned")?;
    config.selected_input_device = selected_input_device.filter(|value| !value.trim().is_empty());
    config.language_hints = language_hints
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect();
    config.live_transcription = live_transcription;
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
        soniox_key_configured: keychain::has_api_key(),
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
fn delete_session(session_id: String, app_state: State<AppState>) -> Result<(), String> {
    app_state.db_handle()?.delete_session(&session_id)
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
            save_soniox_key,
            delete_soniox_key,
            soniox_key_status,
            get_preferences,
            save_preferences,
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
            list_speakers,
            list_speakers_with_stats,
            rename_speaker,
            delete_speaker,
            get_speaker_samples,
            merge_speakers,
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
            StoredEmbedding {
                id: "e1".into(),
                speaker_id: "s1".into(),
                speaker_label: Some("Alice".into()),
                vector: vec![1.0, 0.0],
                source_session_id: "x".into(),
                created_at: chrono::Utc::now(),
                model_version: EMBEDDING_VERSION.into(),
            },
            StoredEmbedding {
                id: "e2".into(),
                speaker_id: "s2".into(),
                speaker_label: Some("Bob".into()),
                vector: vec![0.0, 1.0],
                source_session_id: "x".into(),
                created_at: chrono::Utc::now(),
                model_version: EMBEDDING_VERSION.into(),
            },
        ];
        let matched = best_speaker_match(&[1.0, 0.0], &known).unwrap();
        assert_eq!(matched.0, "s1");
        assert!(best_speaker_match(&[0.71, 0.70], &known).is_none());
    }
}

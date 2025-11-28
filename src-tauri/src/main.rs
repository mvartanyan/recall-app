#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, StreamConfig,
};
mod db;
mod embedding;
mod config;
mod state;
use state::AppState;
use db::{Crypto, Db, SegmentRecord, Session, Speaker, StoredEmbedding};
use chrono::Utc;
use reqwest::blocking::{multipart, Client};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use tauri::{
    image::Image,
    menu::{MenuBuilder, MenuId, MenuItem},
    tray::TrayIconBuilder,
    Emitter, Manager, State,
};

#[derive(Debug)]
enum SampleChunk {
    F32(Vec<f32>),
    I16(Vec<i16>),
}

const TARGET_SPEAKER_MS: u64 = 10_000;
const MATCH_THRESHOLD: f32 = 0.78;

#[derive(Debug, Deserialize, Clone)]
struct ApiSegment {
    speaker: String,
    start_ms: u64,
    end_ms: u64,
    text: String,
}

#[derive(Debug, Deserialize, Clone)]
struct ApiTranscribeResponse {
    transcript: String,
    summary: Option<String>,
    speakers: Vec<String>,
    segments: Option<Vec<ApiSegment>>,
    audio_url: Option<String>,
}

#[derive(Debug, Serialize)]
struct AppStatus {
    encryption_enabled: bool,
    db_open: bool,
    needs_password: bool,
    api_base: Option<String>,
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
            (self.samples.len() as u64 * 1000) / self.sample_rate as u64
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
    fn start(&self) -> Result<PathBuf, String> {
        let mut guard = self.current.lock().map_err(|_| "Lock poisoned")?;
        if guard.is_some() {
            return Err("Recording already in progress".into());
        }

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| "No input device found".to_string())?;
        let input_config = device
            .default_input_config()
            .map_err(|e| format!("Failed to get input config: {e}"))?;
        let sample_format = input_config.sample_format();
        let config: StreamConfig = input_config.into();
        let sample_rate = config.sample_rate.0;
        let channels = config.channels;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();
        let output = std::env::temp_dir().join(format!("recall-{timestamp}.wav"));
        let output_for_api = output.clone();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let output_for_thread = output.clone();
        let handle = thread::spawn(move || -> Result<PathBuf, String> {
            let wav_spec = match sample_format {
                SampleFormat::F32 => hound::WavSpec {
                    channels,
                    sample_rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
                SampleFormat::I16 | SampleFormat::U16 => hound::WavSpec {
                    channels,
                    sample_rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
                _ => return Err("Unsupported sample format".into()),
            };

            let (data_tx, data_rx) = mpsc::channel::<SampleChunk>();
            let stop_flag = Arc::new(AtomicBool::new(false));
            let cb_flag = stop_flag.clone();
            let err_fn = |err| eprintln!("recording error: {err}");

            let stream = match sample_format {
                SampleFormat::F32 => {
                    let tx = data_tx.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[f32], _| {
                                if !cb_flag.load(Ordering::Relaxed) {
                                    let _ = tx.send(SampleChunk::F32(data.to_vec()));
                                }
                            },
                            err_fn,
                            None,
                        )
                        .map_err(|e| format!("Failed to build input stream: {e}"))?
                }
                SampleFormat::I16 => {
                    let tx = data_tx.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[i16], _| {
                                if !cb_flag.load(Ordering::Relaxed) {
                                    let _ = tx.send(SampleChunk::I16(data.to_vec()));
                                }
                            },
                            err_fn,
                            None,
                        )
                        .map_err(|e| format!("Failed to build input stream: {e}"))?
                }
                SampleFormat::U16 => {
                    let tx = data_tx.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[u16], _| {
                                if !cb_flag.load(Ordering::Relaxed) {
                                    let converted: Vec<i16> =
                                        data.iter().map(|s| (*s as i32 - 32768) as i16).collect();
                                    let _ = tx.send(SampleChunk::I16(converted));
                                }
                            },
                            err_fn,
                            None,
                        )
                        .map_err(|e| format!("Failed to build input stream: {e}"))?
                }
                _ => return Err("Unsupported sample format".into()),
            };

            stream
                .play()
                .map_err(|e| format!("Failed to start input stream: {e}"))?;

            let writer_output = output_for_thread.clone();
            let writer_stop = stop_flag.clone();
            let writer = thread::spawn(move || -> Result<(), String> {
                let mut writer = hound::WavWriter::create(&writer_output, wav_spec)
                    .map_err(|e| e.to_string())?;
                for chunk in data_rx.iter() {
                    if writer_stop.load(Ordering::SeqCst) {
                        break;
                    }
                    match chunk {
                        SampleChunk::F32(data) => {
                            for sample in data {
                                writer.write_sample(sample).map_err(|e| e.to_string())?;
                            }
                        }
                        SampleChunk::I16(data) => {
                            for sample in data {
                                writer.write_sample(sample).map_err(|e| e.to_string())?;
                            }
                        }
                    }
                }
                writer.finalize().map_err(|e| e.to_string())?;
                Ok(())
            });

            let _ = stop_rx.recv();
            stop_flag.store(true, Ordering::SeqCst);
            drop(stream);
            // allow callback to unwind
            thread::sleep(Duration::from_millis(50));
            drop(data_tx);
            let _ = writer
                .join()
                .map_err(|_| "Writer join error".to_string())??;
            Ok(output)
        });

        *guard = Some(Recorder {
            stop_tx: Some(stop_tx),
            handle: Some(handle),
        });

        Ok(output_for_api)
    }

    fn stop(&self) -> Result<PathBuf, String> {
        let mut guard = self.current.lock().map_err(|_| "Lock poisoned")?;
        let mut recorder = guard
            .take()
            .ok_or_else(|| "No active recording".to_string())?;

        if let Some(tx) = recorder.stop_tx.take() {
            let _ = tx.send(());
        }

        if let Some(handle) = recorder.handle.take() {
            let path = handle.join().map_err(|_| "Join error".to_string())??;
            return Ok(path);
        }

        Err("No recorder thread found".into())
    }
}

#[tauri::command]
fn start_recording(state: State<RecordingManager>) -> Result<PathBuf, String> {
    state.start()
}

#[tauri::command]
fn stop_recording(state: State<RecordingManager>) -> Result<PathBuf, String> {
    state.stop()
}

#[tauri::command]
fn transcribe_file(
    path: String,
    api_base: Option<String>,
    app_state: State<AppState>,
) -> Result<String, String> {
    let api_base = api_base
        .or_else(|| {
            let cfg = app_state.config.lock().ok()?.clone();
            cfg.api_base
        })
        .unwrap_or_else(|| "http://localhost:8787".to_string());

    // ensure embedder is available before processing results
    {
        let embedder_loaded = app_state.embedder.lock().map_err(|_| "embedder lock")?.is_some();
        if !embedder_loaded {
            app_state.load_embedder()?;
        }
    }

    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard
        .as_ref()
        .ok_or("Database not initialized (unlock to proceed)")?;
    let _ = db.encrypted;

    let url = Url::parse(&api_base)
        .map_err(|e| format!("Invalid API base: {e}"))?
        .join("v1/transcribe")
        .map_err(|e| format!("Invalid endpoint: {e}"))?;

    let file_bytes = std::fs::read(&path).map_err(|e| format!("Failed to read file: {e}"))?;
    let part = multipart::Part::bytes(file_bytes).file_name("audio.wav");
    let form = multipart::Form::new().part("file", part);

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(240))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let res = client
        .post(url)
        .multipart(form)
        .send()
        .map_err(|e| format!("HTTP error: {e}"))?;

    if !res.status().is_success() {
        return Err(format!("API responded with status {}", res.status()));
    }

    let api_resp: ApiTranscribeResponse = res
        .json()
        .map_err(|e| format!("Decode error: {e}"))?;
    let _ = (&api_resp.summary, &api_resp.speakers, &api_resp.audio_url);

    let audio_clip = read_audio_clip(&path)?;
    let segments = normalize_segments(api_resp.segments.clone(), &api_resp.transcript, &audio_clip);

    let session_id = db
        .insert_session(&api_resp.transcript)
        .map_err(|e| format!("DB error: {e}"))?;

    {
        let mut embedder_guard = app_state.embedder.lock().map_err(|_| "embedder lock")?;
        let embedder = embedder_guard
            .as_mut()
            .ok_or("Embedder not initialized")?;
        process_segments(&audio_clip, &segments, &session_id, db, embedder)?;
    }

    let _ = std::fs::remove_file(&path);

    Ok(api_resp.transcript)
}

fn read_audio_clip(path: &str) -> Result<AudioClip, String> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| format!("Failed to open audio for embeddings: {e}"))?;
    let spec = reader.spec();
    let channels = std::cmp::max(spec.channels as usize, 1);
    let mut interleaved: Vec<f32> = Vec::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            for sample in reader.samples::<i16>() {
                let s = sample.map_err(|e| format!("Sample decode error: {e}"))?;
                interleaved.push(s as f32 / i16::MAX as f32);
            }
        }
        (hound::SampleFormat::Int, 24) | (hound::SampleFormat::Int, 32) => {
            for sample in reader.samples::<i32>() {
                let s = sample.map_err(|e| format!("Sample decode error: {e}"))?;
                interleaved.push(s as f32 / i32::MAX as f32);
            }
        }
        (hound::SampleFormat::Float, _) => {
            for sample in reader.samples::<f32>() {
                let s = sample.map_err(|e| format!("Sample decode error: {e}"))?;
                interleaved.push(s);
            }
        }
        _ => return Err("Unsupported WAV format for embedding".into()),
    }
    if interleaved.is_empty() {
        return Err("Audio buffer is empty".into());
    }
    let mut mono = Vec::with_capacity(interleaved.len() / channels + 1);
    for frame in interleaved.chunks(channels) {
        let sum: f32 = frame.iter().sum();
        mono.push(sum / channels as f32);
    }
    Ok(AudioClip {
        samples: mono,
        sample_rate: spec.sample_rate,
    })
}

fn normalize_segments(
    segments: Option<Vec<ApiSegment>>,
    transcript: &str,
    audio: &AudioClip,
) -> Vec<ApiSegment> {
    let mut segs = segments.unwrap_or_default();
    if segs.is_empty() {
        let end_ms = audio.duration_ms().max(1_000);
        segs.push(ApiSegment {
            speaker: "speaker_0".to_string(),
            start_ms: 0,
            end_ms,
            text: transcript.to_string(),
        });
    }

    let max_end = audio.duration_ms();
    for seg in segs.iter_mut() {
        if seg.end_ms == 0 || seg.end_ms < seg.start_ms {
            seg.end_ms = seg.start_ms.saturating_add(1_000);
        }
        if max_end > 0 && seg.end_ms > max_end {
            seg.end_ms = max_end;
        }
    }
    segs.sort_by_key(|s| s.start_ms);
    segs
}

fn collect_audio_by_speaker(
    audio: &AudioClip,
    segments: &[ApiSegment],
) -> HashMap<String, Vec<f32>> {
    let mut buckets: HashMap<String, Vec<f32>> = HashMap::new();
    let total_samples = audio.samples.len();
    let target_samples =
        std::cmp::max(1, ((audio.sample_rate as u64 * TARGET_SPEAKER_MS) / 1000) as usize);
    let sr = audio.sample_rate as f64;

    for seg in segments {
        let start = ((seg.start_ms as f64 / 1000.0) * sr).floor() as usize;
        let end = ((seg.end_ms as f64 / 1000.0) * sr).ceil() as usize;
        if end <= start {
            continue;
        }
        let start_idx = std::cmp::min(start, total_samples);
        let end_idx = std::cmp::min(end, total_samples);
        if end_idx <= start_idx {
            continue;
        }
        let entry = buckets.entry(seg.speaker.clone()).or_default();
        let remaining = target_samples.saturating_sub(entry.len());
        if remaining == 0 {
            continue;
        }
        let take_len = std::cmp::min(remaining, end_idx - start_idx);
        entry.extend_from_slice(&audio.samples[start_idx..start_idx + take_len]);
    }

    buckets
}

fn best_match<'a>(
    embedding: &[f32],
    known: &'a [StoredEmbedding],
) -> Option<(&'a StoredEmbedding, f32)> {
    let mut best: Option<(&StoredEmbedding, f32)> = None;
    for record in known {
        if record.vector.len() != embedding.len() {
            continue;
        }
        let score = embedding::cosine_similarity(embedding, &record.vector);
        match best {
            Some((_, current)) if score <= current => continue,
            _ => best = Some((record, score)),
        }
    }
    if let Some((rec, score)) = best {
        if score >= MATCH_THRESHOLD {
            return Some((rec, score));
        }
    }
    None
}

fn process_segments(
    audio: &AudioClip,
    segments: &[ApiSegment],
    session_id: &str,
    db: &Db,
    embedder: &mut crate::embedding::Embedder,
) -> Result<(), String> {
    let mut diarization_to_profile: HashMap<String, (String, String)> = HashMap::new();
    let mut known_embeddings = db.list_embeddings()?;
    let speakers = db.list_speakers()?;
    let mut next_label_index = speakers.len() + 1;

    for (speaker_key, pcm) in collect_audio_by_speaker(audio, segments) {
        if pcm.is_empty() {
            continue;
        }
        let embedding_vec = embedder.embed(&pcm)?;
        let (speaker_id, speaker_label) = if let Some((matched, _score)) = best_match(&embedding_vec, &known_embeddings) {
            let label = matched
                .speaker_label
                .clone()
                .unwrap_or_else(|| {
                    let generated = format!("Speaker {}", next_label_index);
                    next_label_index += 1;
                    generated
                });
            if matched.speaker_label.is_none() {
                db.rename_speaker(&matched.speaker_id, &label)?;
            }
            (matched.speaker_id.clone(), label)
        } else {
            let label = format!("Speaker {}", next_label_index);
            next_label_index += 1;
            let id = db.insert_speaker(Some(&label))?;
            (id, label)
        };

        let embedding_id = db.insert_embedding(&speaker_id, session_id, &embedding_vec)?;
        known_embeddings.push(StoredEmbedding {
            id: embedding_id,
            speaker_id: speaker_id.clone(),
            speaker_label: Some(speaker_label.clone()),
            vector: embedding_vec,
            source_session_id: session_id.to_string(),
            created_at: Utc::now(),
        });
        diarization_to_profile.insert(speaker_key, (speaker_id, speaker_label));
    }

    for seg in segments {
        let (speaker_id, speaker_label) = diarization_to_profile
            .get(&seg.speaker)
            .cloned()
            .unwrap_or_else(|| (String::new(), seg.speaker.clone()));
        let speaker_id_opt = if speaker_id.is_empty() {
            None
        } else {
            Some(speaker_id.as_str())
        };
        db.insert_segment(
            session_id,
            seg.start_ms as i64,
            seg.end_ms as i64,
            speaker_id_opt,
            Some(&speaker_label),
            &seg.text,
        )
        .map_err(|e| format!("DB error: {e}"))?;
    }

    Ok(())
}

#[tauri::command]
fn unlock_db(password: String, app_state: State<AppState>) -> Result<(), String> {
    let cfg = app_state.config.lock().map_err(|_| "config lock")?.clone();
    if !cfg.encryption_enabled {
        return Err("Encryption is not enabled".into());
    }
    let salt = Db::load_existing_salt(app_state.db_path()).unwrap_or(None);
    let crypto = Crypto::new(Some(&password), salt);
    app_state.open_db(crypto)
}

#[tauri::command]
fn enable_encryption(password: String, app_state: State<AppState>) -> Result<(), String> {
    {
        let mut cfg = app_state.config.lock().map_err(|_| "config lock")?;
        cfg.encryption_enabled = true;
        cfg.save(&app_state.config_path)?;
    }
    // Recreate DB encrypted (note: existing plaintext data not migrated).
    {
        let mut db_guard = app_state.db.lock().map_err(|_| "db lock")?;
        *db_guard = None;
    }
    let db_path = app_state.data_dir.join("recall.db");
    let _ = std::fs::remove_file(&db_path);
    let crypto = Crypto::new(Some(&password), None);
    app_state.open_db(crypto)
}

#[tauri::command]
fn app_status(app_state: State<AppState>) -> Result<AppStatus, String> {
    let cfg = app_state
        .config
        .lock()
        .map_err(|_| "config lock")?
        .clone();
    let db_open = app_state.db.lock().map_err(|_| "DB lock poisoned")?.is_some();
    Ok(AppStatus {
        encryption_enabled: cfg.encryption_enabled,
        db_open,
        needs_password: cfg.encryption_enabled && !db_open,
        api_base: cfg.api_base,
    })
}

#[tauri::command]
fn list_sessions(app_state: State<AppState>) -> Result<Vec<Session>, String> {
    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.list_sessions()
}

#[tauri::command]
fn list_segments(session_id: String, app_state: State<AppState>) -> Result<Vec<SegmentRecord>, String> {
    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.list_segments(&session_id)
}

#[tauri::command]
fn update_transcript(
    session_id: String,
    transcript: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.update_session_transcript(&session_id, &transcript)
}

#[tauri::command]
fn delete_session(session_id: String, app_state: State<AppState>) -> Result<(), String> {
    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.delete_session(&session_id)
}

#[tauri::command]
fn list_speakers(app_state: State<AppState>) -> Result<Vec<Speaker>, String> {
    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.list_speakers()
}

#[tauri::command]
fn rename_speaker(
    speaker_id: String,
    new_label: String,
    app_state: State<AppState>,
) -> Result<(), String> {
    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.rename_speaker(&speaker_id, &new_label)
}

#[tauri::command]
fn delete_speaker(speaker_id: String, app_state: State<AppState>) -> Result<(), String> {
    let db_guard = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.delete_speaker(&speaker_id)
}

fn build_tray(app: &mut tauri::App) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, MenuId::new("open"), "Open", true, None::<&str>)?;
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

    // Simple solid icon (1x1 RGBA).
    let icon = Image::new(&[30, 60, 120, 255], 1, 1);

    TrayIconBuilder::new()
        .icon(icon)
        .menu(&menu)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "open" => {
                if let Some(win) = app.get_webview_window("main") {
                    let _ = win.show();
                    let _ = win.set_focus();
                }
            }
            "start" => {
                let _ = app.emit("recording:start", ());
            }
            "stop" => {
                let _ = app.emit("recording:stop", ());
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
            start_recording,
            stop_recording,
            transcribe_file,
            unlock_db,
            enable_encryption,
            app_status,
            list_sessions,
            list_segments,
            update_transcript,
            delete_session,
            list_speakers,
            rename_speaker,
            delete_speaker
        ])
        .manage(RecordingManager::default())
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::env::temp_dir().join("recall"));
            std::fs::create_dir_all(&data_dir).ok();
            let app_state = AppState::new(data_dir);
            {
                let cfg = app_state.config.lock().unwrap().clone();
                if !cfg.encryption_enabled {
                    let _ = app_state.open_db(Crypto::new(None, None));
                }
            }
            app.manage(app_state);

            build_tray(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

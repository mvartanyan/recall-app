#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
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
use db::{Crypto, Db};
use reqwest::blocking::{multipart, Client};
use reqwest::Url;
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

#[derive(Debug)]
struct Recorder {
    stop_tx: Option<mpsc::Sender<()>>,
    handle: Option<JoinHandle<Result<PathBuf, String>>>,
}

#[derive(Default)]
struct RecordingManager {
    current: Mutex<Option<Recorder>>,
}

struct AppState {
    db: std::sync::Arc<Mutex<Db>>,
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
        .or_else(|| std::env::var("RECALL_API_BASE").ok())
        .unwrap_or_else(|| "http://localhost:8000".to_string());

    let url = Url::parse(&api_base)
        .map_err(|e| format!("Invalid API base: {e}"))?
        .join("v1/transcribe-local")
        .map_err(|e| format!("Invalid endpoint: {e}"))?;

    let file_bytes = std::fs::read(&path).map_err(|e| format!("Failed to read file: {e}"))?;
    let part = multipart::Part::bytes(file_bytes).file_name("audio.wav");
    let form = multipart::Form::new().part("file", part);

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
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

    let json: serde_json::Value = res.json().map_err(|e| format!("Decode error: {e}"))?;
    let transcript = json
        .get("transcript")
        .and_then(|v| v.as_str())
        .unwrap_or("(no transcript)")
        .to_string();

    // Persist session with transcript and delete audio file.
    let db = app_state.db.lock().map_err(|_| "DB lock poisoned")?;
    let _session_id = db
        .insert_session(&transcript)
        .map_err(|e| format!("DB error: {e}"))?;
    let _ = std::fs::remove_file(&path);

    Ok(transcript)
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
            transcribe_file
        ])
        .manage(RecordingManager::default())
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::env::temp_dir().join("recall"));
            std::fs::create_dir_all(&data_dir).ok();
            let db_path = data_dir.join("recall.db");
            let crypto = Crypto::new(None, None);
            let db = Db::open(db_path, crypto).map_err(|e| e.to_string())?;
            app.manage(AppState {
                db: std::sync::Arc::new(std::sync::Mutex::new(db)),
            });

            build_tray(app)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

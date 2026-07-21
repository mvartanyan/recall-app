use std::{path::PathBuf, time::Duration};

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::time::{sleep, timeout};
use tokio_tungstenite::{connect_async, tungstenite::Message};

const URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";

fn main() -> Result<(), String> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?
        .block_on(run())
}

async fn run() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let path = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "WAV path is required".to_string())?;
    let key_path = arguments.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
            .join("Library/Application Support/com.example.recall/soniox-api-key")
    });
    let api_key = std::fs::read_to_string(&key_path)
        .map_err(|error| format!("Could not read {}: {error}", key_path.display()))?;
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(format!("{} is empty", key_path.display()));
    }
    let mut reader = hound::WavReader::open(path).map_err(|error| error.to_string())?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("Probe requires mono signed 16-bit PCM WAV".into());
    }
    let mut audio = Vec::new();
    for sample in reader.samples::<i16>() {
        audio.extend_from_slice(&sample.map_err(|error| error.to_string())?.to_le_bytes());
    }

    let (socket, _) = timeout(Duration::from_secs(10), connect_async(URL))
        .await
        .map_err(|_| "WebSocket connection timed out".to_string())?
        .map_err(|error| error.to_string())?;
    println!("connected");
    let (mut writer, mut receiver) = socket.split();
    writer
        .send(Message::Text(
            json!({
                "api_key": api_key,
                "model": "stt-rt-v5",
                "audio_format": "pcm_s16le",
                "sample_rate": spec.sample_rate,
                "num_channels": 1,
                "enable_speaker_diarization": true,
                "enable_language_identification": true,
                "enable_endpoint_detection": false,
                "language_hints": ["en", "fr", "de", "es", "ru"],
            })
            .to_string(),
        ))
        .await
        .map_err(|error| error.to_string())?;
    println!("configured");

    let send = async move {
        let chunk_size = (spec.sample_rate as usize * 2) / 10;
        for chunk in audio.chunks(chunk_size) {
            writer
                .send(Message::Binary(chunk.to_vec()))
                .await
                .map_err(|error| error.to_string())?;
            sleep(Duration::from_millis(100)).await;
        }
        writer
            .send(Message::Text(String::new()))
            .await
            .map_err(|error| error.to_string())?;
        println!("audio_finished");
        Ok::<(), String>(())
    };
    let receive = async move {
        let mut updates = 0usize;
        let mut tokens = 0usize;
        while let Some(message) = timeout(Duration::from_secs(15), receiver.next())
            .await
            .map_err(|_| "Timed out waiting for Soniox response".to_string())?
        {
            let message = message.map_err(|error| error.to_string())?;
            let text = match message {
                Message::Text(text) => text.to_string(),
                Message::Binary(bytes) => {
                    String::from_utf8(bytes.to_vec()).map_err(|error| error.to_string())?
                }
                Message::Close(_) => return Err("Soniox closed before finished".into()),
                _ => continue,
            };
            let response: Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
            if response.get("error_code").is_some() {
                return Err(format!(
                    "{}: {}",
                    response["error_type"].as_str().unwrap_or("realtime_error"),
                    response["error_message"].as_str().unwrap_or("No details")
                ));
            }
            updates += 1;
            tokens += response["tokens"].as_array().map(Vec::len).unwrap_or(0);
            if response["finished"].as_bool().unwrap_or(false) {
                println!("finished updates={updates} tokens={tokens}");
                return Ok::<(), String>(());
            }
        }
        Err("Soniox response stream ended before finished".into())
    };
    futures_util::future::try_join(send, receive).await?;
    Ok(())
}

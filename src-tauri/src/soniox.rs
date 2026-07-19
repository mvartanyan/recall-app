use std::{
    collections::HashSet,
    path::Path,
    thread,
    time::{Duration, Instant},
};

use futures_util::{SinkExt, StreamExt};
use reqwest::blocking::{multipart, Client, Response};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::Emitter;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const REST_BASE: &str = "https://api.soniox.com/v1";
const REALTIME_URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";
const ASYNC_MODEL: &str = "stt-async-v5";
const REALTIME_MODEL: &str = "stt-rt-v5";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptSegment {
    pub speaker: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub transcript: String,
    pub speakers: Vec<String>,
    pub segments: Vec<TranscriptSegment>,
}

#[derive(Debug, Deserialize)]
struct IdResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct StatusResponse {
    status: String,
    error_type: Option<String>,
    error_message: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Token {
    #[serde(default)]
    text: String,
    #[serde(default)]
    start_ms: Option<u64>,
    #[serde(default)]
    end_ms: Option<u64>,
    #[serde(default)]
    speaker: Option<Value>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    is_final: bool,
}

#[derive(Debug, Deserialize)]
struct TranscriptResponse {
    #[serde(default)]
    text: String,
    #[serde(default)]
    tokens: Vec<Token>,
}

#[derive(Debug, Deserialize)]
struct RealtimeResponse {
    #[serde(default)]
    tokens: Vec<Token>,
    #[serde(default)]
    finished: bool,
    error_code: Option<u16>,
    error_type: Option<String>,
    error_message: Option<String>,
    request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveTranscriptEvent {
    pub text: String,
    pub final_text: String,
    pub finished: bool,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug)]
pub enum LiveAudioMessage {
    Audio(Vec<u8>),
    Finish,
}

pub fn transcribe_file<F>(
    path: &Path,
    api_key: &str,
    language_hints: &[String],
    mut progress: F,
) -> Result<TranscriptResult, String>
where
    F: FnMut(&str, String),
{
    let client = Client::builder()
        .timeout(Duration::from_secs(90))
        .user_agent("Recall/0.1")
        .build()
        .map_err(|error| format!("Could not create Soniox client: {error}"))?;
    let mut file_id: Option<String> = None;
    let mut transcription_id: Option<String> = None;

    let result = (|| {
        progress("soniox:upload:start", "Uploading recording".into());
        let bytes = std::fs::read(path)
            .map_err(|error| format!("Could not read the recording: {error}"))?;
        if bytes.is_empty() {
            return Err("The recording is empty".into());
        }
        let filename = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("recording.wav")
            .to_string();
        let upload = client
            .post(format!("{REST_BASE}/files"))
            .bearer_auth(api_key)
            .multipart(
                multipart::Form::new()
                    .part("file", multipart::Part::bytes(bytes).file_name(filename)),
            )
            .send()
            .map_err(|error| format!("Soniox upload failed: {error}"))?;
        let uploaded: IdResponse = decode_response(upload, "upload recording")?;
        file_id = Some(uploaded.id.clone());
        progress("soniox:upload:done", "Recording uploaded".into());

        progress(
            "soniox:transcription:start",
            format!("Starting final transcription with {ASYNC_MODEL}"),
        );
        let hints = normalize_language_hints(language_hints);
        let mut payload = json!({
            "model": ASYNC_MODEL,
            "file_id": uploaded.id,
            "enable_speaker_diarization": true,
            "enable_language_identification": true,
        });
        if !hints.is_empty() {
            payload["language_hints"] = json!(hints);
        }
        let create = client
            .post(format!("{REST_BASE}/transcriptions"))
            .bearer_auth(api_key)
            .json(&payload)
            .send()
            .map_err(|error| format!("Could not start Soniox transcription: {error}"))?;
        let created: IdResponse = decode_response(create, "start transcription")?;
        transcription_id = Some(created.id.clone());
        progress(
            "soniox:transcription:waiting",
            "Soniox is processing the recording".into(),
        );

        let deadline = Instant::now() + Duration::from_secs(20 * 60);
        let mut previous_status = String::new();
        loop {
            if Instant::now() >= deadline {
                return Err("Soniox transcription timed out after 20 minutes".into());
            }
            let response = client
                .get(format!("{REST_BASE}/transcriptions/{}", created.id))
                .bearer_auth(api_key)
                .send()
                .map_err(|error| format!("Could not poll Soniox transcription: {error}"))?;
            let status: StatusResponse = decode_response(response, "poll transcription")?;
            if status.status != previous_status {
                previous_status = status.status.clone();
                progress(
                    "soniox:transcription:status",
                    format!("Soniox status: {}", status.status),
                );
            }
            match status.status.as_str() {
                "completed" => break,
                "error" | "failed" => {
                    let kind = status.error_type.unwrap_or_else(|| "unknown_error".into());
                    let message = status
                        .error_message
                        .unwrap_or_else(|| "No details returned".into());
                    return Err(format!("Soniox transcription failed ({kind}): {message}"));
                }
                _ => thread::sleep(Duration::from_secs(2)),
            }
        }

        progress(
            "soniox:transcript:download:start",
            "Downloading final diarized transcript".into(),
        );
        let response = client
            .get(format!(
                "{REST_BASE}/transcriptions/{}/transcript",
                created.id
            ))
            .bearer_auth(api_key)
            .send()
            .map_err(|error| format!("Could not download Soniox transcript: {error}"))?;
        let transcript: TranscriptResponse = decode_response(response, "download transcript")?;
        let parsed = parse_transcript(transcript);
        progress(
            "soniox:transcript:download:done",
            format!(
                "Downloaded {} interventions from {} diarized speakers",
                parsed.segments.len(),
                parsed.speakers.len()
            ),
        );
        Ok(parsed)
    })();

    progress(
        "soniox:cleanup:start",
        "Removing provider-side artifacts".into(),
    );
    let mut cleanup_errors = Vec::new();
    if let Some(id) = transcription_id.as_deref() {
        if let Err(error) = delete_resource(&client, api_key, "transcriptions", id) {
            cleanup_errors.push(error);
        }
    }
    if let Some(id) = file_id.as_deref() {
        if let Err(error) = delete_resource(&client, api_key, "files", id) {
            cleanup_errors.push(error);
        }
    }
    if cleanup_errors.is_empty() {
        progress(
            "soniox:cleanup:done",
            "Provider-side artifacts removed".into(),
        );
    } else {
        progress(
            "soniox:cleanup:warning",
            format!("Cleanup incomplete: {}", cleanup_errors.join("; ")),
        );
    }
    result
}

pub async fn run_realtime(
    api_key: String,
    language_hints: Vec<String>,
    sample_rate: u32,
    mut audio_rx: mpsc::UnboundedReceiver<LiveAudioMessage>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    emit_live(&app_handle, "Connecting live captions", "", "", false, None);
    let (mut socket, _) = connect_async(REALTIME_URL)
        .await
        .map_err(|error| format!("Could not connect to Soniox live transcription: {error}"))?;
    let hints = normalize_language_hints(&language_hints);
    let mut config = json!({
        "api_key": api_key,
        "model": REALTIME_MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": sample_rate,
        "num_channels": 1,
        "enable_speaker_diarization": true,
        "enable_language_identification": true,
        "enable_endpoint_detection": false,
    });
    if !hints.is_empty() {
        config["language_hints"] = json!(hints);
    }
    socket
        .send(Message::Text(config.to_string()))
        .await
        .map_err(|error| format!("Could not configure Soniox live transcription: {error}"))?;
    emit_live(&app_handle, "Live captions connected", "", "", false, None);

    let (mut writer, mut reader) = socket.split();
    let send_audio = async move {
        while let Some(message) = audio_rx.recv().await {
            match message {
                LiveAudioMessage::Audio(bytes) => writer
                    .send(Message::Binary(bytes))
                    .await
                    .map_err(|error| format!("Could not stream audio to Soniox: {error}"))?,
                LiveAudioMessage::Finish => {
                    writer
                        .send(Message::Text(String::new()))
                        .await
                        .map_err(|error| format!("Could not finish Soniox live stream: {error}"))?;
                    return Ok::<(), String>(());
                }
            }
        }
        writer
            .send(Message::Text(String::new()))
            .await
            .map_err(|error| format!("Could not finish Soniox live stream: {error}"))?;
        Ok::<(), String>(())
    };

    let receive_captions = async move {
        let mut final_tokens: Vec<Token> = Vec::new();
        while let Some(incoming) = reader.next().await {
            let incoming =
                incoming.map_err(|error| format!("Soniox live connection error: {error}"))?;
            let text = match incoming {
                Message::Text(value) => value.to_string(),
                Message::Binary(value) => String::from_utf8(value.to_vec())
                    .map_err(|_| "Soniox returned a non-UTF8 response".to_string())?,
                Message::Close(_) => return Ok::<(), String>(()),
                _ => continue,
            };
            let response: RealtimeResponse = serde_json::from_str(&text)
                .map_err(|error| format!("Could not decode Soniox live response: {error}"))?;
            if response.error_code.is_some() {
                let kind = response
                    .error_type
                    .unwrap_or_else(|| "realtime_error".into());
                let message = response
                    .error_message
                    .unwrap_or_else(|| "No details returned".into());
                let request = response
                    .request_id
                    .map(|id| format!(" Request ID: {id}"))
                    .unwrap_or_default();
                return Err(format!(
                    "Soniox live transcription failed ({kind}): {message}.{request}"
                ));
            }
            let mut non_final = Vec::new();
            for token in response.tokens {
                if token.is_final {
                    final_tokens.push(token);
                } else {
                    non_final.push(token);
                }
            }
            let mut display = final_tokens.clone();
            display.extend(non_final);
            emit_live(
                &app_handle,
                if response.finished {
                    "Live captions finished"
                } else {
                    "Live"
                },
                &render_tokens(&display),
                &render_tokens(&final_tokens),
                response.finished,
                None,
            );
            if response.finished {
                return Ok(());
            }
        }
        Ok(())
    };

    futures_util::future::try_join(send_audio, receive_captions)
        .await
        .map(|_| ())
}

pub fn emit_realtime_error(app_handle: &tauri::AppHandle, error: String) {
    emit_live(
        app_handle,
        "Live captions unavailable",
        "",
        "",
        true,
        Some(error),
    );
}

fn emit_live(
    app_handle: &tauri::AppHandle,
    status: &str,
    text: &str,
    final_text: &str,
    finished: bool,
    error: Option<String>,
) {
    let _ = app_handle.emit(
        "live-transcription",
        LiveTranscriptEvent {
            text: text.to_string(),
            final_text: final_text.to_string(),
            finished,
            status: status.to_string(),
            error,
        },
    );
}

fn normalize_language_hints(languages: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    languages
        .iter()
        .filter_map(|language| {
            let value = language.trim().split('-').next()?.to_lowercase();
            if value.is_empty() || !seen.insert(value.clone()) {
                None
            } else {
                Some(value)
            }
        })
        .collect()
}

fn parse_transcript(response: TranscriptResponse) -> TranscriptResult {
    let mut segments: Vec<TranscriptSegment> = Vec::new();
    let mut speakers = Vec::new();
    let mut seen_speakers = HashSet::new();
    for token in response.tokens {
        if token.text.is_empty() {
            continue;
        }
        let speaker = speaker_name(token.speaker.as_ref());
        if speaker != "unknown" && seen_speakers.insert(speaker.clone()) {
            speakers.push(speaker.clone());
        }
        let start_ms = token.start_ms.unwrap_or(0);
        let end_ms = token.end_ms.unwrap_or(start_ms);
        if let Some(previous) = segments.last_mut() {
            if previous.speaker == speaker {
                previous.text.push_str(&token.text);
                previous.end_ms = previous.end_ms.max(end_ms);
                continue;
            }
        }
        segments.push(TranscriptSegment {
            speaker,
            start_ms,
            end_ms,
            text: token.text,
        });
    }
    for segment in &mut segments {
        segment.text = clean_text(&segment.text);
    }
    segments.retain(|segment| !segment.text.is_empty());
    let transcript = if response.text.trim().is_empty() {
        segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        response.text.trim().to_string()
    };
    if segments.is_empty() && !transcript.is_empty() {
        segments.push(TranscriptSegment {
            speaker: "unknown".into(),
            start_ms: 0,
            end_ms: 0,
            text: transcript.clone(),
        });
    }
    TranscriptResult {
        transcript,
        speakers,
        segments,
    }
}

fn speaker_name(value: Option<&Value>) -> String {
    let raw = match value {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        _ => return "unknown".into(),
    };
    if raw.starts_with("speaker_") {
        raw
    } else {
        format!("speaker_{raw}")
    }
}

fn clean_text(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" :", ":")
        .replace(" ;", ";")
}

fn render_tokens(tokens: &[Token]) -> String {
    let mut rendered = String::new();
    let mut current_speaker = String::new();
    for token in tokens {
        if token.text.is_empty() {
            continue;
        }
        let speaker = speaker_name(token.speaker.as_ref());
        if speaker != current_speaker {
            if !rendered.is_empty() {
                rendered.push_str("\n\n");
            }
            rendered.push_str(&format!("{}: ", display_speaker(&speaker)));
            current_speaker = speaker;
        }
        rendered.push_str(&token.text);
    }
    rendered.trim().to_string()
}

fn display_speaker(speaker: &str) -> String {
    speaker
        .strip_prefix("speaker_")
        .map(|value| format!("Speaker {value}"))
        .unwrap_or_else(|| speaker.to_string())
}

fn decode_response<T: DeserializeOwned>(response: Response, operation: &str) -> Result<T, String> {
    let status = response.status();
    let body = response.text().map_err(|error| {
        format!("Could not read Soniox response while trying to {operation}: {error}")
    })?;
    if !status.is_success() {
        let detail = if body.len() > 1_000 {
            &body[..1_000]
        } else {
            &body
        };
        return Err(format!("Soniox could not {operation} ({status}): {detail}"));
    }
    serde_json::from_str(&body).map_err(|error| {
        format!("Could not decode Soniox response while trying to {operation}: {error}")
    })
}

fn delete_resource(client: &Client, api_key: &str, resource: &str, id: &str) -> Result<(), String> {
    let response = client
        .delete(format!("{REST_BASE}/{resource}/{id}"))
        .bearer_auth(api_key)
        .send()
        .map_err(|error| format!("Could not delete Soniox {resource}: {error}"))?;
    if response.status().is_success() || response.status().as_u16() == 404 {
        Ok(())
    } else {
        Err(format!(
            "Soniox returned {} while deleting {resource}",
            response.status()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn async_tokens_are_grouped_by_contiguous_speaker() {
        let response = TranscriptResponse {
            text: "Hello there. Hi.".into(),
            tokens: vec![
                Token {
                    text: "Hello".into(),
                    start_ms: Some(0),
                    end_ms: Some(100),
                    speaker: Some(json!("1")),
                    language: Some("en".into()),
                    is_final: true,
                },
                Token {
                    text: " there.".into(),
                    start_ms: Some(100),
                    end_ms: Some(300),
                    speaker: Some(json!("1")),
                    language: Some("en".into()),
                    is_final: true,
                },
                Token {
                    text: " Hi.".into(),
                    start_ms: Some(310),
                    end_ms: Some(500),
                    speaker: Some(json!(2)),
                    language: Some("en".into()),
                    is_final: true,
                },
            ],
        };
        let result = parse_transcript(response);
        assert_eq!(result.speakers, vec!["speaker_1", "speaker_2"]);
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].text, "Hello there.");
    }

    #[test]
    fn language_hints_are_deduplicated_and_normalized() {
        let hints = vec!["en-US".into(), "de-DE".into(), "en".into(), " ru ".into()];
        assert_eq!(normalize_language_hints(&hints), vec!["en", "de", "ru"]);
    }
}

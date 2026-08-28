use std::{
    collections::{BTreeSet, HashSet, VecDeque},
    path::Path,
    sync::Once,
    thread,
    time::{Duration, Instant},
};

use futures_util::{SinkExt, StreamExt};
use reqwest::blocking::{multipart, Client, Response};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{Emitter, Manager};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const REST_BASE: &str = "https://api.soniox.com/v1";
const REALTIME_URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";
const ASYNC_MODEL: &str = "stt-async-v5";
const REALTIME_MODEL: &str = "stt-rt-v5";
const REST_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const REST_REQUEST_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);
const TRANSCRIPTION_DEADLINE: Duration = Duration::from_secs(2 * 60 * 60);
static TLS_PROVIDER: Once = Once::new();

fn ensure_tls_provider() {
    TLS_PROVIDER.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

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

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
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
    source_language: Option<String>,
    #[serde(default)]
    translation_status: Option<String>,
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
    final_audio_proc_ms: u64,
    #[serde(default)]
    total_audio_proc_ms: u64,
    #[serde(default)]
    finished: bool,
    error_code: Option<u16>,
    error_type: Option<String>,
    error_message: Option<String>,
    request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveCaptionTranslation {
    pub text: String,
    pub source_language: String,
    pub is_final: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveCaptionSegment {
    pub id: String,
    pub source_text: String,
    pub source_language: Option<String>,
    pub source_final: bool,
    pub translation: Option<LiveCaptionTranslation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveCaptionTurn {
    pub id: String,
    pub sequence: u64,
    pub speaker: String,
    pub segments: Vec<LiveCaptionSegment>,
    #[serde(skip_serializing)]
    stream_epoch: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveCaptionMarker {
    pub id: String,
    pub after_sequence: Option<u64>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveTranscriptEvent {
    pub revision: u64,
    pub text: String,
    pub final_text: String,
    pub turns: Vec<LiveCaptionTurn>,
    pub markers: Vec<LiveCaptionMarker>,
    pub final_audio_proc_ms: u64,
    pub total_audio_proc_ms: u64,
    pub target_language: Option<String>,
    pub translation_warning: Option<String>,
    pub finished: bool,
    pub status: String,
    pub error: Option<String>,
}

impl LiveTranscriptEvent {
    pub fn idle() -> Self {
        Self {
            revision: 0,
            text: String::new(),
            final_text: String::new(),
            turns: Vec::new(),
            markers: Vec::new(),
            final_audio_proc_ms: 0,
            total_audio_proc_ms: 0,
            target_language: None,
            translation_warning: None,
            finished: false,
            status: "Live captions idle".into(),
            error: None,
        }
    }

    pub fn starting(enabled: bool) -> Self {
        let mut event = Self::idle();
        event.status = if enabled {
            "Starting live captions".into()
        } else {
            "Live captions disabled".into()
        };
        event
    }
}

#[derive(Debug)]
pub enum LiveAudioMessage {
    Audio(Vec<u8>),
    Reconfigure {
        revision: u64,
        language_hints: Vec<String>,
        expected_speakers: Option<u8>,
    },
    Finish,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeOptions {
    pub language_hints: Vec<String>,
    pub expected_speakers: Option<u8>,
    pub preferred_language: String,
    pub no_translation_languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LiveContextProgressEvent {
    stage: String,
    detail: String,
    revision: u64,
    language_hints: Vec<String>,
    expected_speakers: Option<u8>,
}

const RECONFIGURE_SILENCE_MS: u64 = 1_500;
const RECONFIGURE_FORCE_MS: u64 = 5_000;
const RECONFIGURE_SILENCE_RMS: f32 = 0.006;
const REALTIME_FINISH_TIMEOUT: Duration = Duration::from_secs(8);
const REALTIME_CONNECT_ATTEMPTS: usize = 3;

#[derive(Debug, Clone, Serialize)]
pub struct TranslationLanguage {
    pub code: &'static str,
    pub name: &'static str,
}

const TRANSLATION_LANGUAGES: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("sq", "Albanian"),
    ("ar", "Arabic"),
    ("az", "Azerbaijani"),
    ("eu", "Basque"),
    ("be", "Belarusian"),
    ("bn", "Bengali"),
    ("bs", "Bosnian"),
    ("bg", "Bulgarian"),
    ("ca", "Catalan"),
    ("zh", "Chinese"),
    ("hr", "Croatian"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("de", "German"),
    ("el", "Greek"),
    ("gu", "Gujarati"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("kn", "Kannada"),
    ("kk", "Kazakh"),
    ("ko", "Korean"),
    ("lv", "Latvian"),
    ("lt", "Lithuanian"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("ml", "Malayalam"),
    ("mr", "Marathi"),
    ("no", "Norwegian"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("pa", "Punjabi"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sr", "Serbian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("es", "Spanish"),
    ("sw", "Swahili"),
    ("sv", "Swedish"),
    ("tl", "Tagalog"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("vi", "Vietnamese"),
    ("cy", "Welsh"),
];

pub fn supported_translation_languages() -> Vec<TranslationLanguage> {
    TRANSLATION_LANGUAGES
        .iter()
        .map(|(code, name)| TranslationLanguage { code, name })
        .collect()
}

pub fn normalize_translation_language(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    let base = normalized.split('-').next()?;
    let base = if base == "jp" { "ja" } else { base };
    TRANSLATION_LANGUAGES
        .iter()
        .any(|(code, _)| *code == base)
        .then(|| base.to_string())
}

pub fn normalize_language_hint(value: &str) -> Option<String> {
    normalize_translation_language(value)
}

pub fn transcribe_file<F>(
    path: &Path,
    api_key: &str,
    language_hints: &[String],
    expected_speakers: Option<u8>,
    mut progress: F,
) -> Result<TranscriptResult, String>
where
    F: FnMut(&str, String),
{
    let client = Client::builder()
        .connect_timeout(REST_CONNECT_TIMEOUT)
        .timeout(REST_REQUEST_TIMEOUT)
        .user_agent("Recall/0.1")
        .build()
        .map_err(|error| format!("Could not create the STT provider client: {error}"))?;
    let mut file_id: Option<String> = None;
    let mut transcription_id: Option<String> = None;

    let result = (|| {
        progress("stt:upload:start", "Uploading recording".into());
        if path
            .metadata()
            .map_err(|error| format!("Could not inspect the recording: {error}"))?
            .len()
            == 0
        {
            return Err("The recording is empty".into());
        }
        let filename = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("recording.wav")
            .to_string();
        let file_part = multipart::Part::file(path)
            .map_err(|error| format!("Could not open the recording for upload: {error}"))?
            .file_name(filename);
        let upload = client
            .post(format!("{REST_BASE}/files"))
            .bearer_auth(api_key)
            .multipart(multipart::Form::new().part("file", file_part))
            .send()
            .map_err(|error| format!("STT provider upload failed: {error}"))?;
        let uploaded: IdResponse = decode_response(upload, "upload recording")?;
        file_id = Some(uploaded.id.clone());
        progress("stt:upload:done", "Recording uploaded".into());

        progress(
            "stt:transcription:start",
            format!("Starting final transcription with {ASYNC_MODEL}"),
        );
        let hints = normalize_language_hints(language_hints);
        let mut payload = json!({
            "model": ASYNC_MODEL,
            "file_id": uploaded.id,
            "enable_speaker_diarization": true,
            "enable_language_identification": true,
            "context": meeting_context(&hints, expected_speakers),
        });
        if !hints.is_empty() {
            payload["language_hints"] = json!(hints);
            payload["language_hints_strict"] = json!(false);
        }
        let create = client
            .post(format!("{REST_BASE}/transcriptions"))
            .bearer_auth(api_key)
            .json(&payload)
            .send()
            .map_err(|error| format!("Could not start STT transcription: {error}"))?;
        let created: IdResponse = decode_response(create, "start transcription")?;
        transcription_id = Some(created.id.clone());
        progress(
            "stt:transcription:waiting",
            "The STT provider is processing the recording".into(),
        );

        let deadline = Instant::now() + TRANSCRIPTION_DEADLINE;
        let mut previous_status = String::new();
        loop {
            if Instant::now() >= deadline {
                return Err("STT transcription timed out after 2 hours".into());
            }
            let response = client
                .get(format!("{REST_BASE}/transcriptions/{}", created.id))
                .bearer_auth(api_key)
                .send()
                .map_err(|error| format!("Could not poll the STT transcription: {error}"))?;
            let status: StatusResponse = decode_response(response, "poll transcription")?;
            if status.status != previous_status {
                previous_status = status.status.clone();
                progress(
                    "stt:transcription:status",
                    format!("STT provider status: {}", status.status),
                );
            }
            match status.status.as_str() {
                "completed" => break,
                "error" | "failed" => {
                    let kind = status.error_type.unwrap_or_else(|| "unknown_error".into());
                    let message = status
                        .error_message
                        .unwrap_or_else(|| "No details returned".into());
                    return Err(format!("STT transcription failed ({kind}): {message}"));
                }
                _ => thread::sleep(Duration::from_secs(2)),
            }
        }

        progress(
            "stt:transcript:download:start",
            "Downloading final diarized transcript".into(),
        );
        let response = client
            .get(format!(
                "{REST_BASE}/transcriptions/{}/transcript",
                created.id
            ))
            .bearer_auth(api_key)
            .send()
            .map_err(|error| format!("Could not download the STT transcript: {error}"))?;
        let transcript: TranscriptResponse = decode_response(response, "download transcript")?;
        let parsed = parse_transcript(transcript);
        progress(
            "stt:transcript:download:done",
            format!(
                "Downloaded {} interventions from {} diarized speakers",
                parsed.segments.len(),
                parsed.speakers.len()
            ),
        );
        Ok(parsed)
    })();

    progress(
        "stt:cleanup:start",
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
        progress("stt:cleanup:done", "Provider-side artifacts removed".into());
    } else {
        progress(
            "stt:cleanup:warning",
            format!("Cleanup incomplete: {}", cleanup_errors.join("; ")),
        );
    }
    result
}

type RealtimeSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Debug)]
struct SilenceDetector {
    sample_rate: u32,
    silent_samples: u64,
}

#[derive(Debug)]
struct PendingRealtimeOptions {
    options: RealtimeOptions,
    revision: u64,
    requested_at: Instant,
}

fn context_restart_decision(silence_ready: bool, pending_for: Duration) -> (bool, bool) {
    let restart = silence_ready || pending_for >= Duration::from_millis(RECONFIGURE_FORCE_MS);
    (restart, restart && !silence_ready)
}

impl SilenceDetector {
    fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            silent_samples: 0,
        }
    }

    fn observe(&mut self, bytes: &[u8]) -> bool {
        let sample_count = bytes.len() / 2;
        if sample_count == 0 || self.sample_rate == 0 {
            return self.ready();
        }
        let sum_squares = bytes
            .chunks_exact(2)
            .map(|sample| {
                let value = i16::from_le_bytes([sample[0], sample[1]]) as f32 / i16::MAX as f32;
                value * value
            })
            .sum::<f32>();
        let rms = (sum_squares / sample_count as f32).sqrt();
        if rms <= RECONFIGURE_SILENCE_RMS {
            self.silent_samples = self.silent_samples.saturating_add(sample_count as u64);
        } else {
            self.silent_samples = 0;
        }
        self.ready()
    }

    fn ready(&self) -> bool {
        self.silent_samples.saturating_mul(1_000)
            >= self.sample_rate as u64 * RECONFIGURE_SILENCE_MS
    }

    fn reset(&mut self) {
        self.silent_samples = 0;
    }
}

#[derive(Debug, Default)]
struct LiveTranscriptAccumulator {
    final_tokens: Vec<Token>,
    provisional_tokens: Vec<Token>,
    markers: Vec<LiveCaptionMarker>,
    audio_offset_ms: u64,
    epoch_final_audio_ms: u64,
    epoch_total_audio_ms: u64,
    last_diagnostic_audio_ms: u64,
    stream_epoch: u32,
    received_response: bool,
}

impl LiveTranscriptAccumulator {
    fn excluded_languages(
        options: &RealtimeOptions,
        translation_target: Option<&str>,
    ) -> HashSet<String> {
        options
            .no_translation_languages
            .iter()
            .filter_map(|language| normalize_translation_language(language))
            .chain(translation_target.map(str::to_string))
            .collect()
    }

    fn event(
        &self,
        options: &RealtimeOptions,
        status: &str,
        finished: bool,
    ) -> LiveTranscriptEvent {
        let (translation_target, translation_warning) =
            live_translation_policy(&options.preferred_language);
        let display = display_tokens(&self.final_tokens, &self.provisional_tokens);
        let excluded_languages = Self::excluded_languages(options, translation_target.as_deref());
        LiveTranscriptEvent {
            revision: 0,
            text: render_original_tokens(&display),
            final_text: render_original_tokens(&self.final_tokens),
            turns: build_live_turns(&display, &excluded_languages),
            markers: self.markers.clone(),
            final_audio_proc_ms: self.audio_offset_ms + self.epoch_final_audio_ms,
            total_audio_proc_ms: self.audio_offset_ms + self.epoch_total_audio_ms,
            target_language: translation_target,
            translation_warning,
            finished,
            status: status.into(),
            error: None,
        }
    }

    fn emit_status(
        &self,
        app_handle: &tauri::AppHandle,
        options: &RealtimeOptions,
        status: &str,
        finished: bool,
    ) -> u64 {
        emit_live(app_handle, self.event(options, status, finished))
    }

    fn apply_response(
        &mut self,
        response: RealtimeResponse,
        options: &RealtimeOptions,
        status: &str,
        terminal: bool,
        app_handle: &tauri::AppHandle,
    ) -> Result<bool, String> {
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
            return Err(format!("Live STT failed ({kind}): {message}.{request}"));
        }
        if !self.received_response {
            self.received_response = true;
            eprintln!("[live] receiving caption updates");
        }
        let (new_final, non_final): (Vec<_>, Vec<_>) = response
            .tokens
            .into_iter()
            .partition(|token| token.is_final);
        let new_final_count = new_final.len();
        let non_final_count = non_final.len();
        append_final_tokens(&mut self.final_tokens, new_final);
        self.provisional_tokens = non_final;
        self.epoch_final_audio_ms = response.final_audio_proc_ms;
        self.epoch_total_audio_ms = response.total_audio_proc_ms;

        let (translation_target, _) = live_translation_policy(&options.preferred_language);
        let excluded_languages = Self::excluded_languages(options, translation_target.as_deref());
        let display = display_tokens(&self.final_tokens, &self.provisional_tokens);
        let source_token_count = display
            .iter()
            .filter(|token| is_source_token(token))
            .count();
        let translation_token_count = display
            .iter()
            .filter(|token| is_translation_token(token))
            .count();
        let speaker_labels = display_diagnostic_values(
            display
                .iter()
                .filter(|token| is_source_token(token))
                .map(|token| speaker_name(token.speaker.as_ref())),
        );
        let source_languages = display_diagnostic_values(
            display
                .iter()
                .filter(|token| is_source_token(token))
                .filter_map(normalized_source_language),
        );
        let turns = build_live_turns(&display, &excluded_languages);
        let turn_count = turns.len();
        let segment_count = turns.iter().map(|turn| turn.segments.len()).sum::<usize>();
        let global_audio_ms = self.audio_offset_ms + response.total_audio_proc_ms;
        let should_log_snapshot = response.finished
            || global_audio_ms >= self.last_diagnostic_audio_ms.saturating_add(5_000);
        let finished = terminal && response.finished;
        let revision = self.emit_status(
            app_handle,
            options,
            if finished {
                "Live captions finished"
            } else {
                status
            },
            finished,
        );
        if should_log_snapshot {
            eprintln!(
                "[live] snapshot revision={revision} epoch={} final_audio_ms={} total_audio_ms={} accumulated_final_tokens={} new_final_tokens={new_final_count} provisional_tokens={non_final_count} source_tokens={source_token_count} translation_tokens={translation_token_count} speaker_labels={speaker_labels} source_languages={source_languages} turns={} segments={segment_count}",
                self.stream_epoch,
                self.audio_offset_ms + response.final_audio_proc_ms,
                global_audio_ms,
                self.final_tokens.len(),
                turn_count,
            );
            self.last_diagnostic_audio_ms = global_audio_ms;
        }
        Ok(response.finished)
    }

    fn start_next_epoch(&mut self, options: &RealtimeOptions) {
        self.provisional_tokens.clear();
        let (translation_target, _) = live_translation_policy(&options.preferred_language);
        let turns = build_live_turns(
            &self.final_tokens,
            &Self::excluded_languages(options, translation_target.as_deref()),
        );
        let after_sequence = turns.last().map(|turn| turn.sequence);
        let normalized_languages = normalize_language_hints(&options.language_hints);
        let languages = if normalized_languages.is_empty() {
            "none specified".into()
        } else {
            normalized_languages.join(", ")
        };
        let speakers = options
            .expected_speakers
            .map(|count| format!("{count} expected speakers"))
            .unwrap_or_else(|| "speaker count open".into());
        self.markers.push(LiveCaptionMarker {
            id: format!("live-restart-{}", self.stream_epoch + 1),
            after_sequence,
            text: format!(
                "Live captions restarted after a pause · {speakers} · likely languages: {languages}"
            ),
        });
        self.final_tokens.push(Token {
            translation_status: Some("recall_stream_boundary".into()),
            is_final: true,
            ..Token::default()
        });
        self.audio_offset_ms = self
            .audio_offset_ms
            .saturating_add(self.epoch_total_audio_ms);
        self.epoch_final_audio_ms = 0;
        self.epoch_total_audio_ms = 0;
        self.stream_epoch = self.stream_epoch.saturating_add(1);
        self.received_response = false;
    }
}

fn emit_live_context_progress(
    app_handle: &tauri::AppHandle,
    stage: &str,
    detail: impl Into<String>,
    revision: u64,
    options: &RealtimeOptions,
) {
    let detail = detail.into();
    let language_hints = normalize_language_hints(&options.language_hints);
    eprintln!("[live context r{revision}] {stage}: {detail}");
    let _ = app_handle.emit(
        "live-context:progress",
        LiveContextProgressEvent {
            stage: stage.into(),
            detail,
            revision,
            language_hints,
            expected_speakers: options.expected_speakers,
        },
    );
}

fn live_context_detail(prefix: &str, revision: u64, options: &RealtimeOptions) -> String {
    let languages = display_diagnostic_values(normalize_language_hints(&options.language_hints));
    let speakers = options
        .expected_speakers
        .map(|count| count.to_string())
        .unwrap_or_else(|| "open".into());
    format!(
        "{prefix} (revision {revision}) - likely languages: {languages}; expected speakers: {speakers}"
    )
}

fn decode_realtime_response(message: Message) -> Result<Option<RealtimeResponse>, String> {
    let text = match message {
        Message::Text(value) => value.to_string(),
        Message::Binary(value) => String::from_utf8(value.to_vec())
            .map_err(|_| "The STT provider returned a non-UTF8 response".to_string())?,
        Message::Close(_) => return Ok(None),
        _ => return Ok(None),
    };
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|error| format!("Could not decode the live STT response: {error}"))
}

async fn connect_realtime_socket(
    api_key: &str,
    options: &RealtimeOptions,
    sample_rate: u32,
) -> Result<RealtimeSocket, String> {
    let (translation_target, _) = live_translation_policy(&options.preferred_language);
    let mut last_error = String::new();
    for attempt in 1..=REALTIME_CONNECT_ATTEMPTS {
        eprintln!("[live] connecting to the realtime STT provider attempt={attempt}");
        match tokio::time::timeout(Duration::from_secs(10), connect_async(REALTIME_URL)).await {
            Ok(Ok((mut socket, _))) => {
                let config = realtime_config(
                    api_key,
                    &options.language_hints,
                    options.expected_speakers,
                    translation_target.as_deref(),
                    sample_rate,
                );
                socket
                    .send(Message::Text(config.to_string()))
                    .await
                    .map_err(|error| format!("Could not configure live STT: {error}"))?;
                eprintln!(
                    "[live] connected and configured likely_languages={} expected_speakers={} translation_target={}",
                    display_diagnostic_values(normalize_language_hints(&options.language_hints)),
                    options
                        .expected_speakers
                        .map(|count| count.to_string())
                        .unwrap_or_else(|| "unspecified".into()),
                    translation_target.as_deref().unwrap_or("none"),
                );
                return Ok(socket);
            }
            Ok(Err(error)) => last_error = format!("Could not connect to live STT: {error}"),
            Err(_) => last_error = "Timed out connecting to live STT after 10 seconds".into(),
        }
        if attempt < REALTIME_CONNECT_ATTEMPTS {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }
    Err(last_error)
}

async fn finish_realtime_epoch(
    socket: &mut RealtimeSocket,
    accumulator: &mut LiveTranscriptAccumulator,
    options: &RealtimeOptions,
    terminal: bool,
    app_handle: &tauri::AppHandle,
) -> Result<(), String> {
    socket
        .send(Message::Text(String::new()))
        .await
        .map_err(|error| format!("Could not finish the live STT stream: {error}"))?;
    let deadline = tokio::time::Instant::now() + REALTIME_FINISH_TIMEOUT;
    loop {
        let incoming = match tokio::time::timeout_at(deadline, socket.next()).await {
            Ok(incoming) => incoming,
            Err(_) => {
                eprintln!("[live] timed out waiting for the realtime stream to finish");
                return Ok(());
            }
        };
        let Some(incoming) = incoming else {
            return Ok(());
        };
        let message = incoming.map_err(|error| format!("Live STT connection error: {error}"))?;
        let closed = matches!(message, Message::Close(_));
        if let Some(response) = decode_realtime_response(message)? {
            if accumulator.apply_response(
                response,
                options,
                if terminal {
                    "Finishing live captions"
                } else {
                    "Applying updated meeting context"
                },
                terminal,
                app_handle,
            )? {
                return Ok(());
            }
        }
        if closed {
            return Ok(());
        }
    }
}

pub async fn run_realtime(
    api_key: String,
    mut options: RealtimeOptions,
    sample_rate: u32,
    mut audio_rx: mpsc::UnboundedReceiver<LiveAudioMessage>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    ensure_tls_provider();
    let mut accumulator = LiveTranscriptAccumulator::default();
    accumulator.emit_status(&app_handle, &options, "Connecting live captions", false);
    let mut socket = connect_realtime_socket(&api_key, &options, sample_rate).await?;
    accumulator.emit_status(&app_handle, &options, "Live captions connected", false);
    let mut detector = SilenceDetector::new(sample_rate);
    let mut pending_options: Option<PendingRealtimeOptions> = None;
    let mut active_revision = 0u64;
    let mut sent_audio = false;

    loop {
        enum RealtimeLoopEvent {
            Audio(Option<LiveAudioMessage>),
            Provider(Option<Result<Message, tokio_tungstenite::tungstenite::Error>>),
        }
        let event = tokio::select! {
            message = audio_rx.recv() => RealtimeLoopEvent::Audio(message),
            incoming = socket.next() => RealtimeLoopEvent::Provider(incoming),
        };
        let mut restart_now = false;
        let mut restart_forced = false;
        match event {
            RealtimeLoopEvent::Audio(Some(LiveAudioMessage::Audio(bytes))) => {
                let silence_ready = detector.observe(&bytes);
                socket.send(Message::Binary(bytes)).await.map_err(|error| {
                    format!("Could not stream audio to the STT provider: {error}")
                })?;
                if !sent_audio {
                    sent_audio = true;
                    eprintln!("[live] streaming microphone audio");
                }
                if let Some(pending) = pending_options.as_ref() {
                    (restart_now, restart_forced) =
                        context_restart_decision(silence_ready, pending.requested_at.elapsed());
                }
            }
            RealtimeLoopEvent::Audio(Some(LiveAudioMessage::Reconfigure {
                revision,
                language_hints,
                expected_speakers,
            })) => {
                if revision <= active_revision
                    || pending_options
                        .as_ref()
                        .is_some_and(|pending| revision <= pending.revision)
                {
                    continue;
                }
                let mut requested = options.clone();
                requested.language_hints = language_hints;
                requested.expected_speakers = expected_speakers;
                if requested == options {
                    pending_options = None;
                    active_revision = revision;
                    emit_live_context_progress(
                        &app_handle,
                        "sent",
                        live_context_detail(
                            "Context already active; nothing new sent to STT",
                            revision,
                            &requested,
                        ),
                        revision,
                        &requested,
                    );
                    accumulator.emit_status(&app_handle, &options, "Live", false);
                } else {
                    pending_options = Some(PendingRealtimeOptions {
                        options: requested.clone(),
                        revision,
                        requested_at: Instant::now(),
                    });
                    emit_live_context_progress(
                        &app_handle,
                        "pending",
                        live_context_detail(
                            "Pending - waiting up to 5 seconds for a quiet restart",
                            revision,
                            &requested,
                        ),
                        revision,
                        &requested,
                    );
                    accumulator.emit_status(
                        &app_handle,
                        &options,
                        "Pending STT context update",
                        false,
                    );
                    restart_now = detector.ready();
                }
            }
            RealtimeLoopEvent::Audio(Some(LiveAudioMessage::Finish))
            | RealtimeLoopEvent::Audio(None) => {
                finish_realtime_epoch(&mut socket, &mut accumulator, &options, true, &app_handle)
                    .await?;
                eprintln!("[live] caption stream finished");
                return Ok(());
            }
            RealtimeLoopEvent::Provider(Some(Ok(message))) => {
                let closed = matches!(message, Message::Close(_));
                if let Some(response) = decode_realtime_response(message)? {
                    let provider_finished = accumulator.apply_response(
                        response,
                        &options,
                        "Live",
                        false,
                        &app_handle,
                    )?;
                    if provider_finished && pending_options.is_none() {
                        return Err("Live STT ended before the recording stopped".into());
                    }
                }
                if closed {
                    return Err("Live STT disconnected before the recording stopped".into());
                }
            }
            RealtimeLoopEvent::Provider(Some(Err(error))) => {
                return Err(format!("Live STT connection error: {error}"));
            }
            RealtimeLoopEvent::Provider(None) => {
                return Err("Live STT disconnected before the recording stopped".into());
            }
        }

        if !restart_now {
            continue;
        }
        let pending = pending_options
            .as_ref()
            .expect("restart has pending options");
        let restart_reason = if restart_forced {
            format!("Forced after {RECONFIGURE_FORCE_MS} ms without a quiet pause")
        } else {
            format!("Detected at least {RECONFIGURE_SILENCE_MS} ms of quiet audio")
        };
        emit_live_context_progress(
            &app_handle,
            "sending",
            live_context_detail(
                &format!("Sending to STT - {restart_reason}"),
                pending.revision,
                &pending.options,
            ),
            pending.revision,
            &pending.options,
        );
        accumulator.emit_status(
            &app_handle,
            &options,
            "Restarting live captions at a quiet pause",
            false,
        );
        if let Err(error) =
            finish_realtime_epoch(&mut socket, &mut accumulator, &options, false, &app_handle).await
        {
            let pending = pending_options
                .as_ref()
                .expect("restart has pending options");
            emit_live_context_progress(
                &app_handle,
                "failed",
                &error,
                pending.revision,
                &pending.options,
            );
            return Err(error);
        }

        let mut next_options = pending_options.take().expect("restart has pending options");
        let mut buffered_audio = VecDeque::new();
        let mut finish_requested = false;
        while let Ok(message) = audio_rx.try_recv() {
            match message {
                LiveAudioMessage::Audio(bytes) => buffered_audio.push_back(bytes),
                LiveAudioMessage::Reconfigure {
                    revision,
                    language_hints,
                    expected_speakers,
                } => {
                    if revision > next_options.revision {
                        next_options.options.language_hints = language_hints;
                        next_options.options.expected_speakers = expected_speakers;
                        next_options.revision = revision;
                        next_options.requested_at = Instant::now();
                    }
                }
                LiveAudioMessage::Finish => finish_requested = true,
            }
        }

        options = next_options.options;
        active_revision = next_options.revision;
        accumulator.start_next_epoch(&options);
        accumulator.emit_status(&app_handle, &options, "Reconnecting live captions", false);
        emit_live_context_progress(
            &app_handle,
            "sending",
            live_context_detail(
                &format!(
                    "Sending to STT; buffered {} audio chunk(s)",
                    buffered_audio.len()
                ),
                active_revision,
                &options,
            ),
            active_revision,
            &options,
        );
        socket = match connect_realtime_socket(&api_key, &options, sample_rate).await {
            Ok(socket) => socket,
            Err(error) => {
                emit_live_context_progress(
                    &app_handle,
                    "failed",
                    &error,
                    active_revision,
                    &options,
                );
                return Err(error);
            }
        };
        detector.reset();
        while let Some(bytes) = buffered_audio.pop_front() {
            detector.observe(&bytes);
            if let Err(error) = socket.send(Message::Binary(bytes)).await {
                let error = format!("Could not flush buffered live audio: {error}");
                emit_live_context_progress(
                    &app_handle,
                    "failed",
                    &error,
                    active_revision,
                    &options,
                );
                return Err(error);
            }
        }
        accumulator.emit_status(&app_handle, &options, "Live", false);
        emit_live_context_progress(
            &app_handle,
            "sent",
            live_context_detail(
                "Sent to STT; live captions resumed",
                active_revision,
                &options,
            ),
            active_revision,
            &options,
        );
        if finish_requested {
            finish_realtime_epoch(&mut socket, &mut accumulator, &options, true, &app_handle)
                .await?;
            return Ok(());
        }
    }
}

fn realtime_config(
    api_key: &str,
    language_hints: &[String],
    expected_speakers: Option<u8>,
    translation_target: Option<&str>,
    sample_rate: u32,
) -> Value {
    let hints = normalize_language_hints(language_hints);
    let mut config = json!({
        "api_key": api_key,
        "model": REALTIME_MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": sample_rate,
        "num_channels": 1,
        "enable_speaker_diarization": true,
        "enable_language_identification": true,
        "enable_endpoint_detection": false,
        "context": meeting_context(&hints, expected_speakers),
    });
    if let Some(target_language) = translation_target {
        config["translation"] = json!({
            "type": "one_way",
            "target_language": target_language,
        });
    }
    if !hints.is_empty() {
        config["language_hints"] = json!(hints);
        config["language_hints_strict"] = json!(false);
    }
    config
}

fn meeting_context(language_hints: &[String], expected_speakers: Option<u8>) -> Value {
    let hints = normalize_language_hints(language_hints);
    let language_context = if hints.is_empty() {
        "Languages are not known in advance and may change within the conversation.".to_string()
    } else {
        format!(
            "Likely spoken language codes: {}. Speakers may code-switch; other languages may still occur.",
            hints.join(", ")
        )
    };
    let speaker_context = match expected_speakers {
        Some(1) => "1 speaker is expected. Keep that voice under one stable speaker label, including across language changes.".to_string(),
        Some(count) => format!(
            "{count} speakers are expected. Keep each distinct voice under a stable speaker label, change labels when the person speaking changes, and do not split one speaker merely because they change language."
        ),
        None => "One or more speakers may participate. Keep each distinct voice under a stable speaker label, change labels when the person speaking changes, and do not split one speaker merely because they change language.".to_string(),
    };
    json!({
        "general": [
            {
                "key": "setting",
                "value": "Multilingual meeting recording"
            },
            {
                "key": "languages",
                "value": language_context
            },
            {
                "key": "instructions",
                "value": "Transcribe each passage in the language actually spoken and preserve its script in the source transcript. Re-evaluate the language promptly after a code-switch. Do not convert or transliterate speech into a different language in the source transcript."
            },
            {
                "key": "speakers",
                "value": speaker_context
            }
        ]
    })
}

fn display_diagnostic_values<I>(values: I) -> String
where
    I: IntoIterator<Item = String>,
{
    let values = values.into_iter().collect::<BTreeSet<_>>();
    if values.is_empty() {
        "none".into()
    } else {
        values.into_iter().collect::<Vec<_>>().join(",")
    }
}

fn live_translation_policy(preferred_language: &str) -> (Option<String>, Option<String>) {
    match normalize_translation_language(preferred_language) {
        Some(language) => (Some(language), None),
        None => (
            None,
            Some(format!(
                "Preferred language {} is unavailable for live STT translation. Original live captions will continue.",
                preferred_language.trim()
            )),
        ),
    }
}

pub fn emit_realtime_error(app_handle: &tauri::AppHandle, error: String) {
    let state = app_handle.state::<crate::state::AppState>();
    let mut payload = state
        .live_transcript
        .lock()
        .map(|snapshot| snapshot.clone())
        .unwrap_or_else(|_| LiveTranscriptEvent::idle());
    payload.finished = true;
    payload.status = "Live captions unavailable".into();
    payload.error = Some(error);
    emit_live(app_handle, payload);
}

fn emit_live(app_handle: &tauri::AppHandle, mut payload: LiveTranscriptEvent) -> u64 {
    let state = app_handle.state::<crate::state::AppState>();
    payload.revision = state.next_live_transcript_revision();
    let revision = payload.revision;
    if let Ok(mut snapshot) = state.live_transcript.lock() {
        *snapshot = payload.clone();
    }
    let _ = app_handle.emit("live-transcription", payload);
    revision
}

fn normalize_language_hints(languages: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    languages
        .iter()
        .filter_map(|language| {
            let value = normalize_language_hint(language)?;
            if !seen.insert(value.clone()) {
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

fn render_original_tokens(tokens: &[Token]) -> String {
    let mut rendered = String::new();
    let mut current_speaker = String::new();
    for token in tokens.iter() {
        if is_stream_boundary(token) {
            current_speaker.clear();
            continue;
        }
        if token.translation_status.as_deref() == Some("translation") {
            continue;
        }
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

#[derive(Debug)]
struct SourceRun {
    start_index: usize,
    stream_epoch: u32,
    speaker: String,
    source_language: Option<String>,
    is_final: bool,
    text: String,
}

#[derive(Debug)]
struct TranslationRun {
    start_index: usize,
    last_index: usize,
    stream_epoch: u32,
    source_language: Option<String>,
    is_final: bool,
    text: String,
}

fn is_translation_token(token: &Token) -> bool {
    token.translation_status.as_deref() == Some("translation")
}

fn is_stream_boundary(token: &Token) -> bool {
    token.translation_status.as_deref() == Some("recall_stream_boundary")
}

fn is_source_token(token: &Token) -> bool {
    matches!(
        token.translation_status.as_deref(),
        None | Some("none") | Some("original")
    )
}

fn normalized_source_language(token: &Token) -> Option<String> {
    token
        .language
        .as_deref()
        .or(token.source_language.as_deref())
        .and_then(normalize_translation_language)
}

fn normalized_translation_source_language(token: &Token) -> Option<String> {
    token
        .source_language
        .as_deref()
        .and_then(normalize_translation_language)
}

/// A position map deliberately remains private to the native formatter. Soniox
/// translation tokens carry a source language, but no timestamps or source
/// token references, so the only safe association is provider order plus the
/// closest preceding source segment with the same normalized language.
#[derive(Debug)]
struct SourceSegmentPosition {
    start_index: usize,
    stream_epoch: u32,
    source_language: Option<String>,
    turn_index: usize,
    segment_index: usize,
}

fn build_live_turns(
    tokens: &[Token],
    excluded_languages: &HashSet<String>,
) -> Vec<LiveCaptionTurn> {
    let source_runs = collect_source_runs(tokens);
    let translation_runs = collect_translation_runs(tokens);
    let mut turns = Vec::<LiveCaptionTurn>::new();
    let mut positions = Vec::<SourceSegmentPosition>::new();

    for run in source_runs {
        let turn_index = match turns.last() {
            Some(turn)
                if turn.speaker == display_speaker(&run.speaker)
                    && turn.stream_epoch == run.stream_epoch =>
            {
                turns.len() - 1
            }
            _ => {
                let sequence = turns.len() as u64;
                turns.push(LiveCaptionTurn {
                    id: format!("live-turn-{sequence}"),
                    sequence,
                    speaker: display_speaker(&run.speaker),
                    segments: Vec::new(),
                    stream_epoch: run.stream_epoch,
                });
                turns.len() - 1
            }
        };
        let segment_index = turns[turn_index].segments.len();
        turns[turn_index].segments.push(LiveCaptionSegment {
            id: format!("live-turn-{turn_index}-segment-{segment_index}"),
            source_text: run.text,
            source_language: run.source_language.clone(),
            source_final: run.is_final,
            translation: None,
        });
        positions.push(SourceSegmentPosition {
            start_index: run.start_index,
            stream_epoch: run.stream_epoch,
            source_language: run.source_language,
            turn_index,
            segment_index,
        });
    }

    for translation in translation_runs {
        let Some(source_language) = translation.source_language else {
            eprintln!("[live] omitting translation without a valid source language");
            continue;
        };
        if excluded_languages.contains(&source_language) {
            eprintln!(
                "[live] omitting translation from excluded source language {source_language}"
            );
            continue;
        }
        let Some(position) = positions.iter().rev().find(|position| {
            position.start_index < translation.start_index
                && position.stream_epoch == translation.stream_epoch
                && position.source_language.as_deref() == Some(source_language.as_str())
        }) else {
            eprintln!(
                "[live] omitting unmatched translation for source language {source_language}"
            );
            continue;
        };
        let segment = &mut turns[position.turn_index].segments[position.segment_index];
        match &mut segment.translation {
            Some(existing) => {
                // A provider translation chunk can have a different token count
                // from its source. Consecutive compatible chunks therefore extend
                // the preceding segment instead of attempting word alignment.
                existing.text.push_str(&translation.text);
                existing.is_final &= translation.is_final;
            }
            None => {
                segment.translation = Some(LiveCaptionTranslation {
                    text: translation.text,
                    source_language,
                    is_final: translation.is_final,
                });
            }
        }
    }
    turns
}

fn collect_source_runs(tokens: &[Token]) -> Vec<SourceRun> {
    let mut runs = Vec::new();
    let mut stream_epoch = 0_u32;
    for (index, token) in tokens.iter().enumerate() {
        if is_stream_boundary(token) {
            stream_epoch = stream_epoch.saturating_add(1);
            continue;
        }
        if token.text.is_empty() {
            continue;
        }
        if !is_source_token(token) {
            if !is_translation_token(token) {
                eprintln!("[live] omitting token with unknown translation status");
            }
            continue;
        }
        let speaker = speaker_name(token.speaker.as_ref());
        let source_language = normalized_source_language(token);
        // Translation tokens can be interleaved with their originals. They are
        // not source boundaries: only a speaker or language change creates a
        // new source segment.
        let should_extend = runs.last().is_some_and(|run: &SourceRun| {
            run.stream_epoch == stream_epoch
                && run.speaker == speaker
                && run.source_language == source_language
        });
        if should_extend {
            let run = runs.last_mut().expect("source run exists");
            run.is_final &= token.is_final;
            run.text.push_str(&token.text);
        } else {
            runs.push(SourceRun {
                start_index: index,
                stream_epoch,
                speaker,
                source_language,
                is_final: token.is_final,
                text: token.text.clone(),
            });
        }
    }
    runs
}

fn collect_translation_runs(tokens: &[Token]) -> Vec<TranslationRun> {
    let mut runs = Vec::new();
    let mut stream_epoch = 0_u32;
    for (index, token) in tokens.iter().enumerate() {
        if is_stream_boundary(token) {
            stream_epoch = stream_epoch.saturating_add(1);
            continue;
        }
        if !is_translation_token(token) || token.text.is_empty() {
            continue;
        }
        let source_language = normalized_translation_source_language(token);
        let should_extend = runs.last().is_some_and(|run: &TranslationRun| {
            run.stream_epoch == stream_epoch
                && run.last_index + 1 == index
                && run.source_language == source_language
        });
        if should_extend {
            let run = runs.last_mut().expect("translation run exists");
            run.last_index = index;
            run.is_final &= token.is_final;
            run.text.push_str(&token.text);
        } else {
            runs.push(TranslationRun {
                start_index: index,
                last_index: index,
                stream_epoch,
                source_language,
                is_final: token.is_final,
                text: token.text.clone(),
            });
        }
    }
    runs
}

fn append_final_tokens(final_tokens: &mut Vec<Token>, incoming: Vec<Token>) {
    // The realtime provider emits finalized tokens exactly once. Preserve every
    // token in arrival order, including repeated words with identical metadata.
    final_tokens.extend(incoming);
}

fn display_tokens(final_tokens: &[Token], non_final: &[Token]) -> Vec<Token> {
    final_tokens
        .iter()
        .cloned()
        .chain(non_final.iter().cloned())
        .collect()
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
        format!("Could not read the STT provider response while trying to {operation}: {error}")
    })?;
    if !status.is_success() {
        let detail = if body.len() > 1_000 {
            &body[..1_000]
        } else {
            &body
        };
        return Err(format!(
            "The STT provider could not {operation} ({status}): {detail}"
        ));
    }
    serde_json::from_str(&body).map_err(|error| {
        format!("Could not decode the STT provider response while trying to {operation}: {error}")
    })
}

fn delete_resource(client: &Client, api_key: &str, resource: &str, id: &str) -> Result<(), String> {
    let response = client
        .delete(format!("{REST_BASE}/{resource}/{id}"))
        .bearer_auth(api_key)
        .send()
        .map_err(|error| format!("Could not delete STT provider {resource}: {error}"))?;
    if response.status().is_success() || response.status().as_u16() == 404 {
        Ok(())
    } else {
        Err(format!(
            "The STT provider returned {} while deleting {resource}",
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
                    source_language: None,
                    translation_status: None,
                    is_final: true,
                },
                Token {
                    text: " there.".into(),
                    start_ms: Some(100),
                    end_ms: Some(300),
                    speaker: Some(json!("1")),
                    language: Some("en".into()),
                    source_language: None,
                    translation_status: None,
                    is_final: true,
                },
                Token {
                    text: " Hi.".into(),
                    start_ms: Some(310),
                    end_ms: Some(500),
                    speaker: Some(json!(2)),
                    language: Some("en".into()),
                    source_language: None,
                    translation_status: None,
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

    #[test]
    fn live_snapshot_starts_in_the_requested_mode() {
        assert_eq!(
            LiveTranscriptEvent::starting(true).status,
            "Starting live captions"
        );
        assert_eq!(
            LiveTranscriptEvent::starting(false).status,
            "Live captions disabled"
        );
    }

    #[test]
    fn realtime_tls_provider_is_selected_explicitly() {
        ensure_tls_provider();
        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }

    #[test]
    fn realtime_config_streams_raw_mono_pcm_with_diarization() {
        let config = realtime_config(
            "test-key",
            &["ru-RU".into(), "en-US".into()],
            Some(3),
            Some("de"),
            48_000,
        );
        assert_eq!(config["model"], REALTIME_MODEL);
        assert_eq!(config["audio_format"], "pcm_s16le");
        assert_eq!(config["sample_rate"], 48_000);
        assert_eq!(config["num_channels"], 1);
        assert_eq!(config["enable_speaker_diarization"], true);
        assert_eq!(config["enable_language_identification"], true);
        assert_eq!(config["language_hints"], json!(["ru", "en"]));
        assert_eq!(config["language_hints_strict"], false);
        assert_eq!(config["translation"]["type"], "one_way");
        assert_eq!(config["translation"]["target_language"], "de");
        assert_eq!(
            config["context"]["general"][3]["value"],
            "3 speakers are expected. Keep each distinct voice under a stable speaker label, change labels when the person speaking changes, and do not split one speaker merely because they change language."
        );
        assert!(config["context"]["general"][2]["value"]
            .as_str()
            .unwrap()
            .contains("language actually spoken"));
    }

    #[test]
    fn meeting_context_keeps_unknown_speaker_counts_open_without_losing_diarization_guidance() {
        let context = meeting_context(&["en".into(), "bn".into()], None);

        assert!(context["general"][1]["value"]
            .as_str()
            .unwrap()
            .contains("en, bn"));
        assert!(context["general"][3]["value"]
            .as_str()
            .unwrap()
            .starts_with("One or more speakers may participate"));
    }

    fn original(text: &str, speaker: u64, language: Option<&str>, is_final: bool) -> Token {
        Token {
            text: text.into(),
            speaker: Some(json!(speaker)),
            language: language.map(str::to_string),
            translation_status: Some("original".into()),
            is_final,
            ..Token::default()
        }
    }

    fn translation(text: &str, source_language: Option<&str>, is_final: bool) -> Token {
        Token {
            text: text.into(),
            source_language: source_language.map(str::to_string),
            translation_status: Some("translation".into()),
            is_final,
            ..Token::default()
        }
    }

    fn pcm_bytes(value: i16, samples: usize) -> Vec<u8> {
        (0..samples)
            .flat_map(|_| value.to_le_bytes())
            .collect::<Vec<_>>()
    }

    #[test]
    fn silence_detector_waits_for_a_sustained_pause_and_speech_resets_it() {
        let mut detector = SilenceDetector::new(16_000);
        assert!(!detector.observe(&pcm_bytes(0, 16_000)));
        assert!(!detector.observe(&pcm_bytes(5_000, 1_600)));
        assert!(!detector.observe(&pcm_bytes(0, 16_000)));
        assert!(detector.observe(&pcm_bytes(0, 8_000)));
    }

    #[test]
    fn live_context_restart_uses_quiet_audio_but_has_a_five_second_deadline() {
        assert_eq!(
            context_restart_decision(false, Duration::from_millis(RECONFIGURE_FORCE_MS - 1)),
            (false, false)
        );
        assert_eq!(
            context_restart_decision(true, Duration::from_millis(200)),
            (true, false)
        );
        assert_eq!(
            context_restart_decision(false, Duration::from_millis(RECONFIGURE_FORCE_MS)),
            (true, true)
        );
    }

    #[test]
    fn stream_boundary_keeps_identical_provider_speaker_labels_separate() {
        let tokens = vec![
            original("Before.", 1, Some("en"), true),
            Token {
                translation_status: Some("recall_stream_boundary".into()),
                is_final: true,
                ..Token::default()
            },
            original("After.", 1, Some("en"), true),
        ];

        let turns = build_live_turns(&tokens, &HashSet::new());

        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, "Speaker 1");
        assert_eq!(turns[1].speaker, "Speaker 1");
        assert_eq!(turns[0].segments[0].source_text, "Before.");
        assert_eq!(turns[1].segments[0].source_text, "After.");
    }

    #[test]
    fn restart_marker_describes_the_context_and_stays_between_epochs() {
        let mut accumulator = LiveTranscriptAccumulator {
            final_tokens: vec![original("Before.", 1, Some("en"), true)],
            epoch_total_audio_ms: 2_000,
            ..LiveTranscriptAccumulator::default()
        };
        let options = RealtimeOptions {
            language_hints: vec!["en".into(), "bn".into()],
            expected_speakers: Some(4),
            preferred_language: "en".into(),
            no_translation_languages: Vec::new(),
        };

        accumulator.start_next_epoch(&options);

        assert_eq!(accumulator.markers.len(), 1);
        assert_eq!(accumulator.markers[0].after_sequence, Some(0));
        assert!(accumulator.markers[0].text.contains("4 expected speakers"));
        assert!(accumulator.markers[0].text.contains("en, bn"));
        assert!(is_stream_boundary(accumulator.final_tokens.last().unwrap()));
        assert_eq!(accumulator.audio_offset_ms, 2_000);
    }

    #[test]
    fn live_turns_pair_unequal_source_and_translation_runs() {
        let mut source_without_status = original("Hallo", 1, Some("de-DE"), false);
        source_without_status.translation_status = Some("none".into());
        let tokens = vec![
            source_without_status,
            original(" Welt", 1, Some("de-DE"), false),
            translation("Hello world", Some("de-DE"), false),
        ];
        let turns = build_live_turns(&tokens, &HashSet::new());

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker, "Speaker 1");
        let segment = &turns[0].segments[0];
        assert_eq!(segment.source_text, "Hallo Welt");
        assert_eq!(segment.source_language.as_deref(), Some("de"));
        assert!(!segment.source_final);
        let translation = segment.translation.as_ref().expect("translation paired");
        assert_eq!(translation.text, "Hello world");
        assert_eq!(translation.source_language, "de");
        assert!(!translation.is_final);
    }

    #[test]
    fn one_speaker_code_switches_stay_in_one_turn_with_ordered_segments() {
        let tokens = vec![
            original("Привет", 1, Some("ru"), false),
            translation("Hello", Some("ru"), false),
            original(" hello", 1, Some("en"), false),
            original(" снова", 1, Some("ru-RU"), false),
            translation(" again", Some("ru-RU"), false),
        ];
        let turns = build_live_turns(&tokens, &HashSet::new());

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker, "Speaker 1");
        assert_eq!(turns[0].segments.len(), 3);
        assert_eq!(turns[0].segments[0].source_text, "Привет");
        assert_eq!(turns[0].segments[0].source_language.as_deref(), Some("ru"));
        assert_eq!(turns[0].segments[1].source_text, " hello");
        assert_eq!(turns[0].segments[1].source_language.as_deref(), Some("en"));
        assert_eq!(turns[0].segments[2].source_text, " снова");
        assert_eq!(turns[0].segments[2].source_language.as_deref(), Some("ru"));
        assert_eq!(
            turns[0].segments[0].translation.as_ref().unwrap().text,
            "Hello"
        );
        assert_eq!(
            turns[0].segments[2].translation.as_ref().unwrap().text,
            " again"
        );
    }

    #[test]
    fn speaker_change_starts_a_new_turn() {
        let tokens = vec![
            original("Hallo", 1, Some("de"), true),
            translation("Hello", Some("de"), true),
            original("Bonjour", 2, Some("fr-FR"), false),
            translation("Good morning", Some("fr-FR"), false),
        ];
        let turns = build_live_turns(&tokens, &HashSet::new());

        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker, "Speaker 1");
        assert_eq!(turns[0].segments.len(), 1);
        assert_eq!(turns[1].speaker, "Speaker 2");
        assert_eq!(turns[1].segments.len(), 1);
        assert_eq!(turns[1].segments[0].source_language.as_deref(), Some("fr"));
    }

    #[test]
    fn translations_choose_the_closest_preceding_compatible_segment() {
        let tokens = vec![
            original("First", 1, Some("de"), false),
            original("Second", 2, Some("de"), false),
            translation("Combined", Some("de"), false),
        ];
        let turns = build_live_turns(&tokens, &HashSet::new());

        assert!(turns[0].segments[0].translation.is_none());
        assert_eq!(
            turns[1].segments[0].translation.as_ref().unwrap().text,
            "Combined"
        );
    }

    #[test]
    fn excluded_invalid_and_unmatched_translations_are_omitted() {
        let tokens = vec![
            original("Hallo", 1, Some("de"), false),
            translation("Hello", Some("de"), false),
            original("Bonjour", 2, Some("fr"), false),
            translation("Good morning", None, false),
            translation("Unsafe", Some("xx"), false),
            translation("No source", Some("es"), false),
            original("Hello", 3, Some("en"), false),
            translation("Hallo", Some("en"), false),
        ];
        let turns = build_live_turns(
            &tokens,
            &HashSet::from(["de".to_string(), "en".to_string()]),
        );

        assert_eq!(turns.len(), 3);
        assert!(turns
            .iter()
            .flat_map(|turn| &turn.segments)
            .all(|segment| segment.translation.is_none()));
    }

    #[test]
    fn provisional_turns_and_segments_revise_without_changing_their_stable_identity() {
        let first = build_live_turns(&[original("Hal", 1, Some("de"), false)], &HashSet::new());
        let revised = build_live_turns(
            &[
                original("Hallo", 1, Some("de"), true),
                translation("Hello", Some("de"), false),
            ],
            &HashSet::new(),
        );

        assert_eq!(first[0].id, revised[0].id);
        assert_eq!(first[0].sequence, revised[0].sequence);
        assert_eq!(first[0].segments[0].id, revised[0].segments[0].id);
        assert_eq!(revised[0].segments[0].source_text, "Hallo");
        assert!(revised[0].segments[0].source_final);
        assert!(
            !revised[0].segments[0]
                .translation
                .as_ref()
                .unwrap()
                .is_final
        );
    }

    #[test]
    fn final_tokens_are_appended_once_and_displayed_before_the_current_provisional_tail() {
        let first_final = original("Hello", 1, Some("en"), true);
        let second_final = original(" world", 1, Some("en"), true);
        let mut final_tokens = Vec::new();
        append_final_tokens(&mut final_tokens, vec![first_final.clone()]);
        append_final_tokens(&mut final_tokens, vec![second_final]);
        let provisional = original(" again", 1, Some("en"), false);
        let display = display_tokens(&final_tokens, &[provisional]);

        assert_eq!(final_tokens.len(), 2);
        assert_eq!(
            render_original_tokens(&display),
            "Speaker 1: Hello world again"
        );
        assert_eq!(
            render_original_tokens(&final_tokens),
            "Speaker 1: Hello world"
        );
    }

    #[test]
    fn identical_final_translation_tokens_are_not_deduplicated() {
        let repeated = translation(" yes", Some("de"), true);
        let mut final_tokens = Vec::new();
        append_final_tokens(&mut final_tokens, vec![repeated.clone()]);
        append_final_tokens(&mut final_tokens, vec![repeated]);

        assert_eq!(final_tokens.len(), 2);
        assert_eq!(collect_translation_runs(&final_tokens)[0].text, " yes yes");
    }

    #[test]
    fn translated_and_untranslated_segments_keep_source_metadata_and_identity() {
        let tokens = vec![
            original("Guten Tag", 1, Some("de"), true),
            translation("Good day", Some("de"), true),
            original("Hello", 2, Some("en"), false),
            translation("Hallo", Some("en"), false),
            original("Bonjour", 3, Some("fr"), false),
            translation("Good morning", Some("fr"), false),
        ];
        let turns = build_live_turns(&tokens, &HashSet::from(["en".to_string()]));

        assert_eq!(
            turns.iter().map(|turn| turn.sequence).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(turns[0].id, "live-turn-0");
        assert_eq!(turns[1].id, "live-turn-1");
        assert_eq!(turns[0].segments[0].source_language.as_deref(), Some("de"));
        assert_eq!(
            turns[0].segments[0]
                .translation
                .as_ref()
                .expect("German translation should be paired")
                .source_language,
            "de"
        );
        assert_eq!(turns[1].segments[0].source_language.as_deref(), Some("en"));
        assert!(turns[1].segments[0].translation.is_none());
        assert_eq!(turns[2].segments[0].source_language.as_deref(), Some("fr"));
        assert_eq!(
            turns[2].segments[0]
                .translation
                .as_ref()
                .expect("French translation should be paired")
                .source_language,
            "fr"
        );
        assert!(turns[0].segments[0].source_final);
        assert!(!turns[2].segments[0].source_final);
    }

    #[test]
    fn long_unequal_runs_remain_one_correctly_paired_segment() {
        let mut tokens = (0..24)
            .map(|index| original(&format!(" {index}"), 1, Some("de"), false))
            .collect::<Vec<_>>();
        tokens.push(translation("The complete translation", Some("de"), false));
        let turns = build_live_turns(&tokens, &HashSet::new());

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].sequence, 0);
        assert_eq!(
            turns[0].segments[0].translation.as_ref().unwrap().text,
            "The complete translation"
        );
    }

    #[test]
    fn final_and_provisional_tokens_in_one_provider_chunk_stay_one_fluid_pair() {
        let tokens = vec![
            original("Guten", 1, Some("de"), true),
            original(" Mor", 1, Some("de"), false),
            translation("Good", Some("de"), true),
            translation(" mor", Some("de"), false),
        ];
        let turns = build_live_turns(&tokens, &HashSet::new());

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].segments[0].source_text, "Guten Mor");
        assert!(!turns[0].segments[0].source_final);
        let translation = turns[0].segments[0].translation.as_ref().unwrap();
        assert_eq!(translation.text, "Good mor");
        assert!(!translation.is_final);
    }

    #[test]
    fn preferred_language_is_normalized_against_the_supported_catalogue() {
        assert_eq!(normalize_translation_language("DE-de"), Some("de".into()));
        assert_eq!(normalize_translation_language("xx"), None);
    }

    #[test]
    fn unsupported_live_translation_target_keeps_original_stt_enabled() {
        let (target, warning) = live_translation_policy("xx");
        assert!(target.is_none());
        assert!(warning
            .unwrap()
            .contains("Original live captions will continue"));
        let config = realtime_config("test-key", &[], None, target.as_deref(), 48_000);
        assert!(config.get("translation").is_none());
        assert_eq!(config["enable_language_identification"], true);
    }
}

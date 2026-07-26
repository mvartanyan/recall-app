use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    time::SystemTime,
};

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::{password_hash::SaltString, Argon2};
use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use rand::RngCore;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;
use zeroize::Zeroize;

use crate::{
    jamie_import::{
        is_generic_speaker_label, validate_import_draft, JamieArchive, JamieImportDraft,
        JamieKnownPerson, JAMIE_IMPORTER_VERSION,
    },
    recap::RecapPayload,
};

const SUGGESTION_REFERENCE_COMPATIBILITY_THRESHOLD: f32 = 0.94;

#[derive(Clone)]
pub struct Crypto {
    key: Option<aes_gcm::Key<Aes256Gcm>>,
    salt: Option<String>,
}

impl Crypto {
    pub fn new(password: Option<&str>, salt: Option<String>) -> Self {
        if let Some(pw) = password {
            let salt = salt.unwrap_or_else(|| SaltString::generate(&mut OsRng).to_string());
            let salt_obj =
                SaltString::from_b64(&salt).unwrap_or_else(|_| SaltString::generate(&mut OsRng));
            let argon2 = Argon2::default();
            let mut key_bytes = [0u8; 32];
            let salt_bytes: &[u8] = salt_obj.as_salt().as_str().as_bytes();
            let _ = argon2
                .hash_password_into(pw.as_bytes(), salt_bytes, &mut key_bytes)
                .map_err(|_| "kdf failed");
            let key = aes_gcm::Key::<Aes256Gcm>::from_slice(&key_bytes).to_owned();
            key_bytes.zeroize();
            Crypto {
                key: Some(key),
                salt: Some(salt),
            }
        } else {
            Crypto { key: None, salt }
        }
    }

    pub fn encrypt(&self, data: &[u8]) -> (String, String) {
        if let Some(key) = &self.key {
            let cipher = Aes256Gcm::new(key);
            let mut nonce_bytes = [0u8; 12];
            OsRng.fill_bytes(&mut nonce_bytes);
            let nonce = Nonce::from_slice(&nonce_bytes);
            let ct = cipher
                .encrypt(nonce, data)
                .expect("encryption failure should not happen");
            let nonce_b64 = general_purpose::STANDARD.encode(nonce_bytes);
            let ct_b64 = general_purpose::STANDARD.encode(ct);
            (nonce_b64, ct_b64)
        } else {
            (String::new(), general_purpose::STANDARD.encode(data))
        }
    }

    pub fn decrypt(&self, nonce_b64: &str, ct_b64: &str) -> Result<Vec<u8>, String> {
        let data = general_purpose::STANDARD
            .decode(ct_b64)
            .map_err(|e| format!("b64 decode error: {e}"))?;
        if let Some(key) = &self.key {
            let nonce_bytes = general_purpose::STANDARD
                .decode(nonce_b64)
                .map_err(|e| format!("b64 decode nonce error: {e}"))?;
            let nonce = Nonce::from_slice(&nonce_bytes);
            let cipher = Aes256Gcm::new(key);
            cipher
                .decrypt(nonce, data.as_ref())
                .map_err(|e| format!("decrypt error: {e}"))
        } else {
            Ok(data)
        }
    }

    pub fn salt(&self) -> Option<String> {
        self.salt.clone()
    }
}

pub struct Db {
    conn: std::sync::Mutex<Connection>,
    crypto: Crypto,
    path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub title: String,
    pub duration_ms: i64,
    pub transcript: String,
    pub processing_status: Option<String>,
    pub processing_error: Option<String>,
    pub processing_run_id: Option<String>,
    pub recoverable_audio: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub title: String,
    pub duration_ms: i64,
    pub processing_status: Option<String>,
    pub processing_error: Option<String>,
    pub processing_run_id: Option<String>,
    pub recoverable_audio: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessingJob {
    pub session_id: String,
    pub run_id: String,
    pub audio_path: String,
    pub status: String,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportedSessionArtifact {
    pub session_id: String,
    pub source_provider: String,
    pub source_meeting_sha256: String,
    pub imported_at: DateTime<Utc>,
    pub executive_summary: String,
    pub full_summary: String,
    pub tasks: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JamieImportResult {
    pub import_id: Option<String>,
    pub backup_path: Option<String>,
    pub imported_meetings: usize,
    pub already_imported_meetings: usize,
    pub imported_interventions: usize,
    pub created_people: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct JamieRollbackResult {
    pub import_id: String,
    pub backup_path: String,
    pub removed_meetings: usize,
    pub removed_people: usize,
    pub preserved_people: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportBatchSummary {
    pub id: String,
    pub source_provider: String,
    pub source_file_sha256: String,
    pub imported_at: DateTime<Utc>,
    pub status: String,
    pub meeting_count: usize,
    pub rolled_back_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentRecord {
    pub id: String,
    pub session_id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker_id: Option<String>,
    pub speaker_label: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoredEmbedding {
    pub id: String,
    pub speaker_id: String,
    pub speaker_label: Option<String>,
    pub vector: Vec<f32>,
    pub source_session_id: String,
    pub created_at: DateTime<Utc>,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeakerSample {
    pub id: String,
    pub speaker_id: String,
    pub sample_b64: String,
    pub sample_rate: u32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Speaker {
    pub id: String,
    pub label: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SpeakerStats {
    pub id: String,
    pub label: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_seen_at: Option<DateTime<Utc>>,
    pub sample_count: usize,
    pub embedding_count: usize,
    pub conversation_count: usize,
    pub likely_match: Option<VoiceMatchSuggestion>,
    pub duplicate_name_conflict: bool,
    pub duplicate_name_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct IdentityProfileRow {
    pub id: String,
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub last_seen_at: Option<DateTime<Utc>>,
    pub sample_count: usize,
    pub active_voiceprint_count: usize,
    pub inactive_voiceprint_count: usize,
    pub conversation_count: usize,
    pub intervention_count: usize,
    pub provisional: bool,
    pub imported: bool,
    pub duplicate_name_conflict: bool,
    pub duplicate_name_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct IdentityProfilePage {
    pub items: Vec<IdentityProfileRow>,
    pub total: usize,
    pub page: usize,
    pub page_size: usize,
    pub page_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct UnassignedIdentityKey {
    pub session_id: String,
    pub speaker_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct UnassignedIdentityRow {
    pub key: UnassignedIdentityKey,
    pub display_label: String,
    pub session_title: String,
    pub session_created_at: DateTime<Utc>,
    pub intervention_count: usize,
    pub first_start_ms: i64,
    pub last_end_ms: i64,
    pub generic: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct UnassignedIdentityPage {
    pub items: Vec<UnassignedIdentityRow>,
    pub total: usize,
    pub page: usize,
    pub page_size: usize,
    pub page_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentityConsolidationRequest {
    pub profile_ids: Vec<String>,
    pub unassigned_groups: Vec<UnassignedIdentityKey>,
    pub target_speaker_id: Option<String>,
    pub final_label: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct IdentityConsolidationPreview {
    pub target_speaker_id: Option<String>,
    pub target_label: String,
    pub source_profiles: Vec<IdentityProfileRow>,
    pub unassigned_groups: Vec<UnassignedIdentityRow>,
    pub affected_session_ids: Vec<String>,
    pub affected_conversation_count: usize,
    pub affected_intervention_count: usize,
    pub stale_recap_count: usize,
    pub active_voiceprint_count: usize,
    pub inactive_voiceprint_count: usize,
    pub samples_to_delete: usize,
    pub imported_source_profile_count: usize,
    pub creates_new_person: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct IdentityConsolidationResult {
    pub target_speaker_id: String,
    pub target_label: String,
    pub merged_profile_count: usize,
    pub assigned_group_count: usize,
    pub affected_conversation_count: usize,
    pub affected_intervention_count: usize,
    pub activated_voiceprints: usize,
    pub quarantined_voiceprints: usize,
    pub deleted_samples: usize,
    pub backup_path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceMatchSuggestion {
    pub decision_id: String,
    pub speaker_id: String,
    pub label: String,
    pub score: f32,
    pub runner_up_label: Option<String>,
    pub runner_up_score: Option<f32>,
    pub support_count: usize,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceMatchDecision {
    pub id: String,
    pub session_id: String,
    pub provider_speakers: Vec<String>,
    pub resulting_speaker_id: Option<String>,
    pub best_speaker_id: Option<String>,
    pub best_speaker_label: Option<String>,
    pub runner_up_speaker_id: Option<String>,
    pub runner_up_speaker_label: Option<String>,
    pub best_score: Option<f32>,
    pub runner_up_score: Option<f32>,
    pub support_count: usize,
    pub selected_duration_ms: u64,
    pub selected_window_count: usize,
    pub consistency_score: Option<f32>,
    pub model_version: String,
    pub decision: String,
    pub reason: String,
    pub created_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub resolution: Option<String>,
}

pub struct VoiceMatchDecisionSave<'a> {
    pub session_id: &'a str,
    pub provider_speakers: &'a [String],
    pub resulting_speaker_id: Option<&'a str>,
    pub best_speaker_id: Option<&'a str>,
    pub runner_up_speaker_id: Option<&'a str>,
    pub best_score: Option<f32>,
    pub runner_up_score: Option<f32>,
    pub support_count: usize,
    pub selected_duration_ms: u64,
    pub selected_window_count: usize,
    pub consistency_score: Option<f32>,
    pub model_version: &'a str,
    pub decision: &'a str,
    pub reason: &'a str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RenameSpeakerResult {
    pub status: String,
    pub conflicting_speaker_id: Option<String>,
    pub conflicting_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SuggestionAcceptance {
    pub target_speaker_id: String,
    pub target_label: String,
    pub activated_voiceprints: usize,
    pub quarantined_voiceprints: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SpeakerMergeResult {
    pub target_speaker_id: String,
    pub target_label: String,
    pub activated_voiceprints: usize,
    pub quarantined_voiceprints: usize,
    pub replaced_target_voiceprints: bool,
}

#[derive(Debug, Clone)]
pub struct AgendaRecord {
    pub source_kind: String,
    pub filename: String,
    pub mime_type: String,
    pub content: Vec<u8>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgendaMetadata {
    pub source_kind: String,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: usize,
    pub updated_at: DateTime<Utc>,
    pub text_content: Option<String>,
}

impl AgendaRecord {
    pub fn metadata(&self) -> AgendaMetadata {
        AgendaMetadata {
            source_kind: self.source_kind.clone(),
            filename: self.filename.clone(),
            mime_type: self.mime_type.clone(),
            size_bytes: self.content.len(),
            updated_at: self.updated_at,
            text_content: (self.source_kind == "text")
                .then(|| String::from_utf8_lossy(&self.content).to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RecapRecord {
    pub session_id: String,
    pub generated_at: DateTime<Utc>,
    pub model: String,
    pub prompt_version: String,
    pub schema_version: String,
    pub source_fingerprint: String,
    pub payload: RecapPayload,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

pub struct RecapSave<'a> {
    pub session_id: &'a str,
    pub title: &'a str,
    pub model: &'a str,
    pub prompt_version: &'a str,
    pub schema_version: &'a str,
    pub source_fingerprint: &'a str,
    pub payload: &'a RecapPayload,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

fn is_provisional_label(label: &str) -> bool {
    let trimmed = label.trim();
    let Some(suffix) = trimmed
        .get(..5)
        .filter(|prefix| prefix.eq_ignore_ascii_case("VOICE"))
    else {
        return false;
    };
    let number = &trimmed[suffix.len()..];
    !number.is_empty() && number.chars().all(|character| character.is_ascii_digit())
}

pub(crate) fn normalized_person_name(label: &str) -> String {
    label
        .nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn display_person_name(label: &str) -> String {
    label
        .nfc()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn natural_label_cmp(left: &str, right: &str) -> Ordering {
    let left = left.to_lowercase();
    let right = right.to_lowercase();
    let mut left_chars = left.chars().peekable();
    let mut right_chars = right.chars().peekable();
    loop {
        match (left_chars.peek().copied(), right_chars.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(left_char), Some(right_char))
                if left_char.is_ascii_digit() && right_char.is_ascii_digit() =>
            {
                let mut left_number = String::new();
                let mut right_number = String::new();
                while left_chars
                    .peek()
                    .is_some_and(|value| value.is_ascii_digit())
                {
                    left_number.push(left_chars.next().unwrap_or_default());
                }
                while right_chars
                    .peek()
                    .is_some_and(|value| value.is_ascii_digit())
                {
                    right_number.push(right_chars.next().unwrap_or_default());
                }
                let number_order = left_number
                    .parse::<u128>()
                    .unwrap_or_default()
                    .cmp(&right_number.parse::<u128>().unwrap_or_default());
                if number_order != Ordering::Equal {
                    return number_order;
                }
                let width_order = left_number.len().cmp(&right_number.len());
                if width_order != Ordering::Equal {
                    return width_order;
                }
            }
            (Some(left_char), Some(right_char)) => {
                left_chars.next();
                right_chars.next();
                let order = left_char.cmp(&right_char);
                if order != Ordering::Equal {
                    return order;
                }
            }
        }
    }
}

fn bounded_page(page: usize, page_size: usize, total: usize) -> (usize, usize, usize) {
    let page_size = page_size.clamp(1, 100);
    let page_count = total.div_ceil(page_size).max(1);
    let page = page.clamp(1, page_count);
    (page, page_size, page_count)
}

impl Db {
    pub fn open(path: impl AsRef<Path>, crypto: Crypto) -> Result<Self, String> {
        let path = path.as_ref();
        Self::backup_before_migration(path)?;
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        if path != Path::new(":memory:") {
            Self::restrict_file_permissions(path)?;
            Self::restrict_existing_backup_permissions(path)?;
        }
        let db = Db {
            conn: std::sync::Mutex::new(conn),
            crypto,
            path: (path != Path::new(":memory:")).then(|| path.to_path_buf()),
        };
        db.init_schema()?;
        db.mark_interrupted_processing_jobs()?;
        db.persist_salt_if_missing()?;
        Ok(db)
    }

    fn backup_before_migration(path: &Path) -> Result<(), String> {
        if path == Path::new(":memory:") || !path.is_file() {
            return Ok(());
        }
        let conn = Connection::open(path).map_err(|error| error.to_string())?;
        let recap_migration_needed = Self::table_exists(&conn, "sessions")?
            && (!Self::table_exists(&conn, "session_agendas")?
                || !Self::table_exists(&conn, "session_recaps")?);
        if recap_migration_needed {
            let backup = Self::recap_migration_backup_path(path);
            if !backup.exists() {
                std::fs::copy(path, &backup).map_err(|error| {
                    format!(
                        "Could not back up the existing Recall database to {} before adding recaps: {error}",
                        backup.display()
                    )
                })?;
                Self::restrict_file_permissions(&backup)?;
                eprintln!(
                    "[database] backed up the pre-recap database to {}",
                    backup.display()
                );
            }
        }
        let processing_migration_needed = Self::table_exists(&conn, "sessions")?
            && !Self::table_exists(&conn, "processing_jobs")?;
        if processing_migration_needed {
            let backup = Self::processing_migration_backup_path(path);
            if !backup.exists() {
                std::fs::copy(path, &backup).map_err(|error| {
                    format!(
                        "Could not back up the existing Recall database to {} before adding recoverable processing jobs: {error}",
                        backup.display()
                    )
                })?;
                Self::restrict_file_permissions(&backup)?;
                eprintln!(
                    "[database] backed up the pre-processing-job database to {}",
                    backup.display()
                );
            }
        }
        let voice_match_table_exists = Self::table_exists(&conn, "voice_match_decisions")?;
        let voice_match_migration_needed = Self::table_exists(&conn, "sessions")?
            && (!voice_match_table_exists
                || !Self::column_exists(&conn, "voice_match_decisions", "best_speaker_id")?);
        if voice_match_migration_needed {
            let backup = Self::voice_match_migration_backup_path(path);
            if !backup.exists() {
                std::fs::copy(path, &backup).map_err(|error| {
                    format!(
                        "Could not back up the existing Recall database to {} before adding voice-match diagnostics: {error}",
                        backup.display()
                    )
                })?;
                Self::restrict_file_permissions(&backup)?;
                eprintln!(
                    "[database] backed up the pre-voice-match database to {}",
                    backup.display()
                );
            }
        }
        let import_migration_needed =
            Self::table_exists(&conn, "sessions")? && !Self::table_exists(&conn, "import_batches")?;
        if import_migration_needed {
            let backup = Self::import_migration_backup_path(path);
            if !backup.exists() {
                std::fs::copy(path, &backup).map_err(|error| {
                    format!(
                        "Could not back up the existing Recall database to {} before adding archive imports: {error}",
                        backup.display()
                    )
                })?;
                Self::restrict_file_permissions(&backup)?;
                eprintln!(
                    "[database] backed up the pre-import database to {}",
                    backup.display()
                );
            }
        }
        let checks = [
            ("sessions", "title"),
            ("sessions", "duration_ms"),
            ("segments", "speaker_id"),
            ("embeddings", "model_version"),
            ("embeddings", "is_reference"),
        ];
        let mut missing = Vec::new();
        for (table, column) in checks {
            if Self::table_exists(&conn, table)? && !Self::column_exists(&conn, table, column)? {
                missing.push((table, column));
            }
        }
        drop(conn);
        if missing.is_empty() {
            return Ok(());
        }
        let reference_only = missing.as_slice() == [("embeddings", "is_reference")];
        let backup = if reference_only {
            Self::reference_migration_backup_path(path)
        } else {
            Self::migration_backup_path(path)
        };
        if backup.exists() {
            return Ok(());
        }
        std::fs::copy(path, &backup).map_err(|error| {
            format!(
                "Could not back up the existing Recall database to {} before migration: {error}",
                backup.display()
            )
        })?;
        Self::restrict_file_permissions(&backup)?;
        eprintln!(
            "[database] backed up the pre-migration database to {}",
            backup.display()
        );
        Ok(())
    }

    #[cfg(unix)]
    fn restrict_file_permissions(path: &Path) -> Result<(), String> {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|error| {
            format!(
                "Could not restrict database permissions for {}: {error}",
                path.display()
            )
        })
    }

    #[cfg(not(unix))]
    fn restrict_file_permissions(_path: &Path) -> Result<(), String> {
        Ok(())
    }

    fn restrict_existing_backup_permissions(path: &Path) -> Result<(), String> {
        if path.file_name().and_then(|value| value.to_str()) != Some("recall.db") {
            return Ok(());
        }
        let Some(directory) = path.parent() else {
            return Ok(());
        };
        for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
            let candidate = entry.map_err(|error| error.to_string())?.path();
            let Some(filename) = candidate.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if filename.starts_with("recall.pre-") && filename.ends_with(".db") {
                Self::restrict_file_permissions(&candidate)?;
            }
        }
        Ok(())
    }

    fn migration_backup_path(path: &Path) -> PathBuf {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("recall");
        path.with_file_name(format!("{stem}.pre-standalone-v1.db"))
    }

    fn reference_migration_backup_path(path: &Path) -> PathBuf {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("recall");
        path.with_file_name(format!("{stem}.pre-voice-reference-v1.db"))
    }

    fn recap_migration_backup_path(path: &Path) -> PathBuf {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("recall");
        path.with_file_name(format!("{stem}.pre-recap-v1.db"))
    }

    fn processing_migration_backup_path(path: &Path) -> PathBuf {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("recall");
        path.with_file_name(format!("{stem}.pre-processing-v1.db"))
    }

    fn voice_match_migration_backup_path(path: &Path) -> PathBuf {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("recall");
        path.with_file_name(format!("{stem}.pre-voice-match-v1.db"))
    }

    fn import_migration_backup_path(path: &Path) -> PathBuf {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("recall");
        path.with_file_name(format!("{stem}.pre-import-v1.db"))
    }

    fn table_exists(conn: &Connection, table: &str) -> Result<bool, String> {
        conn.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1
             )",
            params![table],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())
    }

    fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool, String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(|error| error.to_string())?;
        let names = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|error| error.to_string())?;
        for name in names {
            if name.map_err(|error| error.to_string())? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn load_existing_salt(path: impl AsRef<Path>) -> Result<Option<String>, String> {
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        let mut stmt = match conn.prepare("SELECT value FROM meta WHERE key='salt'") {
            Ok(stmt) => stmt,
            Err(e) => {
                if e.to_string().contains("no such table") {
                    return Ok(None);
                }
                return Err(e.to_string());
            }
        };
        let salt_opt: Option<String> = stmt
            .query_row([], |row| row.get(0))
            .optional()
            .map_err(|e| e.to_string())?;
        Ok(salt_opt)
    }

    fn init_schema(&self) -> Result<(), String> {
        let conn_guard = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let embeddings_existed = Self::table_exists(&conn_guard, "embeddings")?;
        let embeddings_had_reference =
            embeddings_existed && Self::column_exists(&conn_guard, "embeddings", "is_reference")?;
        conn_guard
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                 CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT,
                    duration_ms INTEGER DEFAULT 0,
                    transcript_nonce TEXT,
                    transcript_ct TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS speakers (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    created_at TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    speaker_id TEXT,
                    vector_nonce TEXT,
                    vector_ct TEXT NOT NULL,
                    source_session_id TEXT,
                    created_at TEXT NOT NULL,
                    model_version TEXT,
                    is_reference INTEGER NOT NULL DEFAULT 1
                 );
                 CREATE TABLE IF NOT EXISTS speaker_samples (
                    id TEXT PRIMARY KEY,
                    speaker_id TEXT NOT NULL,
                    sample_b64 TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    speaker_label TEXT,
                    speaker_id TEXT,
                    text_nonce TEXT,
                    text_ct TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS session_agendas (
                    session_id TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    content_nonce TEXT,
                    content_ct TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS session_recaps (
                    session_id TEXT PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    payload_nonce TEXT,
                    payload_ct TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0
                 );
                 CREATE TABLE IF NOT EXISTS processing_jobs (
                    session_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS voice_match_decisions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    provider_speakers_json TEXT NOT NULL,
                    resulting_speaker_id TEXT,
                    best_speaker_id TEXT,
                    runner_up_speaker_id TEXT,
                    best_score REAL,
                    runner_up_score REAL,
                    support_count INTEGER NOT NULL DEFAULT 0,
                    selected_duration_ms INTEGER NOT NULL DEFAULT 0,
                    selected_window_count INTEGER NOT NULL DEFAULT 0,
                    consistency_score REAL,
                    model_version TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution TEXT
                 );
                 CREATE TABLE IF NOT EXISTS import_batches (
                    id TEXT PRIMARY KEY,
                    source_provider TEXT NOT NULL,
                    source_file_sha256 TEXT NOT NULL,
                    source_exported_at TEXT,
                    importer_version TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    meeting_count INTEGER NOT NULL DEFAULT 0,
                    manifest_nonce TEXT,
                    manifest_ct TEXT NOT NULL,
                    rolled_back_at TEXT
                 );
                 CREATE TABLE IF NOT EXISTS imported_sessions (
                    source_provider TEXT NOT NULL,
                    source_meeting_sha256 TEXT NOT NULL,
                    import_id TEXT NOT NULL,
                    session_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY(source_provider, source_meeting_sha256)
                 );
                 CREATE TABLE IF NOT EXISTS session_import_artifacts (
                    session_id TEXT PRIMARY KEY,
                    source_provider TEXT NOT NULL,
                    source_meeting_sha256 TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    executive_summary_nonce TEXT,
                    executive_summary_ct TEXT NOT NULL,
                    full_summary_nonce TEXT,
                    full_summary_ct TEXT NOT NULL,
                    tasks_nonce TEXT,
                    tasks_ct TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS import_created_speakers (
                    import_id TEXT NOT NULL,
                    speaker_id TEXT NOT NULL,
                    PRIMARY KEY(import_id, speaker_id)
                 );",
            )
            .map_err(|e| e.to_string())?;
        let legacy_suggestion_column =
            Self::column_exists(&conn_guard, "voice_match_decisions", "suggested_speaker_id")?;
        Self::add_column_if_missing(
            &conn_guard,
            "voice_match_decisions",
            "best_speaker_id",
            "TEXT",
        )?;
        if legacy_suggestion_column {
            conn_guard
                .execute(
                    "UPDATE voice_match_decisions
                        SET best_speaker_id=COALESCE(best_speaker_id, suggested_speaker_id)",
                    [],
                )
                .map_err(|error| error.to_string())?;
        }
        conn_guard
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS voice_match_decisions_session_idx
                    ON voice_match_decisions(session_id, created_at);
                 CREATE INDEX IF NOT EXISTS voice_match_decisions_result_idx
                    ON voice_match_decisions(resulting_speaker_id, decision, resolved_at);
                 CREATE INDEX IF NOT EXISTS imported_sessions_import_idx
                    ON imported_sessions(import_id);
                 CREATE INDEX IF NOT EXISTS import_batches_source_idx
                    ON import_batches(source_provider, source_file_sha256, imported_at);",
            )
            .map_err(|e| e.to_string())?;

        Self::add_column_if_missing(&conn_guard, "segments", "speaker_id", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "segments", "speaker_label", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "sessions", "title", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "sessions", "duration_ms", "INTEGER DEFAULT 0")?;
        Self::add_column_if_missing(&conn_guard, "embeddings", "model_version", "TEXT")?;
        Self::add_column_if_missing(
            &conn_guard,
            "import_batches",
            "meeting_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        if embeddings_existed && !embeddings_had_reference {
            Self::add_column_if_missing(
                &conn_guard,
                "embeddings",
                "is_reference",
                "INTEGER NOT NULL DEFAULT 0",
            )?;
        }
        let reference_migration_complete: bool = conn_guard
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM meta WHERE key='voice_reference_migration_v1')",
                [],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        if !reference_migration_complete {
            // Earlier builds did not distinguish enrollment voiceprints from
            // automatically accumulated observations. Preserve the oldest
            // vector for each profile/model as its conservative reference and
            // quarantine the rest from matching instead of deleting anything.
            // The marker makes this restart-safe if the app stops after ALTER
            // TABLE but before reference initialization completes.
            conn_guard
                .execute_batch(
                    "BEGIN IMMEDIATE;
                     UPDATE embeddings SET is_reference = 0;
                     UPDATE embeddings AS candidate
                        SET is_reference = 1
                      WHERE NOT EXISTS (
                            SELECT 1
                              FROM embeddings AS earlier
                             WHERE earlier.speaker_id = candidate.speaker_id
                               AND COALESCE(earlier.model_version, '') = COALESCE(candidate.model_version, '')
                               AND (
                                    earlier.created_at < candidate.created_at
                                    OR (earlier.created_at = candidate.created_at AND earlier.id < candidate.id)
                               )
                      );
                     INSERT OR REPLACE INTO meta(key, value)
                     VALUES('voice_reference_migration_v1', 'complete');
                     COMMIT;",
                )
                .map_err(|error| error.to_string())?;
        }
        conn_guard
            .execute_batch(
                "CREATE INDEX IF NOT EXISTS sessions_created_at_idx
                    ON sessions(created_at DESC);
                 CREATE INDEX IF NOT EXISTS segments_speaker_session_idx
                    ON segments(speaker_id, session_id);
                 CREATE INDEX IF NOT EXISTS segments_session_start_idx
                    ON segments(session_id, start_ms, id);
                 CREATE INDEX IF NOT EXISTS embeddings_speaker_model_reference_idx
                    ON embeddings(speaker_id, model_version, is_reference);
                 CREATE INDEX IF NOT EXISTS speaker_samples_speaker_idx
                    ON speaker_samples(speaker_id);",
            )
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    fn clear_processing_artifacts_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        session_id: &str,
    ) -> Result<usize, String> {
        let affected_speakers = {
            let mut stmt = tx
                .prepare(
                    "SELECT DISTINCT s.id, s.label
                       FROM speakers s
                      WHERE s.id IN (
                            SELECT speaker_id FROM segments
                             WHERE session_id=?1 AND speaker_id IS NOT NULL
                            UNION
                            SELECT speaker_id FROM embeddings
                             WHERE source_session_id=?1 AND speaker_id IS NOT NULL
                            UNION
                            SELECT resulting_speaker_id FROM voice_match_decisions
                             WHERE session_id=?1 AND resulting_speaker_id IS NOT NULL
                      )",
                )
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![session_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })
                .map_err(|error| error.to_string())?;
            let mut speakers = Vec::new();
            for row in rows {
                speakers.push(row.map_err(|error| error.to_string())?);
            }
            speakers
        };

        tx.execute(
            "DELETE FROM segments WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM voice_match_decisions WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;

        let mut removed_voices = 0;
        for (speaker_id, label) in affected_speakers {
            let provisional = label
                .as_deref()
                .map(|value| value.trim().is_empty() || is_provisional_label(value))
                .unwrap_or(true);
            if !provisional {
                continue;
            }
            let still_referenced: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM segments WHERE speaker_id=?1)",
                    params![speaker_id],
                    |row| row.get(0),
                )
                .map_err(|error| error.to_string())?;
            if still_referenced {
                continue;
            }
            tx.execute(
                "DELETE FROM embeddings WHERE speaker_id=?1",
                params![speaker_id],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "DELETE FROM speaker_samples WHERE speaker_id=?1",
                params![speaker_id],
            )
            .map_err(|error| error.to_string())?;
            removed_voices += tx
                .execute("DELETE FROM speakers WHERE id=?1", params![speaker_id])
                .map_err(|error| error.to_string())?;
        }
        Ok(removed_voices)
    }

    fn mark_interrupted_processing_jobs(&self) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let interrupted = {
            let mut stmt = tx
                .prepare(
                    "SELECT session_id FROM processing_jobs
                      WHERE status IN ('queued', 'processing')",
                )
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            let mut session_ids = Vec::new();
            for row in rows {
                session_ids.push(row.map_err(|error| error.to_string())?);
            }
            session_ids
        };
        if interrupted.is_empty() {
            return Ok(());
        }
        let now: DateTime<Utc> = SystemTime::now().into();
        for session_id in &interrupted {
            Self::clear_processing_artifacts_in_transaction(&tx, session_id)?;
            tx.execute(
                "UPDATE processing_jobs
                    SET status='failed',
                        error='Final transcription was interrupted when Recall closed. The recording is still available and can be retried.',
                        updated_at=?1
                  WHERE session_id=?2",
                params![now.to_rfc3339(), session_id],
            )
            .map_err(|error| error.to_string())?;
        }
        tx.commit().map_err(|error| error.to_string())?;
        eprintln!(
            "[database] marked {} interrupted transcription job(s) as recoverable failures",
            interrupted.len()
        );
        Ok(())
    }

    fn add_column_if_missing(
        conn: &Connection,
        table: &str,
        column: &str,
        decl: &str,
    ) -> Result<(), String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(|e| e.to_string())?;
        let mut existing_cols: HashSet<String> = HashSet::new();
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| e.to_string())?;
        for col in rows {
            existing_cols.insert(col.map_err(|e| e.to_string())?);
        }
        if !existing_cols.contains(column) {
            conn.execute(
                &format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"),
                [],
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn persist_salt_if_missing(&self) -> Result<(), String> {
        if let Some(salt) = self.crypto.salt() {
            let existing = self.load_salt()?;
            if existing.is_none() {
                self.save_salt(&salt)?;
            }
        }
        Ok(())
    }

    fn save_salt(&self, salt: &str) -> Result<(), String> {
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('salt', ?1)",
                params![salt],
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load_salt(&self) -> Result<Option<String>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare("SELECT value FROM meta WHERE key='salt'")
            .map_err(|e| e.to_string())?;
        let salt_opt: Option<String> = stmt
            .query_row([], |row| row.get(0))
            .optional()
            .map_err(|e| e.to_string())?;
        Ok(salt_opt)
    }

    #[allow(dead_code)]
    pub fn insert_session(
        &self,
        title: &str,
        transcript: &str,
        duration_ms: i64,
    ) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let (nonce, ct) = self.crypto.encrypt(transcript.as_bytes());
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO sessions(id, created_at, title, duration_ms, transcript_nonce, transcript_ct) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, now.to_rfc3339(), title, duration_ms, nonce, ct],
            )
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn create_processing_session(
        &self,
        session_id: &str,
        run_id: &str,
        title: &str,
        transcript: &str,
        duration_ms: i64,
        audio_path: &str,
    ) -> Result<(), String> {
        let now: DateTime<Utc> = SystemTime::now().into();
        let (nonce, ct) = self.crypto.encrypt(transcript.as_bytes());
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        tx.execute(
            "INSERT INTO sessions(id, created_at, title, duration_ms, transcript_nonce, transcript_ct)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id,
                now.to_rfc3339(),
                title,
                duration_ms,
                nonce,
                ct
            ],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "INSERT INTO processing_jobs(session_id, run_id, audio_path, status, error, created_at, updated_at)
             VALUES(?1, ?2, ?3, 'processing', NULL, ?4, ?4)",
            params![session_id, run_id, audio_path, now.to_rfc3339()],
        )
        .map_err(|error| error.to_string())?;
        tx.commit().map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn attach_cleanup_recording(
        &self,
        session_id: &str,
        run_id: &str,
        audio_path: &str,
        error: &str,
    ) -> Result<(), String> {
        let now: DateTime<Utc> = SystemTime::now().into();
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO processing_jobs(session_id, run_id, audio_path, status, error, created_at, updated_at)
                 VALUES(?1, ?2, ?3, 'cleanup_failed', ?4, ?5, ?5)",
                params![session_id, run_id, audio_path, error, now.to_rfc3339()],
            )
            .map_err(|failure| failure.to_string())?;
        Ok(())
    }

    pub fn processing_job(&self, session_id: &str) -> Result<Option<ProcessingJob>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let row = conn
            .query_row(
                "SELECT session_id, run_id, audio_path, status, error, created_at, updated_at
                   FROM processing_jobs WHERE session_id=?1",
                params![session_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some((session_id, run_id, audio_path, status, error, created_at, updated_at)) = row
        else {
            return Ok(None);
        };
        Ok(Some(ProcessingJob {
            session_id,
            run_id,
            audio_path,
            status,
            error,
            created_at: DateTime::parse_from_rfc3339(&created_at)
                .map_err(|error| error.to_string())?
                .with_timezone(&Utc),
            updated_at: DateTime::parse_from_rfc3339(&updated_at)
                .map_err(|error| error.to_string())?
                .with_timezone(&Utc),
        }))
    }

    pub fn restart_processing_session(&self, session_id: &str, run_id: &str) -> Result<(), String> {
        let now: DateTime<Utc> = SystemTime::now().into();
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        Self::clear_processing_artifacts_in_transaction(&tx, session_id)?;
        let changed = tx
            .execute(
                "UPDATE processing_jobs
                    SET run_id=?1, status='processing', error=NULL, updated_at=?2
                  WHERE session_id=?3 AND status='failed'",
                params![run_id, now.to_rfc3339(), session_id],
            )
            .map_err(|error| error.to_string())?;
        if changed == 0 {
            return Err("This conversation is not waiting for a transcription retry".into());
        }
        tx.commit().map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn fail_processing_session(&self, session_id: &str, error: &str) -> Result<(), String> {
        let now: DateTime<Utc> = SystemTime::now().into();
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|failure| failure.to_string())?;
        Self::clear_processing_artifacts_in_transaction(&tx, session_id)?;
        let changed = tx
            .execute(
                "UPDATE processing_jobs
                    SET status='failed', error=?1, updated_at=?2
                  WHERE session_id=?3",
                params![error, now.to_rfc3339(), session_id],
            )
            .map_err(|failure| failure.to_string())?;
        if changed == 0 {
            return Err("Processing job not found".into());
        }
        tx.commit().map_err(|failure| failure.to_string())?;
        Ok(())
    }

    pub fn finalize_processing_session(
        &self,
        session_id: &str,
        title: &str,
        transcript: &str,
        duration_ms: i64,
    ) -> Result<(), String> {
        let now: DateTime<Utc> = SystemTime::now().into();
        let (nonce, ct) = self.crypto.encrypt(transcript.as_bytes());
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let session_changed = tx
            .execute(
                "UPDATE sessions
                    SET title=?1, duration_ms=?2, transcript_nonce=?3, transcript_ct=?4
                  WHERE id=?5",
                params![title, duration_ms, nonce, ct, session_id],
            )
            .map_err(|error| error.to_string())?;
        if session_changed == 0 {
            return Err("Conversation not found".into());
        }
        let job_changed = tx
            .execute(
                "UPDATE processing_jobs
                    SET status='finalized', error=NULL, updated_at=?1
                  WHERE session_id=?2",
                params![now.to_rfc3339(), session_id],
            )
            .map_err(|error| error.to_string())?;
        if job_changed == 0 {
            return Err("Processing job not found".into());
        }
        tx.commit().map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn mark_processing_cleanup_failed(
        &self,
        session_id: &str,
        error: &str,
    ) -> Result<(), String> {
        let now: DateTime<Utc> = SystemTime::now().into();
        let changed = self
            .conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "UPDATE processing_jobs
                    SET status='cleanup_failed', error=?1, updated_at=?2
                  WHERE session_id=?3",
                params![error, now.to_rfc3339(), session_id],
            )
            .map_err(|failure| failure.to_string())?;
        if changed == 0 {
            return Err("Processing job not found".into());
        }
        Ok(())
    }

    pub fn complete_processing_session(&self, session_id: &str) -> Result<(), String> {
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "DELETE FROM processing_jobs WHERE session_id=?1",
                params![session_id],
            )
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub fn delete_session(&self, session_id: &str) -> Result<usize, String> {
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let referenced_speakers = {
            let mut stmt = tx
                .prepare(
                    "SELECT DISTINCT s.id, s.label
                       FROM segments sg
                       JOIN speakers s ON s.id = sg.speaker_id
                      WHERE sg.session_id=?1",
                )
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![session_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })
                .map_err(|error| error.to_string())?;
            let mut speakers = Vec::new();
            for row in rows {
                speakers.push(row.map_err(|error| error.to_string())?);
            }
            speakers
        };
        tx.execute(
            "DELETE FROM segments WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM session_recaps WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM session_agendas WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM processing_jobs WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM voice_match_decisions WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM session_import_artifacts WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        let changed = tx
            .execute("DELETE FROM sessions WHERE id=?1", params![session_id])
            .map_err(|error| error.to_string())?;
        if changed == 0 {
            return Err("Conversation not found".into());
        }
        let mut removed_voices = 0;
        for (speaker_id, label) in referenced_speakers {
            let is_unnamed = label
                .as_deref()
                .map(|value| value.trim().is_empty() || is_provisional_label(value))
                .unwrap_or(true);
            if !is_unnamed {
                continue;
            }
            let still_referenced: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM segments WHERE speaker_id=?1)",
                    params![speaker_id],
                    |row| row.get(0),
                )
                .map_err(|error| error.to_string())?;
            if still_referenced {
                continue;
            }
            tx.execute(
                "DELETE FROM embeddings WHERE speaker_id=?1",
                params![speaker_id],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "DELETE FROM speaker_samples WHERE speaker_id=?1",
                params![speaker_id],
            )
            .map_err(|error| error.to_string())?;
            removed_voices += tx
                .execute("DELETE FROM speakers WHERE id=?1", params![speaker_id])
                .map_err(|error| error.to_string())?;
        }
        tx.commit().map_err(|error| error.to_string())?;
        Ok(removed_voices)
    }

    pub fn update_session_transcript(
        &self,
        session_id: &str,
        transcript: &str,
    ) -> Result<(), String> {
        let (nonce, ct) = self.crypto.encrypt(transcript.as_bytes());
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "UPDATE sessions SET transcript_nonce=?1, transcript_ct=?2 WHERE id=?3",
                params![nonce, ct, session_id],
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn update_session_title(&self, session_id: &str, title: &str) -> Result<(), String> {
        let changed = self
            .conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "UPDATE sessions SET title=?1 WHERE id=?2",
                params![title.trim(), session_id],
            )
            .map_err(|e| e.to_string())?;
        if changed == 0 {
            return Err("Conversation not found".into());
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn list_sessions(&self) -> Result<Vec<Session>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.created_at, COALESCE(s.title, ''),
                        COALESCE(s.duration_ms, 0), s.transcript_nonce, s.transcript_ct,
                        p.status, p.error, p.run_id, p.audio_path
                   FROM sessions s
                   LEFT JOIN processing_jobs p ON p.session_id = s.id
                  ORDER BY s.created_at DESC",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let created_at: String = row.get(1)?;
                let title: String = row.get(2)?;
                let duration_ms: i64 = row.get(3)?;
                let nonce: String = row.get(4)?;
                let ct: String = row.get(5)?;
                let processing_status: Option<String> = row.get(6)?;
                let processing_error: Option<String> = row.get(7)?;
                let processing_run_id: Option<String> = row.get(8)?;
                let audio_path: Option<String> = row.get(9)?;
                Ok((
                    id,
                    created_at,
                    title,
                    duration_ms,
                    nonce,
                    ct,
                    processing_status,
                    processing_error,
                    processing_run_id,
                    audio_path,
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut sessions = Vec::new();
        for row in rows {
            let (
                id,
                created_at,
                title,
                duration_ms,
                nonce,
                ct,
                processing_status,
                processing_error,
                processing_run_id,
                audio_path,
            ) = row.map_err(|e| e.to_string())?;
            let ts = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            let transcript_bytes = self.crypto.decrypt(&nonce, &ct)?;
            let transcript = String::from_utf8(transcript_bytes).unwrap_or_default();
            let recoverable_audio = audio_path
                .as_deref()
                .map(|path| Path::new(path).is_file())
                .unwrap_or(false);
            sessions.push(Session {
                id,
                created_at: ts,
                title,
                duration_ms,
                transcript,
                processing_status,
                processing_error,
                processing_run_id,
                recoverable_audio,
            });
        }
        Ok(sessions)
    }

    pub fn list_session_summaries(&self) -> Result<Vec<SessionSummary>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT s.id, s.created_at, COALESCE(s.title, ''),
                        COALESCE(s.duration_ms, 0),
                        p.status, p.error, p.run_id, p.audio_path
                   FROM sessions s
                   LEFT JOIN processing_jobs p ON p.session_id = s.id
                  ORDER BY s.created_at DESC",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let created_at: String = row.get(1)?;
                let title: String = row.get(2)?;
                let duration_ms: i64 = row.get(3)?;
                let processing_status: Option<String> = row.get(4)?;
                let processing_error: Option<String> = row.get(5)?;
                let processing_run_id: Option<String> = row.get(6)?;
                let audio_path: Option<String> = row.get(7)?;
                Ok((
                    id,
                    created_at,
                    title,
                    duration_ms,
                    processing_status,
                    processing_error,
                    processing_run_id,
                    audio_path,
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut sessions = Vec::new();
        for row in rows {
            let (
                id,
                created_at,
                title,
                duration_ms,
                processing_status,
                processing_error,
                processing_run_id,
                audio_path,
            ) = row.map_err(|e| e.to_string())?;
            let created_at = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            let recoverable_audio = audio_path
                .as_deref()
                .map(|path| Path::new(path).is_file())
                .unwrap_or(false);
            sessions.push(SessionSummary {
                id,
                created_at,
                title,
                duration_ms,
                processing_status,
                processing_error,
                processing_run_id,
                recoverable_audio,
            });
        }
        Ok(sessions)
    }

    pub fn get_session(&self, session_id: &str) -> Result<Option<Session>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let row = conn
            .query_row(
                "SELECT s.id, s.created_at, COALESCE(s.title, ''),
                        COALESCE(s.duration_ms, 0), s.transcript_nonce, s.transcript_ct,
                        p.status, p.error, p.run_id, p.audio_path
                   FROM sessions s
                   LEFT JOIN processing_jobs p ON p.session_id = s.id
                  WHERE s.id=?1",
                params![session_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, Option<String>>(7)?,
                        row.get::<_, Option<String>>(8)?,
                        row.get::<_, Option<String>>(9)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| e.to_string())?;
        let Some((
            id,
            created_at,
            title,
            duration_ms,
            nonce,
            ct,
            processing_status,
            processing_error,
            processing_run_id,
            audio_path,
        )) = row
        else {
            return Ok(None);
        };
        let created_at = DateTime::parse_from_rfc3339(&created_at)
            .map_err(|e| e.to_string())?
            .with_timezone(&Utc);
        let transcript_bytes = self.crypto.decrypt(&nonce, &ct)?;
        let transcript = String::from_utf8(transcript_bytes).unwrap_or_default();
        let recoverable_audio = audio_path
            .as_deref()
            .map(|path| Path::new(path).is_file())
            .unwrap_or(false);
        Ok(Some(Session {
            id,
            created_at,
            title,
            duration_ms,
            transcript,
            processing_status,
            processing_error,
            processing_run_id,
            recoverable_audio,
        }))
    }

    pub fn search_session_ids(&self, query: &str) -> Result<Vec<String>, String> {
        let normalized = query.trim().to_lowercase();
        if normalized.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, COALESCE(title, ''), transcript_nonce, transcript_ct
                   FROM sessions
                  ORDER BY created_at DESC",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        let mut matching_ids = Vec::new();
        for row in rows {
            let (id, title, nonce, ct) = row.map_err(|e| e.to_string())?;
            if title.to_lowercase().contains(&normalized) {
                matching_ids.push(id);
                continue;
            }
            let transcript = String::from_utf8(self.crypto.decrypt(&nonce, &ct)?)
                .unwrap_or_default()
                .to_lowercase();
            if transcript.contains(&normalized) {
                matching_ids.push(id);
            }
        }
        Ok(matching_ids)
    }

    pub fn jamie_known_people(&self) -> Result<Vec<JamieKnownPerson>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT id, label
                   FROM speakers
                  WHERE label IS NOT NULL AND TRIM(label) <> ''
                  ORDER BY LOWER(label), label",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok(JamieKnownPerson {
                    id: row.get(0)?,
                    label: row.get(1)?,
                })
            })
            .map_err(|error| error.to_string())?;
        let mut people = Vec::new();
        for row in rows {
            let person = row.map_err(|error| error.to_string())?;
            if !is_provisional_label(&person.label) {
                people.push(person);
            }
        }
        Ok(people)
    }

    pub fn imported_meeting_fingerprints(
        &self,
        source_provider: &str,
    ) -> Result<HashSet<String>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT source_meeting_sha256
                   FROM imported_sessions
                  WHERE source_provider=?1",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params![source_provider], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        let mut fingerprints = HashSet::new();
        for row in rows {
            fingerprints.insert(row.map_err(|error| error.to_string())?);
        }
        Ok(fingerprints)
    }

    pub fn load_imported_session_artifact(
        &self,
        session_id: &str,
    ) -> Result<Option<ImportedSessionArtifact>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let row = conn
            .query_row(
                "SELECT session_id, source_provider, source_meeting_sha256, imported_at,
                        executive_summary_nonce, executive_summary_ct,
                        full_summary_nonce, full_summary_ct,
                        tasks_nonce, tasks_ct
                   FROM session_import_artifacts
                  WHERE session_id=?1",
                params![session_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, String>(8)?,
                        row.get::<_, String>(9)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some((
            session_id,
            source_provider,
            source_meeting_sha256,
            imported_at,
            executive_nonce,
            executive_ct,
            full_nonce,
            full_ct,
            tasks_nonce,
            tasks_ct,
        )) = row
        else {
            return Ok(None);
        };
        Ok(Some(ImportedSessionArtifact {
            session_id,
            source_provider,
            source_meeting_sha256,
            imported_at: DateTime::parse_from_rfc3339(&imported_at)
                .map_err(|error| error.to_string())?
                .with_timezone(&Utc),
            executive_summary: String::from_utf8(
                self.crypto.decrypt(&executive_nonce, &executive_ct)?,
            )
            .unwrap_or_default(),
            full_summary: String::from_utf8(self.crypto.decrypt(&full_nonce, &full_ct)?)
                .unwrap_or_default(),
            tasks: String::from_utf8(self.crypto.decrypt(&tasks_nonce, &tasks_ct)?)
                .unwrap_or_default(),
        }))
    }

    pub fn list_import_batches(&self) -> Result<Vec<ImportBatchSummary>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT b.id, b.source_provider, b.source_file_sha256,
                        b.imported_at, b.status, b.rolled_back_at,
                        b.meeting_count
                   FROM import_batches b
                  ORDER BY b.imported_at DESC",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            })
            .map_err(|error| error.to_string())?;
        let mut batches = Vec::new();
        for row in rows {
            let (id, provider, source_hash, imported_at, status, rolled_back_at, meeting_count) =
                row.map_err(|error| error.to_string())?;
            batches.push(ImportBatchSummary {
                id,
                source_provider: provider,
                source_file_sha256: source_hash,
                imported_at: DateTime::parse_from_rfc3339(&imported_at)
                    .map_err(|error| error.to_string())?
                    .with_timezone(&Utc),
                status,
                meeting_count: meeting_count.max(0) as usize,
                rolled_back_at: rolled_back_at
                    .map(|value| {
                        DateTime::parse_from_rfc3339(&value)
                            .map(|value| value.with_timezone(&Utc))
                            .map_err(|error| error.to_string())
                    })
                    .transpose()?,
            });
        }
        Ok(batches)
    }

    pub fn import_batch_session_ids(&self, import_id: &str) -> Result<Vec<String>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id
                   FROM imported_sessions
                  WHERE import_id=?1
                  ORDER BY session_id",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params![import_id], |row| row.get::<_, String>(0))
            .map_err(|error| error.to_string())?;
        let mut session_ids = Vec::new();
        for row in rows {
            session_ids.push(row.map_err(|error| error.to_string())?);
        }
        Ok(session_ids)
    }

    pub fn import_jamie_archive(
        &self,
        archive: &JamieArchive,
        draft: &JamieImportDraft,
    ) -> Result<JamieImportResult, String> {
        let excluded = draft
            .excluded_meetings
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let existing = self.imported_meeting_fingerprints("Jamie")?;
        let included = archive
            .meetings
            .iter()
            .filter(|meeting| !excluded.contains(meeting.source_fingerprint.as_str()))
            .collect::<Vec<_>>();
        let already_imported_meetings = included
            .iter()
            .filter(|meeting| existing.contains(&meeting.source_fingerprint))
            .count();
        let new_meetings = included
            .into_iter()
            .filter(|meeting| !existing.contains(&meeting.source_fingerprint))
            .collect::<Vec<_>>();
        if new_meetings.is_empty() {
            return Ok(JamieImportResult {
                import_id: None,
                backup_path: None,
                imported_meetings: 0,
                already_imported_meetings,
                imported_interventions: 0,
                created_people: 0,
            });
        }
        let known_people = self.jamie_known_people()?;
        let validation_errors = validate_import_draft(archive, draft, &known_people);
        if !validation_errors.is_empty() {
            return Err(format!(
                "The Jamie import review is incomplete: {}",
                validation_errors.join(" ")
            ));
        }

        let backup = self.verified_runtime_backup("pre-jamie-import")?;
        let now = Utc::now();
        let import_id = Uuid::new_v4().to_string();
        let decision_map = draft
            .identity_decisions
            .iter()
            .map(|decision| (decision.alias.as_str(), decision))
            .collect::<HashMap<_, _>>();
        let used_aliases = new_meetings
            .iter()
            .flat_map(|meeting| meeting.segments.iter())
            .map(|segment| segment.speaker_label.as_str())
            .collect::<HashSet<_>>();

        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let existing_people = {
            let mut stmt = tx
                .prepare("SELECT id, label FROM speakers WHERE label IS NOT NULL")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|error| error.to_string())?;
            let mut values = HashMap::new();
            for row in rows {
                let (id, label) = row.map_err(|error| error.to_string())?;
                values.insert(id, label);
            }
            values
        };
        let mut created_groups = HashMap::<String, (String, String)>::new();
        let mut resolved_aliases = HashMap::<String, (Option<String>, String)>::new();
        for alias in used_aliases {
            if is_generic_speaker_label(alias) {
                resolved_aliases.insert(alias.to_string(), (None, alias.to_string()));
                continue;
            }
            let decision = decision_map
                .get(alias)
                .ok_or_else(|| format!("No identity decision exists for {alias}"))?;
            match decision.action.as_str() {
                "map_existing" => {
                    let target_id = decision
                        .target_speaker_id
                        .as_ref()
                        .ok_or_else(|| format!("{alias}: missing target person"))?;
                    let label = existing_people
                        .get(target_id)
                        .ok_or_else(|| format!("{alias}: target person no longer exists"))?;
                    resolved_aliases
                        .insert(alias.to_string(), (Some(target_id.clone()), label.clone()));
                }
                "create_named" => {
                    let label =
                        display_person_name(decision.display_name.as_deref().unwrap_or_default());
                    let normalized = normalized_person_name(&label);
                    let (speaker_id, canonical_label) =
                        if let Some(existing) = created_groups.get(&normalized) {
                            existing.clone()
                        } else {
                            let speaker_id = Uuid::new_v4().to_string();
                            tx.execute(
                                "INSERT INTO speakers(id, label, created_at) VALUES(?1, ?2, ?3)",
                                params![speaker_id, label, now.to_rfc3339()],
                            )
                            .map_err(|error| error.to_string())?;
                            created_groups.insert(normalized, (speaker_id.clone(), label.clone()));
                            (speaker_id, label)
                        };
                    resolved_aliases.insert(alias.to_string(), (Some(speaker_id), canonical_label));
                }
                "unresolved" => {
                    resolved_aliases.insert(alias.to_string(), (None, alias.to_string()));
                }
                _ => return Err(format!("{alias}: identity review is incomplete")),
            }
        }

        let mut imported_session_ids = Vec::new();
        let mut imported_interventions = 0usize;
        for meeting in new_meetings {
            let session_id = Uuid::new_v4().to_string();
            let created_at = meeting
                .started_at
                .or(archive.metadata.export_date)
                .unwrap_or(now);
            let resolved_segments = meeting
                .segments
                .iter()
                .map(|segment| {
                    let (speaker_id, label) = resolved_aliases
                        .get(&segment.speaker_label)
                        .cloned()
                        .unwrap_or_else(|| (None, segment.speaker_label.clone()));
                    (segment, speaker_id, label)
                })
                .collect::<Vec<_>>();
            let transcript = resolved_segments
                .iter()
                .map(|(segment, _, label)| format!("{label}: {}", segment.text.trim()))
                .collect::<Vec<_>>()
                .join("\n");
            let (transcript_nonce, transcript_ct) = self.crypto.encrypt(transcript.as_bytes());
            tx.execute(
                "INSERT INTO sessions(
                    id, created_at, title, duration_ms, transcript_nonce, transcript_ct
                 ) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    session_id,
                    created_at.to_rfc3339(),
                    meeting.title,
                    meeting.duration_ms.max(0),
                    transcript_nonce,
                    transcript_ct,
                ],
            )
            .map_err(|error| error.to_string())?;
            for (segment, speaker_id, speaker_label) in resolved_segments {
                let segment_id = Uuid::new_v4().to_string();
                let (text_nonce, text_ct) = self.crypto.encrypt(segment.text.as_bytes());
                tx.execute(
                    "INSERT INTO segments(
                        id, session_id, start_ms, end_ms, speaker_label,
                        speaker_id, text_nonce, text_ct
                     ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    params![
                        segment_id,
                        session_id,
                        segment.start_ms,
                        segment.end_ms,
                        speaker_label,
                        speaker_id,
                        text_nonce,
                        text_ct,
                    ],
                )
                .map_err(|error| error.to_string())?;
                imported_interventions += 1;
            }
            let (executive_nonce, executive_ct) =
                self.crypto.encrypt(meeting.executive_summary.as_bytes());
            let (full_nonce, full_ct) = self.crypto.encrypt(meeting.full_summary.as_bytes());
            let (tasks_nonce, tasks_ct) = self.crypto.encrypt(meeting.tasks.as_bytes());
            tx.execute(
                "INSERT INTO session_import_artifacts(
                    session_id, source_provider, source_meeting_sha256, imported_at,
                    executive_summary_nonce, executive_summary_ct,
                    full_summary_nonce, full_summary_ct, tasks_nonce, tasks_ct
                 ) VALUES(?1, 'Jamie', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    session_id,
                    meeting.source_fingerprint,
                    now.to_rfc3339(),
                    executive_nonce,
                    executive_ct,
                    full_nonce,
                    full_ct,
                    tasks_nonce,
                    tasks_ct,
                ],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "INSERT INTO imported_sessions(
                    source_provider, source_meeting_sha256, import_id, session_id
                 ) VALUES('Jamie', ?1, ?2, ?3)",
                params![meeting.source_fingerprint, import_id, session_id],
            )
            .map_err(|error| error.to_string())?;
            imported_session_ids.push(session_id);
        }

        let created_people = created_groups.len();
        for (speaker_id, _) in created_groups.values() {
            tx.execute(
                "INSERT INTO import_created_speakers(import_id, speaker_id) VALUES(?1, ?2)",
                params![import_id, speaker_id],
            )
            .map_err(|error| error.to_string())?;
        }
        let manifest = serde_json::json!({
            "source_provider": "Jamie",
            "source_file_sha256": archive.metadata.source_sha256,
            "source_exported_at": archive.metadata.export_date,
            "importer_version": JAMIE_IMPORTER_VERSION,
            "identity_decisions": draft.identity_decisions,
            "excluded_meetings": draft.excluded_meetings,
            "imported_session_ids": imported_session_ids,
            "created_speaker_ids": created_groups
                .values()
                .map(|(id, _)| id)
                .collect::<Vec<_>>(),
        });
        let manifest =
            serde_json::to_vec(&manifest).map_err(|error| format!("manifest error: {error}"))?;
        let (manifest_nonce, manifest_ct) = self.crypto.encrypt(&manifest);
        tx.execute(
            "INSERT INTO import_batches(
                id, source_provider, source_file_sha256, source_exported_at,
                importer_version, imported_at, status, meeting_count,
                manifest_nonce, manifest_ct
             ) VALUES(?1, 'Jamie', ?2, ?3, ?4, ?5, 'imported', ?6, ?7, ?8)",
            params![
                import_id,
                archive.metadata.source_sha256,
                archive.metadata.export_date.map(|value| value.to_rfc3339()),
                JAMIE_IMPORTER_VERSION,
                now.to_rfc3339(),
                imported_session_ids.len() as i64,
                manifest_nonce,
                manifest_ct,
            ],
        )
        .map_err(|error| error.to_string())?;
        tx.commit().map_err(|error| error.to_string())?;

        Ok(JamieImportResult {
            import_id: Some(import_id),
            backup_path: Some(backup.to_string_lossy().to_string()),
            imported_meetings: imported_session_ids.len(),
            already_imported_meetings,
            imported_interventions,
            created_people,
        })
    }

    pub fn rollback_import(&self, import_id: &str) -> Result<JamieRollbackResult, String> {
        let batch_status: Option<String> = self
            .conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .query_row(
                "SELECT status FROM import_batches WHERE id=?1",
                params![import_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some(status) = batch_status else {
            return Err("Import batch not found".into());
        };
        if status != "imported" {
            return Err("That import batch has already been rolled back".into());
        }
        let backup = self.verified_runtime_backup("pre-jamie-rollback")?;
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let session_ids = {
            let mut stmt = tx
                .prepare("SELECT session_id FROM imported_sessions WHERE import_id=?1")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![import_id], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            let mut values = Vec::new();
            for row in rows {
                values.push(row.map_err(|error| error.to_string())?);
            }
            values
        };
        for session_id in &session_ids {
            for table in [
                "segments",
                "session_recaps",
                "session_agendas",
                "processing_jobs",
                "voice_match_decisions",
                "session_import_artifacts",
                "imported_sessions",
            ] {
                tx.execute(
                    &format!("DELETE FROM {table} WHERE session_id=?1"),
                    params![session_id],
                )
                .map_err(|error| error.to_string())?;
            }
            tx.execute("DELETE FROM sessions WHERE id=?1", params![session_id])
                .map_err(|error| error.to_string())?;
        }
        let created_speakers = {
            let mut stmt = tx
                .prepare("SELECT speaker_id FROM import_created_speakers WHERE import_id=?1")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![import_id], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            let mut values = Vec::new();
            for row in rows {
                values.push(row.map_err(|error| error.to_string())?);
            }
            values
        };
        let mut removed_people = 0usize;
        let mut preserved_people = 0usize;
        for speaker_id in created_speakers {
            let still_used: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM segments WHERE speaker_id=?1)",
                    params![speaker_id],
                    |row| row.get(0),
                )
                .map_err(|error| error.to_string())?;
            if still_used {
                preserved_people += 1;
                continue;
            }
            tx.execute(
                "DELETE FROM embeddings WHERE speaker_id=?1",
                params![speaker_id],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "DELETE FROM speaker_samples WHERE speaker_id=?1",
                params![speaker_id],
            )
            .map_err(|error| error.to_string())?;
            removed_people += tx
                .execute("DELETE FROM speakers WHERE id=?1", params![speaker_id])
                .map_err(|error| error.to_string())?;
        }
        tx.execute(
            "DELETE FROM import_created_speakers WHERE import_id=?1",
            params![import_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE import_batches
                SET status='rolled_back', rolled_back_at=?1
              WHERE id=?2",
            params![Utc::now().to_rfc3339(), import_id],
        )
        .map_err(|error| error.to_string())?;
        tx.commit().map_err(|error| error.to_string())?;
        Ok(JamieRollbackResult {
            import_id: import_id.to_string(),
            backup_path: backup.to_string_lossy().to_string(),
            removed_meetings: session_ids.len(),
            removed_people,
            preserved_people,
        })
    }

    fn verified_runtime_backup(&self, purpose: &str) -> Result<PathBuf, String> {
        let database_path = self.path.as_ref().ok_or_else(|| {
            "A file-backed database is required for a verified backup".to_string()
        })?;
        let parent = database_path
            .parent()
            .ok_or_else(|| "Recall database path has no parent".to_string())?;
        let stem = database_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("recall");
        let timestamp = Utc::now().format("%Y%m%d-%H%M%S");
        let suffix = &Uuid::new_v4().to_string()[..8];
        let backup = parent.join(format!("{stem}.{purpose}-{timestamp}-{suffix}.db"));
        {
            let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
            conn.execute(
                "VACUUM INTO ?1",
                params![backup.to_string_lossy().to_string()],
            )
            .map_err(|error| format!("Could not create the verified database backup: {error}"))?;
        }
        Self::restrict_file_permissions(&backup)?;
        let verification =
            Connection::open(&backup).map_err(|error| format!("Backup open failed: {error}"))?;
        let integrity: String = verification
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .map_err(|error| format!("Backup integrity check failed: {error}"))?;
        if integrity != "ok" {
            return Err(format!(
                "The database backup failed its integrity check: {integrity}"
            ));
        }
        Ok(backup)
    }

    pub fn insert_segment(
        &self,
        session_id: &str,
        start_ms: i64,
        end_ms: i64,
        speaker_id: Option<&str>,
        speaker_label: Option<&str>,
        text: &str,
    ) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let (nonce, ct) = self.crypto.encrypt(text.as_bytes());
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO segments(id, session_id, start_ms, end_ms, speaker_label, speaker_id, text_nonce, text_ct) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![id, session_id, start_ms, end_ms, speaker_label, speaker_id, nonce, ct],
            )
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn list_segments(&self, session_id: &str) -> Result<Vec<SegmentRecord>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, session_id, start_ms, end_ms, speaker_id, speaker_label, text_nonce, text_ct FROM segments WHERE session_id=?1 ORDER BY start_ms ASC")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![session_id], |row| {
                let id: String = row.get(0)?;
                let session_id: String = row.get(1)?;
                let start_ms: i64 = row.get(2)?;
                let end_ms: i64 = row.get(3)?;
                let speaker_id: Option<String> = row.get(4)?;
                let speaker_label: Option<String> = row.get(5)?;
                let nonce: String = row.get(6)?;
                let ct: String = row.get(7)?;
                Ok((
                    id,
                    session_id,
                    start_ms,
                    end_ms,
                    speaker_id,
                    speaker_label,
                    nonce,
                    ct,
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut segments = Vec::new();
        for row in rows {
            let (id, session_id, start_ms, end_ms, speaker_id, speaker_label, nonce, ct) =
                row.map_err(|e| e.to_string())?;
            let text_bytes = self.crypto.decrypt(&nonce, &ct)?;
            let text = String::from_utf8(text_bytes).unwrap_or_default();
            segments.push(SegmentRecord {
                id,
                session_id,
                start_ms,
                end_ms,
                speaker_id,
                speaker_label,
                text,
            });
        }
        Ok(segments)
    }

    pub fn upsert_agenda(
        &self,
        session_id: &str,
        source_kind: &str,
        filename: &str,
        mime_type: &str,
        content: &[u8],
    ) -> Result<AgendaRecord, String> {
        let now: DateTime<Utc> = SystemTime::now().into();
        let (nonce, ct) = self.crypto.encrypt(content);
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let session_exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sessions WHERE id=?1)",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        if !session_exists {
            return Err("Conversation not found".into());
        }
        conn.execute(
            "INSERT INTO session_agendas(
                session_id, source_kind, filename, mime_type, content_nonce, content_ct, updated_at
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(session_id) DO UPDATE SET
                source_kind=excluded.source_kind,
                filename=excluded.filename,
                mime_type=excluded.mime_type,
                content_nonce=excluded.content_nonce,
                content_ct=excluded.content_ct,
                updated_at=excluded.updated_at",
            params![
                session_id,
                source_kind,
                filename,
                mime_type,
                nonce,
                ct,
                now.to_rfc3339()
            ],
        )
        .map_err(|error| error.to_string())?;
        Ok(AgendaRecord {
            source_kind: source_kind.to_string(),
            filename: filename.to_string(),
            mime_type: mime_type.to_string(),
            content: content.to_vec(),
            updated_at: now,
        })
    }

    pub fn load_agenda(&self, session_id: &str) -> Result<Option<AgendaRecord>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let row = conn
            .query_row(
                "SELECT source_kind, filename, mime_type, content_nonce, content_ct, updated_at
                   FROM session_agendas
                  WHERE session_id=?1",
                params![session_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some((source_kind, filename, mime_type, nonce, ct, updated_at)) = row else {
            return Ok(None);
        };
        let content = self.crypto.decrypt(&nonce, &ct)?;
        let updated_at = DateTime::parse_from_rfc3339(&updated_at)
            .map_err(|error| error.to_string())?
            .with_timezone(&Utc);
        Ok(Some(AgendaRecord {
            source_kind,
            filename,
            mime_type,
            content,
            updated_at,
        }))
    }

    pub fn delete_agenda(&self, session_id: &str) -> Result<bool, String> {
        let changed = self
            .conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "DELETE FROM session_agendas WHERE session_id=?1",
                params![session_id],
            )
            .map_err(|error| error.to_string())?;
        Ok(changed > 0)
    }

    pub fn load_recap(&self, session_id: &str) -> Result<Option<RecapRecord>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let row = conn
            .query_row(
                "SELECT generated_at, model, prompt_version, schema_version, source_fingerprint,
                        payload_nonce, payload_ct, input_tokens, output_tokens
                   FROM session_recaps
                  WHERE session_id=?1",
                params![session_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, i64>(8)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some((
            generated_at,
            model,
            prompt_version,
            schema_version,
            source_fingerprint,
            nonce,
            ct,
            input_tokens,
            output_tokens,
        )) = row
        else {
            return Ok(None);
        };
        let payload_bytes = self.crypto.decrypt(&nonce, &ct)?;
        let payload = serde_json::from_slice::<RecapPayload>(&payload_bytes)
            .map_err(|error| format!("Could not read the saved recap: {error}"))?;
        let generated_at = DateTime::parse_from_rfc3339(&generated_at)
            .map_err(|error| error.to_string())?
            .with_timezone(&Utc);
        Ok(Some(RecapRecord {
            session_id: session_id.to_string(),
            generated_at,
            model,
            prompt_version,
            schema_version,
            source_fingerprint,
            payload,
            input_tokens: input_tokens.max(0) as u64,
            output_tokens: output_tokens.max(0) as u64,
        }))
    }

    pub fn update_recap_source_fingerprint(
        &self,
        session_id: &str,
        source_fingerprint: &str,
    ) -> Result<(), String> {
        let changed = self
            .conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "UPDATE session_recaps SET source_fingerprint=?1 WHERE session_id=?2",
                params![source_fingerprint, session_id],
            )
            .map_err(|error| error.to_string())?;
        if changed == 0 {
            return Err("Saved recap not found".into());
        }
        Ok(())
    }

    pub fn save_recap_and_title(&self, recap: RecapSave<'_>) -> Result<RecapRecord, String> {
        let generated_at: DateTime<Utc> = SystemTime::now().into();
        let payload_bytes = serde_json::to_vec(recap.payload)
            .map_err(|error| format!("Could not serialize the recap: {error}"))?;
        let (nonce, ct) = self.crypto.encrypt(&payload_bytes);
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let changed = tx
            .execute(
                "UPDATE sessions SET title=?1 WHERE id=?2",
                params![recap.title.trim(), recap.session_id],
            )
            .map_err(|error| error.to_string())?;
        if changed == 0 {
            return Err("Conversation not found".into());
        }
        tx.execute(
            "INSERT INTO session_recaps(
                session_id, generated_at, model, prompt_version, schema_version,
                source_fingerprint, payload_nonce, payload_ct, input_tokens, output_tokens
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(session_id) DO UPDATE SET
                generated_at=excluded.generated_at,
                model=excluded.model,
                prompt_version=excluded.prompt_version,
                schema_version=excluded.schema_version,
                source_fingerprint=excluded.source_fingerprint,
                payload_nonce=excluded.payload_nonce,
                payload_ct=excluded.payload_ct,
                input_tokens=excluded.input_tokens,
                output_tokens=excluded.output_tokens",
            params![
                recap.session_id,
                generated_at.to_rfc3339(),
                recap.model,
                recap.prompt_version,
                recap.schema_version,
                recap.source_fingerprint,
                nonce,
                ct,
                recap.input_tokens.min(i64::MAX as u64) as i64,
                recap.output_tokens.min(i64::MAX as u64) as i64,
            ],
        )
        .map_err(|error| error.to_string())?;
        tx.commit().map_err(|error| error.to_string())?;
        Ok(RecapRecord {
            session_id: recap.session_id.to_string(),
            generated_at,
            model: recap.model.to_string(),
            prompt_version: recap.prompt_version.to_string(),
            schema_version: recap.schema_version.to_string(),
            source_fingerprint: recap.source_fingerprint.to_string(),
            payload: recap.payload.clone(),
            input_tokens: recap.input_tokens,
            output_tokens: recap.output_tokens,
        })
    }

    pub fn update_segment_text(&self, segment_id: &str, text: &str) -> Result<(), String> {
        let (nonce, ct) = self.crypto.encrypt(text.trim().as_bytes());
        let changed = self
            .conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "UPDATE segments SET text_nonce=?1, text_ct=?2 WHERE id=?3",
                params![nonce, ct, segment_id],
            )
            .map_err(|e| e.to_string())?;
        if changed == 0 {
            return Err("Transcript intervention not found".into());
        }
        Ok(())
    }

    pub fn assign_segment_speaker(
        &self,
        segment_id: &str,
        speaker_id: Option<&str>,
    ) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let label: Option<String> = match speaker_id {
            Some(id) => conn
                .query_row(
                    "SELECT label FROM speakers WHERE id=?1",
                    params![id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|e| e.to_string())?
                .flatten(),
            None => None,
        };
        let changed = conn
            .execute(
                "UPDATE segments SET speaker_id=?1, speaker_label=?2 WHERE id=?3",
                params![speaker_id, label, segment_id],
            )
            .map_err(|e| e.to_string())?;
        if changed == 0 {
            return Err("Transcript intervention not found".into());
        }
        Ok(())
    }

    pub fn insert_speaker(&self, label: Option<&str>) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        if let Some(label) =
            label.filter(|value| !value.trim().is_empty() && !is_provisional_label(value))
        {
            let normalized = normalized_person_name(label);
            let mut stmt = conn
                .prepare("SELECT label FROM speakers WHERE label IS NOT NULL")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            for row in rows {
                let existing = row.map_err(|error| error.to_string())?;
                if !existing.trim().is_empty()
                    && !is_provisional_label(&existing)
                    && normalized_person_name(&existing) == normalized
                {
                    return Err(format!(
                        "A person named {} already exists. Assign or merge this voice with that profile instead.",
                        display_person_name(&existing)
                    ));
                }
            }
        }
        conn.execute(
            "INSERT INTO speakers(id, label, created_at) VALUES(?1, ?2, ?3)",
            params![id, label, now.to_rfc3339()],
        )
        .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn next_voice_label(&self) -> Result<String, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let maximum: Option<i64> = conn
            .query_row(
                "SELECT MAX(CAST(SUBSTR(label, 6) AS INTEGER)) FROM speakers WHERE label GLOB 'VOICE[0-9]*'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        Ok(format!("VOICE{}", maximum.unwrap_or(0) + 1))
    }

    pub fn create_speaker_for_unattributed_segments(
        &self,
        session_id: &str,
    ) -> Result<(String, String, usize), String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let session_exists: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sessions WHERE id=?1)",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        if !session_exists {
            return Err("Conversation not found".into());
        }
        let processing: bool = tx
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM processing_jobs
                     WHERE session_id=?1 AND status IN ('queued', 'processing')
                 )",
                params![session_id],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        if processing {
            return Err("Wait for final transcription before grouping unknown voices".into());
        }
        let maximum: Option<i64> = tx
            .query_row(
                "SELECT MAX(CAST(SUBSTR(label, 6) AS INTEGER))
                   FROM speakers WHERE label GLOB 'VOICE[0-9]*'",
                [],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        let label = format!("VOICE{}", maximum.unwrap_or(0) + 1);
        tx.execute(
            "INSERT INTO speakers(id, label, created_at) VALUES(?1, ?2, ?3)",
            params![id, label, now.to_rfc3339()],
        )
        .map_err(|error| error.to_string())?;
        let changed = tx
            .execute(
                "UPDATE segments
                    SET speaker_id=?1, speaker_label=?2
                  WHERE session_id=?3 AND speaker_id IS NULL",
                params![id, label, session_id],
            )
            .map_err(|error| error.to_string())?;
        if changed == 0 {
            return Err("This conversation has no unknown interventions".into());
        }
        tx.commit().map_err(|error| error.to_string())?;
        Ok((id, label, changed))
    }

    pub fn list_speakers(&self) -> Result<Vec<Speaker>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, label, created_at FROM speakers ORDER BY created_at ASC")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let label: Option<String> = row.get(1)?;
                let created_at: String = row.get(2)?;
                Ok((id, label, created_at))
            })
            .map_err(|e| e.to_string())?;

        let mut speakers = Vec::new();
        for row in rows {
            let (id, label, created_at) = row.map_err(|e| e.to_string())?;
            let created_at = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            speakers.push(Speaker {
                id,
                label,
                created_at,
            });
        }
        Ok(speakers)
    }

    pub fn list_speakers_with_stats(&self) -> Result<Vec<SpeakerStats>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "WITH sample_stats AS (
                        SELECT speaker_id, COUNT(1) AS sample_count
                          FROM speaker_samples
                         GROUP BY speaker_id
                     ),
                     embedding_stats AS (
                        SELECT speaker_id, COUNT(1) AS embedding_count
                          FROM embeddings
                         WHERE model_version=?1 AND is_reference=1
                         GROUP BY speaker_id
                     ),
                     segment_stats AS (
                        SELECT sg.speaker_id,
                               COUNT(DISTINCT sg.session_id) AS conversation_count,
                               MAX(se.created_at) AS last_seen_at
                          FROM segments sg
                          JOIN sessions se ON se.id=sg.session_id
                         WHERE sg.speaker_id IS NOT NULL
                         GROUP BY sg.speaker_id
                     )
                 SELECT s.id, s.label, s.created_at,
                        COALESCE(sm.sample_count, 0),
                        COALESCE(em.embedding_count, 0),
                        COALESCE(sg.conversation_count, 0),
                        sg.last_seen_at
                   FROM speakers s
                   LEFT JOIN sample_stats sm ON sm.speaker_id=s.id
                   LEFT JOIN embedding_stats em ON em.speaker_id=s.id
                   LEFT JOIN segment_stats sg ON sg.speaker_id=s.id
                  ORDER BY COALESCE(sg.last_seen_at, s.created_at) DESC",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![crate::embedding::EMBEDDING_VERSION], |row| {
                let id: String = row.get(0)?;
                let label: Option<String> = row.get(1)?;
                let created_at: String = row.get(2)?;
                let sample_count: i64 = row.get(3)?;
                let embedding_count: i64 = row.get(4)?;
                let conversation_count: i64 = row.get(5)?;
                let last_seen_at: Option<String> = row.get(6)?;
                Ok((
                    id,
                    label,
                    created_at,
                    sample_count,
                    embedding_count,
                    conversation_count,
                    last_seen_at,
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut raw = Vec::new();
        let mut normalized_name_groups: HashMap<String, usize> = HashMap::new();
        for row in rows {
            let (
                id,
                label,
                created_at,
                sample_count,
                embedding_count,
                conversation_count,
                last_seen_at,
            ) = row.map_err(|e| e.to_string())?;
            let created_at = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            let last_seen_at = last_seen_at
                .map(|value| {
                    DateTime::parse_from_rfc3339(&value)
                        .map(|date| date.with_timezone(&Utc))
                        .map_err(|error| error.to_string())
                })
                .transpose()?;
            if let Some(normalized) = label
                .as_deref()
                .filter(|value| !value.trim().is_empty() && !is_provisional_label(value))
                .map(normalized_person_name)
            {
                *normalized_name_groups.entry(normalized).or_default() += 1;
            }
            raw.push((
                id,
                label,
                created_at,
                last_seen_at,
                sample_count.max(0) as usize,
                embedding_count.max(0) as usize,
                conversation_count.max(0) as usize,
            ));
        }

        let mut suggestions = HashMap::<String, VoiceMatchSuggestion>::new();
        let mut suggestion_stmt = conn
            .prepare(
                "SELECT d.resulting_speaker_id,
                        d.id,
                        d.best_speaker_id,
                        target.label,
                        d.best_score,
                        runner_up.label,
                        d.runner_up_score,
                        d.support_count,
                        d.reason
                   FROM voice_match_decisions d
                   JOIN speakers target ON target.id=d.best_speaker_id
                   LEFT JOIN speakers runner_up ON runner_up.id=d.runner_up_speaker_id
                  WHERE d.resulting_speaker_id IS NOT NULL
                    AND d.decision='suggested'
                    AND d.resolved_at IS NULL
                  ORDER BY d.resulting_speaker_id, d.created_at DESC, d.id DESC",
            )
            .map_err(|error| error.to_string())?;
        let suggestion_rows = suggestion_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    VoiceMatchSuggestion {
                        decision_id: row.get(1)?,
                        speaker_id: row.get(2)?,
                        label: row.get(3)?,
                        score: row.get::<_, f64>(4)? as f32,
                        runner_up_label: row.get(5)?,
                        runner_up_score: row.get::<_, Option<f64>>(6)?.map(|value| value as f32),
                        support_count: row.get::<_, i64>(7)?.max(0) as usize,
                        reason: row.get(8)?,
                    },
                ))
            })
            .map_err(|error| error.to_string())?;
        for row in suggestion_rows {
            let (speaker_id, suggestion) = row.map_err(|error| error.to_string())?;
            suggestions.entry(speaker_id).or_insert(suggestion);
        }

        let mut speakers = Vec::with_capacity(raw.len());
        for (
            id,
            label,
            created_at,
            last_seen_at,
            sample_count,
            embedding_count,
            conversation_count,
        ) in raw
        {
            let duplicate_name_count = label
                .as_deref()
                .filter(|value| !value.trim().is_empty() && !is_provisional_label(value))
                .and_then(|value| normalized_name_groups.get(&normalized_person_name(value)))
                .copied()
                .unwrap_or(0);
            speakers.push(SpeakerStats {
                likely_match: suggestions.remove(&id),
                id,
                label,
                created_at,
                last_seen_at,
                sample_count,
                embedding_count,
                conversation_count,
                duplicate_name_conflict: duplicate_name_count > 1,
                duplicate_name_count,
            });
        }
        Ok(speakers)
    }

    pub fn list_identity_profiles(
        &self,
        search: &str,
        status: &str,
        page: usize,
        page_size: usize,
    ) -> Result<IdentityProfilePage, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "WITH sample_stats AS (
                        SELECT speaker_id, COUNT(1) AS sample_count
                          FROM speaker_samples
                         GROUP BY speaker_id
                     ),
                     embedding_stats AS (
                        SELECT speaker_id,
                               SUM(CASE
                                     WHEN model_version=?1 AND is_reference=1
                                     THEN 1 ELSE 0
                                   END) AS active_voiceprints,
                               SUM(CASE
                                     WHEN model_version<>?1 OR model_version IS NULL
                                          OR is_reference<>1
                                     THEN 1 ELSE 0
                                   END) AS inactive_voiceprints
                          FROM embeddings
                         GROUP BY speaker_id
                     ),
                     segment_stats AS (
                        SELECT sg.speaker_id,
                               COUNT(DISTINCT sg.session_id) AS conversation_count,
                               COUNT(1) AS intervention_count,
                               MAX(se.created_at) AS last_seen_at
                          FROM segments sg
                          JOIN sessions se ON se.id=sg.session_id
                         WHERE sg.speaker_id IS NOT NULL
                         GROUP BY sg.speaker_id
                     ),
                     imported_profiles AS (
                        SELECT DISTINCT speaker_id FROM import_created_speakers
                     )
                 SELECT s.id, s.label, s.created_at, sg.last_seen_at,
                        COALESCE(sm.sample_count, 0),
                        COALESCE(em.active_voiceprints, 0),
                        COALESCE(em.inactive_voiceprints, 0),
                        COALESCE(sg.conversation_count, 0),
                        COALESCE(sg.intervention_count, 0),
                        CASE WHEN ip.speaker_id IS NULL THEN 0 ELSE 1 END
                   FROM speakers s
                   LEFT JOIN sample_stats sm ON sm.speaker_id=s.id
                   LEFT JOIN embedding_stats em ON em.speaker_id=s.id
                   LEFT JOIN segment_stats sg ON sg.speaker_id=s.id
                   LEFT JOIN imported_profiles ip ON ip.speaker_id=s.id",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params![crate::embedding::EMBEDDING_VERSION], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                ))
            })
            .map_err(|error| error.to_string())?;

        let mut profiles = Vec::new();
        let mut duplicate_counts = HashMap::<String, usize>::new();
        for row in rows {
            let (
                id,
                label,
                created_at,
                last_seen_at,
                sample_count,
                active_voiceprint_count,
                inactive_voiceprint_count,
                conversation_count,
                intervention_count,
                imported,
            ) = row.map_err(|error| error.to_string())?;
            let label = label
                .map(|value| display_person_name(&value))
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "Unnamed voice".into());
            let provisional = is_provisional_label(&label);
            if !provisional && label != "Unnamed voice" {
                *duplicate_counts
                    .entry(normalized_person_name(&label))
                    .or_default() += 1;
            }
            profiles.push(IdentityProfileRow {
                id,
                label,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map_err(|error| error.to_string())?
                    .with_timezone(&Utc),
                last_seen_at: last_seen_at
                    .map(|value| {
                        DateTime::parse_from_rfc3339(&value)
                            .map(|date| date.with_timezone(&Utc))
                            .map_err(|error| error.to_string())
                    })
                    .transpose()?,
                sample_count: sample_count.max(0) as usize,
                active_voiceprint_count: active_voiceprint_count.max(0) as usize,
                inactive_voiceprint_count: inactive_voiceprint_count.max(0) as usize,
                conversation_count: conversation_count.max(0) as usize,
                intervention_count: intervention_count.max(0) as usize,
                provisional,
                imported: imported != 0,
                duplicate_name_conflict: false,
                duplicate_name_count: 0,
            });
        }
        for profile in &mut profiles {
            if !profile.provisional && profile.label != "Unnamed voice" {
                profile.duplicate_name_count = duplicate_counts
                    .get(&normalized_person_name(&profile.label))
                    .copied()
                    .unwrap_or(0);
                profile.duplicate_name_conflict = profile.duplicate_name_count > 1;
            }
        }

        let search = search.trim().to_lowercase();
        let status = status.trim();
        profiles.retain(|profile| {
            let search_matches =
                search.is_empty() || profile.label.to_lowercase().contains(&search);
            let status_matches = match status {
                "" | "all" => true,
                "named" => !profile.provisional && profile.label != "Unnamed voice",
                "provisional" => profile.provisional,
                "no_voiceprint" => profile.active_voiceprint_count == 0,
                "conflict" => profile.duplicate_name_conflict,
                "imported" => profile.imported,
                _ => false,
            };
            search_matches && status_matches
        });
        profiles.sort_by(|left, right| {
            natural_label_cmp(&left.label, &right.label).then_with(|| left.id.cmp(&right.id))
        });
        let total = profiles.len();
        let (page, page_size, page_count) = bounded_page(page, page_size, total);
        let offset = (page - 1) * page_size;
        let items = profiles.into_iter().skip(offset).take(page_size).collect();
        Ok(IdentityProfilePage {
            items,
            total,
            page,
            page_size,
            page_count,
        })
    }

    pub fn list_unassigned_identities(
        &self,
        search: &str,
        status: &str,
        page: usize,
        page_size: usize,
    ) -> Result<UnassignedIdentityPage, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT sg.session_id,
                        sg.speaker_label,
                        COALESCE(NULLIF(TRIM(sg.speaker_label), ''), 'Unknown speaker'),
                        COALESCE(se.title, ''),
                        se.created_at,
                        COUNT(1),
                        COALESCE(MIN(sg.start_ms), 0),
                        COALESCE(MAX(sg.end_ms), 0)
                   FROM segments sg
                   JOIN sessions se ON se.id=sg.session_id
                  WHERE sg.speaker_id IS NULL
                  GROUP BY sg.session_id, sg.speaker_label
                  ORDER BY se.created_at DESC, sg.speaker_label",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            })
            .map_err(|error| error.to_string())?;
        let mut groups = Vec::new();
        for row in rows {
            let (
                session_id,
                speaker_label,
                display_label,
                session_title,
                session_created_at,
                intervention_count,
                first_start_ms,
                last_end_ms,
            ) = row.map_err(|error| error.to_string())?;
            groups.push(UnassignedIdentityRow {
                key: UnassignedIdentityKey {
                    session_id,
                    speaker_label,
                },
                generic: is_generic_speaker_label(&display_label),
                display_label,
                session_title: if session_title.trim().is_empty() {
                    "Untitled conversation".into()
                } else {
                    session_title
                },
                session_created_at: DateTime::parse_from_rfc3339(&session_created_at)
                    .map_err(|error| error.to_string())?
                    .with_timezone(&Utc),
                intervention_count: intervention_count.max(0) as usize,
                first_start_ms,
                last_end_ms,
            });
        }
        let search = search.trim().to_lowercase();
        let status = status.trim();
        groups.retain(|group| {
            let search_matches = search.is_empty()
                || group.display_label.to_lowercase().contains(&search)
                || group.session_title.to_lowercase().contains(&search);
            let status_matches = match status {
                "" | "all" => true,
                "generic" => group.generic,
                "labelled" => !group.generic,
                _ => false,
            };
            search_matches && status_matches
        });
        groups.sort_by(|left, right| {
            natural_label_cmp(&left.display_label, &right.display_label)
                .then_with(|| right.session_created_at.cmp(&left.session_created_at))
                .then_with(|| left.key.session_id.cmp(&right.key.session_id))
        });
        let total = groups.len();
        let (page, page_size, page_count) = bounded_page(page, page_size, total);
        let offset = (page - 1) * page_size;
        let items = groups.into_iter().skip(offset).take(page_size).collect();
        Ok(UnassignedIdentityPage {
            items,
            total,
            page,
            page_size,
            page_count,
        })
    }

    fn identity_profile_row_for_id(
        conn: &Connection,
        speaker_id: &str,
    ) -> Result<Option<IdentityProfileRow>, String> {
        let row = conn
            .query_row(
                "SELECT s.id,
                        s.label,
                        s.created_at,
                        (SELECT MAX(se.created_at)
                           FROM segments sg
                           JOIN sessions se ON se.id=sg.session_id
                          WHERE sg.speaker_id=s.id),
                        (SELECT COUNT(1)
                           FROM speaker_samples sm
                          WHERE sm.speaker_id=s.id),
                        (SELECT COUNT(1)
                           FROM embeddings e
                          WHERE e.speaker_id=s.id
                            AND e.model_version=?2
                            AND e.is_reference=1),
                        (SELECT COUNT(1)
                           FROM embeddings e
                          WHERE e.speaker_id=s.id
                            AND (
                                 e.model_version<>?2 OR e.model_version IS NULL
                                 OR e.is_reference<>1
                            )),
                        (SELECT COUNT(DISTINCT sg.session_id)
                           FROM segments sg
                          WHERE sg.speaker_id=s.id),
                        (SELECT COUNT(1)
                           FROM segments sg
                          WHERE sg.speaker_id=s.id),
                        EXISTS(
                           SELECT 1 FROM import_created_speakers ip
                            WHERE ip.speaker_id=s.id
                        )
                   FROM speakers s
                  WHERE s.id=?1",
                params![speaker_id, crate::embedding::EMBEDDING_VERSION],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, i64>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, i64>(9)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some((
            id,
            label,
            created_at,
            last_seen_at,
            sample_count,
            active_voiceprint_count,
            inactive_voiceprint_count,
            conversation_count,
            intervention_count,
            imported,
        )) = row
        else {
            return Ok(None);
        };
        let label = label
            .map(|value| display_person_name(&value))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "Unnamed voice".into());
        let provisional = is_provisional_label(&label);
        let duplicate_name_count = if provisional || label == "Unnamed voice" {
            0
        } else {
            let normalized = normalized_person_name(&label);
            let mut stmt = conn
                .prepare("SELECT label FROM speakers WHERE label IS NOT NULL")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            let mut count = 0usize;
            for row in rows {
                let candidate = row.map_err(|error| error.to_string())?;
                if !candidate.trim().is_empty()
                    && !is_provisional_label(&candidate)
                    && normalized_person_name(&candidate) == normalized
                {
                    count += 1;
                }
            }
            count
        };
        Ok(Some(IdentityProfileRow {
            id,
            label,
            created_at: DateTime::parse_from_rfc3339(&created_at)
                .map_err(|error| error.to_string())?
                .with_timezone(&Utc),
            last_seen_at: last_seen_at
                .map(|value| {
                    DateTime::parse_from_rfc3339(&value)
                        .map(|date| date.with_timezone(&Utc))
                        .map_err(|error| error.to_string())
                })
                .transpose()?,
            sample_count: sample_count.max(0) as usize,
            active_voiceprint_count: active_voiceprint_count.max(0) as usize,
            inactive_voiceprint_count: inactive_voiceprint_count.max(0) as usize,
            conversation_count: conversation_count.max(0) as usize,
            intervention_count: intervention_count.max(0) as usize,
            provisional,
            imported: imported != 0,
            duplicate_name_conflict: duplicate_name_count > 1,
            duplicate_name_count,
        }))
    }

    fn unassigned_identity_row_for_key(
        conn: &Connection,
        key: &UnassignedIdentityKey,
    ) -> Result<Option<UnassignedIdentityRow>, String> {
        let row = conn
            .query_row(
                "SELECT COALESCE(NULLIF(TRIM(sg.speaker_label), ''), 'Unknown speaker'),
                        COALESCE(se.title, ''),
                        se.created_at,
                        COUNT(1),
                        COALESCE(MIN(sg.start_ms), 0),
                        COALESCE(MAX(sg.end_ms), 0)
                   FROM segments sg
                   JOIN sessions se ON se.id=sg.session_id
                  WHERE sg.session_id=?1
                    AND sg.speaker_id IS NULL
                    AND (
                         (?2 IS NULL AND sg.speaker_label IS NULL)
                         OR sg.speaker_label=?2
                    )
                  GROUP BY sg.session_id, sg.speaker_label",
                params![key.session_id, key.speaker_label],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let Some((
            display_label,
            session_title,
            session_created_at,
            intervention_count,
            first_start_ms,
            last_end_ms,
        )) = row
        else {
            return Ok(None);
        };
        Ok(Some(UnassignedIdentityRow {
            key: key.clone(),
            generic: is_generic_speaker_label(&display_label),
            display_label,
            session_title: if session_title.trim().is_empty() {
                "Untitled conversation".into()
            } else {
                session_title
            },
            session_created_at: DateTime::parse_from_rfc3339(&session_created_at)
                .map_err(|error| error.to_string())?
                .with_timezone(&Utc),
            intervention_count: intervention_count.max(0) as usize,
            first_start_ms,
            last_end_ms,
        }))
    }

    fn identity_consolidation_preview_with_connection(
        conn: &Connection,
        request: &IdentityConsolidationRequest,
    ) -> Result<IdentityConsolidationPreview, String> {
        let final_label = display_person_name(&request.final_label);
        if final_label.is_empty() {
            return Err("The canonical person needs a display name".into());
        }
        if is_provisional_label(&final_label)
            || matches!(
                final_label.to_lowercase().as_str(),
                "unknown speaker" | "unnamed voice"
            )
        {
            return Err("The canonical person needs a human-readable name".into());
        }

        let profile_ids = request
            .profile_ids
            .iter()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        if profile_ids.len() != request.profile_ids.len()
            || profile_ids.iter().copied().collect::<HashSet<_>>().len() != profile_ids.len()
        {
            return Err("The selected profile list contains an invalid or duplicate ID".into());
        }
        let group_keys = request
            .unassigned_groups
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        if group_keys.len() != request.unassigned_groups.len() {
            return Err("The selected unassigned-speaker list contains duplicates".into());
        }
        if profile_ids.is_empty() && group_keys.is_empty() {
            return Err("Select at least one profile or unassigned speaker group".into());
        }

        let target_speaker_id = request
            .target_speaker_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if profile_ids.is_empty() {
            if let Some(target_id) = target_speaker_id {
                if Self::identity_profile_row_for_id(conn, target_id)?.is_none() {
                    return Err("The canonical target no longer exists".into());
                }
            }
        } else {
            let Some(target_id) = target_speaker_id else {
                return Err("Choose one selected profile as the canonical target".into());
            };
            if !profile_ids.contains(&target_id) {
                return Err("The canonical target must be one of the selected profiles".into());
            }
        }
        if profile_ids.len() == 1
            && group_keys.is_empty()
            && target_speaker_id == profile_ids.first().copied()
        {
            return Err("Select another profile or an unassigned group to consolidate".into());
        }

        let selected_ids = profile_ids.iter().copied().collect::<HashSet<_>>();
        let normalized_final = normalized_person_name(&final_label);
        let mut names = conn
            .prepare("SELECT id, label FROM speakers WHERE label IS NOT NULL")
            .map_err(|error| error.to_string())?;
        let rows = names
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| error.to_string())?;
        for row in rows {
            let (candidate_id, candidate_label) = row.map_err(|error| error.to_string())?;
            if selected_ids.contains(candidate_id.as_str())
                || target_speaker_id == Some(candidate_id.as_str())
            {
                continue;
            }
            if !candidate_label.trim().is_empty()
                && !is_provisional_label(&candidate_label)
                && normalized_person_name(&candidate_label) == normalized_final
            {
                return Err(format!(
                    "{} already belongs to another profile. Include that profile as the canonical target or choose another name.",
                    display_person_name(&candidate_label)
                ));
            }
        }

        let mut source_profiles = Vec::new();
        for profile_id in &profile_ids {
            let profile = Self::identity_profile_row_for_id(conn, profile_id)?
                .ok_or_else(|| format!("Selected profile {profile_id} no longer exists"))?;
            source_profiles.push(profile);
        }
        source_profiles.sort_by(|left, right| {
            natural_label_cmp(&left.label, &right.label).then_with(|| left.id.cmp(&right.id))
        });

        let mut unassigned_groups = Vec::new();
        for key in &request.unassigned_groups {
            let group = Self::unassigned_identity_row_for_key(conn, key)?.ok_or_else(|| {
                format!(
                    "The selected {} group in conversation {} is no longer unassigned",
                    key.speaker_label.as_deref().unwrap_or("Unknown speaker"),
                    key.session_id
                )
            })?;
            unassigned_groups.push(group);
        }
        unassigned_groups.sort_by(|left, right| {
            natural_label_cmp(&left.display_label, &right.display_label)
                .then_with(|| left.key.session_id.cmp(&right.key.session_id))
        });

        let mut all_profile_ids = selected_ids
            .iter()
            .map(|value| (*value).to_string())
            .collect::<HashSet<_>>();
        if let Some(target_id) = target_speaker_id {
            all_profile_ids.insert(target_id.to_string());
        }
        let mut affected_sessions = HashSet::<String>::new();
        let mut affected_intervention_count = 0usize;
        for profile_id in &all_profile_ids {
            let mut stmt = conn
                .prepare(
                    "SELECT session_id, COUNT(1)
                       FROM segments
                      WHERE speaker_id=?1
                      GROUP BY session_id",
                )
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![profile_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })
                .map_err(|error| error.to_string())?;
            for row in rows {
                let (session_id, count) = row.map_err(|error| error.to_string())?;
                affected_sessions.insert(session_id);
                affected_intervention_count += count.max(0) as usize;
            }
        }
        for group in &unassigned_groups {
            affected_sessions.insert(group.key.session_id.clone());
            affected_intervention_count += group.intervention_count;
        }
        let mut affected_session_ids = affected_sessions.into_iter().collect::<Vec<_>>();
        affected_session_ids.sort();

        let mut stale_recap_count = 0usize;
        for session_id in &affected_session_ids {
            let exists: bool = conn
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM session_recaps WHERE session_id=?1)",
                    params![session_id],
                    |row| row.get(0),
                )
                .map_err(|error| error.to_string())?;
            stale_recap_count += usize::from(exists);
        }
        let active_voiceprint_count = source_profiles
            .iter()
            .map(|profile| profile.active_voiceprint_count)
            .sum();
        let inactive_voiceprint_count = source_profiles
            .iter()
            .map(|profile| profile.inactive_voiceprint_count)
            .sum();
        let samples_to_delete = source_profiles
            .iter()
            .map(|profile| profile.sample_count)
            .sum();
        let imported_source_profile_count = source_profiles
            .iter()
            .filter(|profile| {
                target_speaker_id
                    .map(|target| profile.id != target)
                    .unwrap_or(true)
                    && profile.imported
            })
            .count();
        let mut warnings = Vec::new();
        if stale_recap_count > 0 {
            warnings.push(format!(
                "{stale_recap_count} saved recap{} will be marked out of date.",
                if stale_recap_count == 1 { "" } else { "s" }
            ));
        }
        if inactive_voiceprint_count > 0 {
            warnings.push(format!(
                "{inactive_voiceprint_count} inactive or incompatible voiceprint{} will remain quarantined.",
                if inactive_voiceprint_count == 1 {
                    ""
                } else {
                    "s"
                }
            ));
        }
        if samples_to_delete > 0 {
            warnings.push(format!(
                "{samples_to_delete} temporary voice sample{} will be deleted for privacy.",
                if samples_to_delete == 1 { "" } else { "s" }
            ));
        }
        if imported_source_profile_count > 0 {
            warnings.push(
                "Imported-person ownership will be updated so a later archive rollback cannot remove non-import history or voice data."
                    .into(),
            );
        }
        Ok(IdentityConsolidationPreview {
            target_speaker_id: target_speaker_id.map(str::to_string),
            target_label: final_label,
            source_profiles,
            unassigned_groups,
            affected_conversation_count: affected_session_ids.len(),
            affected_intervention_count,
            stale_recap_count,
            active_voiceprint_count,
            inactive_voiceprint_count,
            samples_to_delete,
            imported_source_profile_count,
            creates_new_person: target_speaker_id.is_none(),
            affected_session_ids,
            warnings,
        })
    }

    pub fn preview_identity_consolidation(
        &self,
        request: &IdentityConsolidationRequest,
    ) -> Result<IdentityConsolidationPreview, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        Self::identity_consolidation_preview_with_connection(&conn, request)
    }

    fn rebuild_session_transcripts_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        crypto: &Crypto,
        session_ids: &[String],
    ) -> Result<(), String> {
        for session_id in session_ids {
            let mut stmt = tx
                .prepare(
                    "SELECT speaker_label, text_nonce, text_ct
                       FROM segments
                      WHERE session_id=?1
                      ORDER BY start_ms, id",
                )
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![session_id], |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(|error| error.to_string())?;
            let mut lines = Vec::new();
            for row in rows {
                let (speaker_label, nonce, ciphertext) = row.map_err(|error| error.to_string())?;
                let text =
                    String::from_utf8(crypto.decrypt(&nonce, &ciphertext)?).unwrap_or_default();
                if text.trim().is_empty() {
                    continue;
                }
                lines.push(format!(
                    "{}: {}",
                    speaker_label.as_deref().unwrap_or("Unknown speaker"),
                    text.trim()
                ));
            }
            drop(stmt);
            let (nonce, ciphertext) = crypto.encrypt(lines.join("\n").as_bytes());
            tx.execute(
                "UPDATE sessions
                    SET transcript_nonce=?1, transcript_ct=?2
                  WHERE id=?3",
                params![nonce, ciphertext, session_id],
            )
            .map_err(|error| error.to_string())?;
        }
        Ok(())
    }

    fn session_ids_for_profiles_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        speaker_ids: &[&str],
    ) -> Result<Vec<String>, String> {
        if speaker_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat_n("?", speaker_ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let mut stmt = tx
            .prepare(&format!(
                "SELECT DISTINCT session_id
                   FROM segments
                  WHERE speaker_id IN ({placeholders})
                  ORDER BY session_id"
            ))
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params_from_iter(speaker_ids.iter()), |row| {
                row.get::<_, String>(0)
            })
            .map_err(|error| error.to_string())?;
        let mut session_ids = Vec::new();
        for row in rows {
            session_ids.push(row.map_err(|error| error.to_string())?);
        }
        Ok(session_ids)
    }

    pub fn consolidate_identities(
        &self,
        request: &IdentityConsolidationRequest,
        expected_affected_session_ids: &[String],
    ) -> Result<IdentityConsolidationResult, String> {
        let initial_preview = self.preview_identity_consolidation(request)?;
        if initial_preview.affected_session_ids != expected_affected_session_ids {
            return Err(
                "The affected conversations changed after the impact preview. Review the operation again."
                    .into(),
            );
        }
        let backup = self.verified_runtime_backup("pre-identity-merge")?;
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let preview = Self::identity_consolidation_preview_with_connection(&tx, request)?;
        if preview.affected_session_ids != expected_affected_session_ids {
            return Err(
                "The affected conversations changed after the impact preview. Review the operation again."
                    .into(),
            );
        }
        let now = Utc::now().to_rfc3339();
        let target_speaker_id = if let Some(target_id) = preview.target_speaker_id.as_deref() {
            target_id.to_string()
        } else {
            let id = Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO speakers(id, label, created_at) VALUES(?1, ?2, ?3)",
                params![id, preview.target_label, now],
            )
            .map_err(|error| error.to_string())?;
            id
        };

        let source_ids = request
            .profile_ids
            .iter()
            .filter(|speaker_id| speaker_id.as_str() != target_speaker_id)
            .cloned()
            .collect::<Vec<_>>();
        let mut provenance_ids = source_ids.iter().cloned().collect::<HashSet<_>>();
        provenance_ids.insert(target_speaker_id.clone());
        let mut import_owners = HashMap::<String, HashSet<String>>::new();
        for speaker_id in &provenance_ids {
            let mut stmt = tx
                .prepare("SELECT import_id FROM import_created_speakers WHERE speaker_id=?1")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![speaker_id], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            let mut owners = HashSet::new();
            for row in rows {
                owners.insert(row.map_err(|error| error.to_string())?);
            }
            import_owners.insert(speaker_id.clone(), owners);
        }
        let mut participating_imports = import_owners
            .values()
            .flat_map(|owners| owners.iter().cloned())
            .collect::<HashSet<_>>();
        let any_unowned_source = source_ids
            .iter()
            .any(|source_id| import_owners.get(source_id).is_none_or(HashSet::is_empty));
        let mut target_receives_audio_or_non_import_history = false;
        for speaker_id in &provenance_ids {
            let has_audio: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM embeddings WHERE speaker_id=?1)",
                    params![speaker_id],
                    |row| row.get(0),
                )
                .map_err(|error| error.to_string())?;
            let has_non_import_history: bool = tx
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1
                          FROM segments sg
                          LEFT JOIN imported_sessions imported
                            ON imported.session_id=sg.session_id
                         WHERE sg.speaker_id=?1
                           AND imported.session_id IS NULL
                     )",
                    params![speaker_id],
                    |row| row.get(0),
                )
                .map_err(|error| error.to_string())?;
            target_receives_audio_or_non_import_history |= has_audio || has_non_import_history;
        }
        for group in &request.unassigned_groups {
            let import_id: Option<String> = tx
                .query_row(
                    "SELECT import_id FROM imported_sessions WHERE session_id=?1",
                    params![group.session_id],
                    |row| row.get(0),
                )
                .optional()
                .map_err(|error| error.to_string())?;
            if let Some(import_id) = import_id {
                participating_imports.insert(import_id);
            } else {
                target_receives_audio_or_non_import_history = true;
            }
        }

        let mut target_references = Self::reference_vectors_in_transaction(
            &tx,
            &self.crypto,
            &target_speaker_id,
            crate::embedding::EMBEDDING_VERSION,
        )?;
        let mut activated_voiceprints = 0usize;
        let mut quarantined_voiceprints = 0usize;
        let mut deleted_samples = 0usize;
        for source_id in &source_ids {
            let source_references = Self::reference_vectors_in_transaction(
                &tx,
                &self.crypto,
                source_id,
                crate::embedding::EMBEDDING_VERSION,
            )?;
            let compatible = source_references
                .iter()
                .filter(|(_, vector)| {
                    target_references.is_empty()
                        || target_references.iter().any(|(_, target)| {
                            crate::embedding::cosine_similarity(vector, target)
                                >= SUGGESTION_REFERENCE_COMPATIBILITY_THRESHOLD
                        })
                })
                .cloned()
                .collect::<Vec<_>>();
            tx.execute(
                "UPDATE embeddings
                    SET speaker_id=?1, is_reference=0
                  WHERE speaker_id=?2",
                params![target_speaker_id, source_id],
            )
            .map_err(|error| error.to_string())?;
            for (embedding_id, vector) in compatible {
                tx.execute(
                    "UPDATE embeddings SET is_reference=1 WHERE id=?1",
                    params![embedding_id],
                )
                .map_err(|error| error.to_string())?;
                target_references.push((embedding_id, vector));
                activated_voiceprints += 1;
            }
            quarantined_voiceprints += source_references.len().saturating_sub(
                source_references
                    .iter()
                    .filter(|(embedding_id, _)| {
                        target_references
                            .iter()
                            .any(|(target_id, _)| target_id == embedding_id)
                    })
                    .count(),
            );
            tx.execute(
                "UPDATE segments
                    SET speaker_id=?1, speaker_label=?2
                  WHERE speaker_id=?3",
                params![target_speaker_id, preview.target_label, source_id],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "UPDATE voice_match_decisions
                    SET resulting_speaker_id=?1,
                        resolved_at=COALESCE(resolved_at, ?2),
                        resolution=COALESCE(resolution, 'profile_merged')
                  WHERE resulting_speaker_id=?3",
                params![target_speaker_id, now, source_id],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "UPDATE voice_match_decisions SET best_speaker_id=?1
                  WHERE best_speaker_id=?2",
                params![target_speaker_id, source_id],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "UPDATE voice_match_decisions SET runner_up_speaker_id=?1
                  WHERE runner_up_speaker_id=?2",
                params![target_speaker_id, source_id],
            )
            .map_err(|error| error.to_string())?;
            tx.execute(
                "DELETE FROM import_created_speakers WHERE speaker_id=?1",
                params![source_id],
            )
            .map_err(|error| error.to_string())?;
            deleted_samples += tx
                .execute(
                    "DELETE FROM speaker_samples WHERE speaker_id=?1",
                    params![source_id],
                )
                .map_err(|error| error.to_string())?;
            tx.execute("DELETE FROM speakers WHERE id=?1", params![source_id])
                .map_err(|error| error.to_string())?;
        }

        for group in &request.unassigned_groups {
            let changed = tx
                .execute(
                    "UPDATE segments
                        SET speaker_id=?1, speaker_label=?2
                      WHERE session_id=?3
                        AND speaker_id IS NULL
                        AND (
                             (?4 IS NULL AND speaker_label IS NULL)
                             OR speaker_label=?4
                        )",
                    params![
                        target_speaker_id,
                        preview.target_label,
                        group.session_id,
                        group.speaker_label
                    ],
                )
                .map_err(|error| error.to_string())?;
            if changed == 0 {
                return Err(format!(
                    "The selected {} group in conversation {} changed before consolidation",
                    group.speaker_label.as_deref().unwrap_or("Unknown speaker"),
                    group.session_id
                ));
            }
        }
        tx.execute(
            "UPDATE speakers SET label=?1 WHERE id=?2",
            params![preview.target_label, target_speaker_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE segments SET speaker_label=?1 WHERE speaker_id=?2",
            params![preview.target_label, target_speaker_id],
        )
        .map_err(|error| error.to_string())?;

        deleted_samples += tx
            .execute(
                "DELETE FROM speaker_samples WHERE speaker_id=?1",
                params![target_speaker_id],
            )
            .map_err(|error| error.to_string())?;
        let target_was_import_owned = import_owners
            .get(&target_speaker_id)
            .is_some_and(|owners| !owners.is_empty());
        if target_was_import_owned
            && (target_receives_audio_or_non_import_history
                || any_unowned_source
                || participating_imports.len() > 1)
        {
            tx.execute(
                "DELETE FROM import_created_speakers WHERE speaker_id=?1",
                params![target_speaker_id],
            )
            .map_err(|error| error.to_string())?;
        }

        Self::rebuild_session_transcripts_in_transaction(
            &tx,
            &self.crypto,
            &preview.affected_session_ids,
        )?;
        tx.commit().map_err(|error| error.to_string())?;
        Ok(IdentityConsolidationResult {
            target_speaker_id,
            target_label: preview.target_label,
            merged_profile_count: source_ids.len(),
            assigned_group_count: request.unassigned_groups.len(),
            affected_conversation_count: preview.affected_conversation_count,
            affected_intervention_count: preview.affected_intervention_count,
            activated_voiceprints,
            quarantined_voiceprints,
            deleted_samples,
            backup_path: backup.to_string_lossy().to_string(),
        })
    }

    pub fn insert_voice_match_decision(
        &self,
        decision: &VoiceMatchDecisionSave<'_>,
    ) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let provider_speakers_json =
            serde_json::to_string(decision.provider_speakers).map_err(|error| error.to_string())?;
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO voice_match_decisions(
                    id,
                    session_id,
                    provider_speakers_json,
                    resulting_speaker_id,
                    best_speaker_id,
                    runner_up_speaker_id,
                    best_score,
                    runner_up_score,
                    support_count,
                    selected_duration_ms,
                    selected_window_count,
                    consistency_score,
                    model_version,
                    decision,
                    reason,
                    created_at
                 ) VALUES(
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                    ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16
                 )",
                params![
                    id,
                    decision.session_id,
                    provider_speakers_json,
                    decision.resulting_speaker_id,
                    decision.best_speaker_id,
                    decision.runner_up_speaker_id,
                    decision.best_score.map(f64::from),
                    decision.runner_up_score.map(f64::from),
                    decision.support_count.min(i64::MAX as usize) as i64,
                    decision.selected_duration_ms.min(i64::MAX as u64) as i64,
                    decision.selected_window_count.min(i64::MAX as usize) as i64,
                    decision.consistency_score.map(f64::from),
                    decision.model_version,
                    decision.decision,
                    decision.reason,
                    now.to_rfc3339(),
                ],
            )
            .map_err(|error| error.to_string())?;
        Ok(id)
    }

    pub fn list_voice_match_decisions(
        &self,
        session_id: &str,
    ) -> Result<Vec<VoiceMatchDecision>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT d.id,
                        d.session_id,
                        d.provider_speakers_json,
                        d.resulting_speaker_id,
                        d.best_speaker_id,
                        suggested.label,
                        d.runner_up_speaker_id,
                        runner_up.label,
                        d.best_score,
                        d.runner_up_score,
                        d.support_count,
                        d.selected_duration_ms,
                        d.selected_window_count,
                        d.consistency_score,
                        d.model_version,
                        d.decision,
                        d.reason,
                        d.created_at,
                        d.resolved_at,
                        d.resolution
                   FROM voice_match_decisions d
                   LEFT JOIN speakers suggested ON suggested.id = d.best_speaker_id
                   LEFT JOIN speakers runner_up ON runner_up.id = d.runner_up_speaker_id
                  WHERE d.session_id=?1
                  ORDER BY d.created_at, d.id",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params![session_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<f64>>(8)?,
                    row.get::<_, Option<f64>>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                    row.get::<_, i64>(12)?,
                    row.get::<_, Option<f64>>(13)?,
                    row.get::<_, String>(14)?,
                    row.get::<_, String>(15)?,
                    row.get::<_, String>(16)?,
                    row.get::<_, String>(17)?,
                    row.get::<_, Option<String>>(18)?,
                    row.get::<_, Option<String>>(19)?,
                ))
            })
            .map_err(|error| error.to_string())?;

        let mut decisions = Vec::new();
        for row in rows {
            let (
                id,
                session_id,
                provider_speakers_json,
                resulting_speaker_id,
                best_speaker_id,
                best_speaker_label,
                runner_up_speaker_id,
                runner_up_speaker_label,
                best_score,
                runner_up_score,
                support_count,
                selected_duration_ms,
                selected_window_count,
                consistency_score,
                model_version,
                decision,
                reason,
                created_at,
                resolved_at,
                resolution,
            ) = row.map_err(|error| error.to_string())?;
            decisions.push(VoiceMatchDecision {
                id,
                session_id,
                provider_speakers: serde_json::from_str(&provider_speakers_json)
                    .map_err(|error| error.to_string())?,
                resulting_speaker_id,
                best_speaker_id,
                best_speaker_label,
                runner_up_speaker_id,
                runner_up_speaker_label,
                best_score: best_score.map(|value| value as f32),
                runner_up_score: runner_up_score.map(|value| value as f32),
                support_count: support_count.max(0) as usize,
                selected_duration_ms: selected_duration_ms.max(0) as u64,
                selected_window_count: selected_window_count.max(0) as usize,
                consistency_score: consistency_score.map(|value| value as f32),
                model_version,
                decision,
                reason,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map_err(|error| error.to_string())?
                    .with_timezone(&Utc),
                resolved_at: resolved_at
                    .map(|value| {
                        DateTime::parse_from_rfc3339(&value)
                            .map(|date| date.with_timezone(&Utc))
                            .map_err(|error| error.to_string())
                    })
                    .transpose()?,
                resolution,
            });
        }
        Ok(decisions)
    }

    pub fn session_ids_for_speakers(&self, speaker_ids: &[&str]) -> Result<Vec<String>, String> {
        if speaker_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = std::iter::repeat_n("?", speaker_ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(&format!(
                "SELECT DISTINCT session_id
                   FROM segments
                  WHERE speaker_id IN ({placeholders})
                  ORDER BY session_id"
            ))
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params_from_iter(speaker_ids.iter()), |row| {
                row.get::<_, String>(0)
            })
            .map_err(|error| error.to_string())?;
        let mut sessions = Vec::new();
        for row in rows {
            sessions.push(row.map_err(|error| error.to_string())?);
        }
        Ok(sessions)
    }

    pub fn rename_speaker(
        &self,
        speaker_id: &str,
        new_label: &str,
    ) -> Result<RenameSpeakerResult, String> {
        let label = display_person_name(new_label);
        if label.is_empty() {
            return Err("Speaker name cannot be empty".into());
        }
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        let affected_session_ids =
            Self::session_ids_for_profiles_in_transaction(&tx, &[speaker_id])?;
        let normalized_label = normalized_person_name(&label);
        let conflicting_profile = {
            let mut stmt = tx
                .prepare("SELECT id, label FROM speakers WHERE id<>?1 AND label IS NOT NULL")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![speaker_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|error| error.to_string())?;
            let mut conflicting_profile = None;
            for row in rows {
                let (candidate_id, candidate_label) = row.map_err(|error| error.to_string())?;
                if !candidate_label.trim().is_empty()
                    && !is_provisional_label(&candidate_label)
                    && normalized_person_name(&candidate_label) == normalized_label
                {
                    conflicting_profile = Some((candidate_id, candidate_label));
                    break;
                }
            }
            conflicting_profile
        };
        if let Some((conflicting_speaker_id, conflicting_label)) = conflicting_profile {
            return Ok(RenameSpeakerResult {
                status: "conflict".into(),
                conflicting_speaker_id: Some(conflicting_speaker_id),
                conflicting_label: Some(conflicting_label),
            });
        }
        let changed = tx
            .execute(
                "UPDATE speakers SET label=?1 WHERE id=?2",
                params![&label, speaker_id],
            )
            .map_err(|e| e.to_string())?;
        if changed == 0 {
            return Err("Speaker profile not found".into());
        }
        tx.execute(
            "UPDATE segments SET speaker_label=?1 WHERE speaker_id=?2",
            params![&label, speaker_id],
        )
        .map_err(|e| e.to_string())?;
        // Discard samples after naming/renaming for privacy.
        tx.execute(
            "DELETE FROM speaker_samples WHERE speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|e| e.to_string())?;
        Self::rebuild_session_transcripts_in_transaction(&tx, &self.crypto, &affected_session_ids)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(RenameSpeakerResult {
            status: "renamed".into(),
            conflicting_speaker_id: None,
            conflicting_label: None,
        })
    }

    pub fn delete_speaker(&self, speaker_id: &str) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        let affected_session_ids =
            Self::session_ids_for_profiles_in_transaction(&tx, &[speaker_id])?;
        let profile: Option<(Option<String>, i64)> = tx
            .query_row(
                "SELECT s.label,
                        (SELECT COUNT(DISTINCT sg.session_id)
                           FROM segments sg
                          WHERE sg.speaker_id = s.id)
                   FROM speakers s
                  WHERE s.id=?1",
                params![speaker_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        let Some((label, conversation_count)) = profile else {
            return Err("Speaker profile not found".into());
        };
        let is_named = label
            .as_deref()
            .map(|value| !value.trim().is_empty() && !is_provisional_label(value))
            .unwrap_or(false);
        if is_named && conversation_count > 0 {
            let label = label.as_deref().unwrap_or("This named person");
            return Err(format!(
                "{label} is used in {conversation_count} conversation{}. Reassign or delete those conversations before deleting the named voice profile.",
                if conversation_count == 1 { "" } else { "s" }
            ));
        }
        tx.execute(
            "DELETE FROM embeddings WHERE speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "UPDATE segments SET speaker_id=NULL, speaker_label=NULL WHERE speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "DELETE FROM speaker_samples WHERE speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|e| e.to_string())?;
        let now: DateTime<Utc> = SystemTime::now().into();
        tx.execute(
            "UPDATE voice_match_decisions
                SET resulting_speaker_id=NULL,
                    resolved_at=COALESCE(resolved_at, ?1),
                    resolution=COALESCE(resolution, 'profile_deleted')
              WHERE resulting_speaker_id=?2",
            params![now.to_rfc3339(), speaker_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE voice_match_decisions
                SET best_speaker_id=NULL,
                    resolved_at=COALESCE(resolved_at, ?1),
                    resolution=COALESCE(resolution, 'suggested_profile_deleted')
              WHERE best_speaker_id=?2",
            params![now.to_rfc3339(), speaker_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE voice_match_decisions SET runner_up_speaker_id=NULL
              WHERE runner_up_speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute("DELETE FROM speakers WHERE id=?1", params![speaker_id])
            .map_err(|e| e.to_string())?;
        Self::rebuild_session_transcripts_in_transaction(&tx, &self.crypto, &affected_session_ids)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn merge_speakers(
        &self,
        source_id: &str,
        target_id: &str,
        replace_target_voiceprints: bool,
    ) -> Result<SpeakerMergeResult, String> {
        if source_id == target_id {
            return Err("Source and target speaker profiles must be different".into());
        }
        // Recall's runtime database is file-backed and must be recoverable before
        // a merge. Unit tests also exercise this method with an in-memory
        // database, where there is no file that SQLite can back up.
        let _backup = if self.path.is_some() {
            Some(self.verified_runtime_backup("pre-identity-merge")?)
        } else {
            None
        };
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        let affected_session_ids =
            Self::session_ids_for_profiles_in_transaction(&tx, &[source_id, target_id])?;
        let target_row: Option<Option<String>> = tx
            .query_row(
                "SELECT label FROM speakers WHERE id=?1",
                params![target_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        let Some(target_label) = target_row else {
            return Err("Target speaker profile not found".into());
        };
        let source_exists: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM speakers WHERE id=?1)",
                params![source_id],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        if !source_exists {
            return Err("Source speaker profile not found".into());
        }
        let import_owners_for = |speaker_id: &str| -> Result<HashSet<String>, String> {
            let mut stmt = tx
                .prepare("SELECT import_id FROM import_created_speakers WHERE speaker_id=?1")
                .map_err(|error| error.to_string())?;
            let rows = stmt
                .query_map(params![speaker_id], |row| row.get::<_, String>(0))
                .map_err(|error| error.to_string())?;
            let mut owners = HashSet::new();
            for row in rows {
                owners.insert(row.map_err(|error| error.to_string())?);
            }
            Ok(owners)
        };
        let target_import_owners = import_owners_for(target_id)?;
        let source_import_owners = import_owners_for(source_id)?;
        let participating_import_count = target_import_owners.union(&source_import_owners).count();
        let target_receives_audio_or_non_import_history = [target_id, source_id].iter().try_fold(
            false,
            |found, speaker_id| -> Result<bool, String> {
                let has_audio: bool = tx
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM embeddings WHERE speaker_id=?1)",
                        params![speaker_id],
                        |row| row.get(0),
                    )
                    .map_err(|error| error.to_string())?;
                let has_non_import_history: bool = tx
                    .query_row(
                        "SELECT EXISTS(
                            SELECT 1
                              FROM segments sg
                              LEFT JOIN imported_sessions imported
                                ON imported.session_id=sg.session_id
                             WHERE sg.speaker_id=?1
                               AND imported.session_id IS NULL
                         )",
                        params![speaker_id],
                        |row| row.get(0),
                    )
                    .map_err(|error| error.to_string())?;
                Ok(found || has_audio || has_non_import_history)
            },
        )?;
        let target_references = Self::reference_vectors_in_transaction(
            &tx,
            &self.crypto,
            target_id,
            crate::embedding::EMBEDDING_VERSION,
        )?;
        let source_references = Self::reference_vectors_in_transaction(
            &tx,
            &self.crypto,
            source_id,
            crate::embedding::EMBEDDING_VERSION,
        )?;
        let compatible_ids = if replace_target_voiceprints {
            source_references
                .iter()
                .map(|(embedding_id, _)| embedding_id.clone())
                .collect::<HashSet<_>>()
        } else {
            source_references
                .iter()
                .filter(|(_, vector)| {
                    target_references.is_empty()
                        || target_references.iter().any(|(_, target)| {
                            crate::embedding::cosine_similarity(vector, target)
                                >= SUGGESTION_REFERENCE_COMPATIBILITY_THRESHOLD
                        })
                })
                .map(|(embedding_id, _)| embedding_id.clone())
                .collect::<HashSet<_>>()
        };
        if replace_target_voiceprints {
            tx.execute(
                "DELETE FROM embeddings WHERE speaker_id=?1",
                params![target_id],
            )
            .map_err(|e| e.to_string())?;
        }
        if replace_target_voiceprints {
            tx.execute(
                "UPDATE embeddings SET speaker_id=?1 WHERE speaker_id=?2",
                params![target_id, source_id],
            )
            .map_err(|e| e.to_string())?;
        } else {
            tx.execute(
                "UPDATE embeddings SET speaker_id=?1, is_reference=0 WHERE speaker_id=?2",
                params![target_id, source_id],
            )
            .map_err(|e| e.to_string())?;
            for embedding_id in &compatible_ids {
                tx.execute(
                    "UPDATE embeddings SET is_reference=1 WHERE id=?1",
                    params![embedding_id],
                )
                .map_err(|error| error.to_string())?;
            }
        }
        tx.execute(
            "UPDATE segments SET speaker_id=?1, speaker_label=?2 WHERE speaker_id=?3",
            params![target_id, target_label, source_id],
        )
        .map_err(|e| e.to_string())?;

        let target_is_named = target_label
            .as_deref()
            .map(|label| !is_provisional_label(label))
            .unwrap_or(false);
        if target_is_named {
            tx.execute(
                "DELETE FROM speaker_samples WHERE speaker_id IN (?1, ?2)",
                params![source_id, target_id],
            )
            .map_err(|e| e.to_string())?;
        } else {
            tx.execute(
                "UPDATE speaker_samples SET speaker_id=?1 WHERE speaker_id=?2",
                params![target_id, source_id],
            )
            .map_err(|e| e.to_string())?;
        }
        let now: DateTime<Utc> = SystemTime::now().into();
        tx.execute(
            "UPDATE voice_match_decisions
                SET resulting_speaker_id=?1,
                    resolved_at=COALESCE(resolved_at, ?2),
                    resolution=COALESCE(resolution, 'profile_merged')
              WHERE resulting_speaker_id=?3",
            params![target_id, now.to_rfc3339(), source_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE voice_match_decisions SET best_speaker_id=?1
              WHERE best_speaker_id=?2",
            params![target_id, source_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE voice_match_decisions SET runner_up_speaker_id=?1
              WHERE runner_up_speaker_id=?2",
            params![target_id, source_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM import_created_speakers WHERE speaker_id=?1",
            params![source_id],
        )
        .map_err(|error| error.to_string())?;
        if !target_import_owners.is_empty()
            && (target_receives_audio_or_non_import_history
                || source_import_owners.is_empty()
                || participating_import_count > 1)
        {
            tx.execute(
                "DELETE FROM import_created_speakers WHERE speaker_id=?1",
                params![target_id],
            )
            .map_err(|error| error.to_string())?;
        }
        tx.execute("DELETE FROM speakers WHERE id=?1", params![source_id])
            .map_err(|e| e.to_string())?;
        Self::rebuild_session_transcripts_in_transaction(&tx, &self.crypto, &affected_session_ids)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(SpeakerMergeResult {
            target_speaker_id: target_id.to_string(),
            target_label: target_label.unwrap_or_else(|| "Unnamed voice".into()),
            activated_voiceprints: compatible_ids.len(),
            quarantined_voiceprints: source_references.len().saturating_sub(compatible_ids.len()),
            replaced_target_voiceprints: replace_target_voiceprints,
        })
    }

    fn reference_vectors_in_transaction(
        tx: &rusqlite::Transaction<'_>,
        crypto: &Crypto,
        speaker_id: &str,
        model_version: &str,
    ) -> Result<Vec<(String, Vec<f32>)>, String> {
        let mut stmt = tx
            .prepare(
                "SELECT id, vector_nonce, vector_ct
                   FROM embeddings
                  WHERE speaker_id=?1 AND model_version=?2 AND is_reference=1",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map(params![speaker_id, model_version], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|error| error.to_string())?;
        let mut vectors = Vec::new();
        for row in rows {
            let (id, nonce, ciphertext) = row.map_err(|error| error.to_string())?;
            let bytes = crypto.decrypt(&nonce, &ciphertext)?;
            if bytes.len() % std::mem::size_of::<f32>() != 0 {
                continue;
            }
            vectors.push((id, bytemuck::cast_slice(&bytes).to_vec()));
        }
        Ok(vectors)
    }

    pub fn accept_voice_match_suggestion(
        &self,
        source_id: &str,
        target_id: &str,
    ) -> Result<SuggestionAcceptance, String> {
        if source_id == target_id {
            return Err("The suggested and provisional profiles must be different".into());
        }
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        let affected_session_ids =
            Self::session_ids_for_profiles_in_transaction(&tx, &[source_id, target_id])?;
        let target_label: Option<String> = tx
            .query_row(
                "SELECT label FROM speakers WHERE id=?1",
                params![target_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| error.to_string())?
            .flatten();
        let Some(target_label) = target_label else {
            return Err("The suggested person no longer exists".into());
        };
        if target_label.trim().is_empty() || is_provisional_label(&target_label) {
            return Err("Suggestions can only be accepted into a named person".into());
        }
        let source_exists: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM speakers WHERE id=?1)",
                params![source_id],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        if !source_exists {
            return Err("The provisional voice profile no longer exists".into());
        }
        let suggestion_exists: bool = tx
            .query_row(
                "SELECT EXISTS(
                    SELECT 1
                      FROM voice_match_decisions
                     WHERE resulting_speaker_id=?1
                       AND best_speaker_id=?2
                       AND decision='suggested'
                       AND resolved_at IS NULL
                 )",
                params![source_id, target_id],
                |row| row.get(0),
            )
            .map_err(|error| error.to_string())?;
        if !suggestion_exists {
            return Err("That likely-person suggestion is no longer current".into());
        }

        let target_references = Self::reference_vectors_in_transaction(
            &tx,
            &self.crypto,
            target_id,
            crate::embedding::EMBEDDING_VERSION,
        )?;
        let source_references = Self::reference_vectors_in_transaction(
            &tx,
            &self.crypto,
            source_id,
            crate::embedding::EMBEDDING_VERSION,
        )?;
        let mut compatible_ids = HashSet::new();
        for (embedding_id, vector) in &source_references {
            let compatible = target_references.is_empty()
                || target_references.iter().any(|(_, target)| {
                    crate::embedding::cosine_similarity(vector, target)
                        >= SUGGESTION_REFERENCE_COMPATIBILITY_THRESHOLD
                });
            if compatible {
                compatible_ids.insert(embedding_id.clone());
            }
        }

        tx.execute(
            "UPDATE embeddings SET speaker_id=?1, is_reference=0 WHERE speaker_id=?2",
            params![target_id, source_id],
        )
        .map_err(|error| error.to_string())?;
        for embedding_id in &compatible_ids {
            tx.execute(
                "UPDATE embeddings SET is_reference=1 WHERE id=?1",
                params![embedding_id],
            )
            .map_err(|error| error.to_string())?;
        }
        tx.execute(
            "UPDATE segments SET speaker_id=?1, speaker_label=?2 WHERE speaker_id=?3",
            params![target_id, &target_label, source_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM speaker_samples WHERE speaker_id IN (?1, ?2)",
            params![source_id, target_id],
        )
        .map_err(|error| error.to_string())?;
        let now: DateTime<Utc> = SystemTime::now().into();
        tx.execute(
            "UPDATE voice_match_decisions
                SET resulting_speaker_id=?1,
                    resolved_at=COALESCE(resolved_at, ?2),
                    resolution=COALESCE(resolution, 'accepted_suggestion')
              WHERE resulting_speaker_id=?3",
            params![target_id, now.to_rfc3339(), source_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE voice_match_decisions SET best_speaker_id=?1
              WHERE best_speaker_id=?2",
            params![target_id, source_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute(
            "UPDATE voice_match_decisions SET runner_up_speaker_id=?1
              WHERE runner_up_speaker_id=?2",
            params![target_id, source_id],
        )
        .map_err(|error| error.to_string())?;
        tx.execute("DELETE FROM speakers WHERE id=?1", params![source_id])
            .map_err(|error| error.to_string())?;
        Self::rebuild_session_transcripts_in_transaction(&tx, &self.crypto, &affected_session_ids)?;
        tx.commit().map_err(|error| error.to_string())?;

        Ok(SuggestionAcceptance {
            target_speaker_id: target_id.to_string(),
            target_label,
            activated_voiceprints: compatible_ids.len(),
            quarantined_voiceprints: source_references.len().saturating_sub(compatible_ids.len()),
        })
    }

    pub fn insert_embedding(
        &self,
        speaker_id: &str,
        session_id: &str,
        vector: &[f32],
        model_version: &str,
    ) -> Result<String, String> {
        self.insert_embedding_with_reference(speaker_id, session_id, vector, model_version, true)
    }

    pub fn insert_embedding_with_reference(
        &self,
        speaker_id: &str,
        session_id: &str,
        vector: &[f32],
        model_version: &str,
        is_reference: bool,
    ) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let bytes: &[u8] = bytemuck::cast_slice(vector);
        let (nonce, ct) = self.crypto.encrypt(bytes);
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO embeddings(id, speaker_id, vector_nonce, vector_ct, source_session_id, created_at, model_version, is_reference) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    id,
                    speaker_id,
                    nonce,
                    ct,
                    session_id,
                    now.to_rfc3339(),
                    model_version,
                    i64::from(is_reference),
                ],
            )
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn insert_sample(
        &self,
        speaker_id: &str,
        sample_b64: &str,
        sample_rate: u32,
    ) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO speaker_samples(id, speaker_id, sample_b64, sample_rate, created_at) VALUES(?1, ?2, ?3, ?4, ?5)",
                params![id, speaker_id, sample_b64, sample_rate as i64, now.to_rfc3339()],
            )
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn list_samples(&self, speaker_id: &str) -> Result<Vec<SpeakerSample>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, sample_b64, sample_rate, created_at FROM speaker_samples WHERE speaker_id=?1 ORDER BY created_at DESC")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![speaker_id], |row| {
                let id: String = row.get(0)?;
                let sample_b64: String = row.get(1)?;
                let sample_rate: i64 = row.get(2)?;
                let created_at: String = row.get(3)?;
                Ok((id, sample_b64, sample_rate, created_at))
            })
            .map_err(|e| e.to_string())?;

        let mut samples = Vec::new();
        for row in rows {
            let (id, sample_b64, sample_rate, created_at) = row.map_err(|e| e.to_string())?;
            let created_at = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            samples.push(SpeakerSample {
                id,
                speaker_id: speaker_id.to_string(),
                sample_b64,
                sample_rate: sample_rate as u32,
                created_at,
            });
        }
        Ok(samples)
    }

    pub fn list_embeddings(&self, model_version: &str) -> Result<Vec<StoredEmbedding>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT e.id, e.speaker_id, s.label, e.vector_nonce, e.vector_ct, e.source_session_id, e.created_at, e.model_version
                 FROM embeddings e
                 LEFT JOIN speakers s ON e.speaker_id = s.id
                 WHERE e.model_version = ?1 AND e.is_reference = 1",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![model_version], |row| {
                let id: String = row.get(0)?;
                let speaker_id: String = row.get(1)?;
                let speaker_label: Option<String> = row.get(2)?;
                let nonce: String = row.get(3)?;
                let ct: String = row.get(4)?;
                let source_session_id: String = row.get(5)?;
                let created_at: String = row.get(6)?;
                let model_version: String = row.get(7)?;
                Ok((
                    id,
                    speaker_id,
                    speaker_label,
                    nonce,
                    ct,
                    source_session_id,
                    created_at,
                    model_version,
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut embeddings = Vec::new();
        for row in rows {
            let (
                id,
                speaker_id,
                speaker_label,
                nonce,
                ct,
                source_session_id,
                created_at,
                model_version,
            ) = row.map_err(|e| e.to_string())?;
            let bytes = self.crypto.decrypt(&nonce, &ct)?;
            if bytes.len() % std::mem::size_of::<f32>() != 0 {
                continue;
            }
            let floats: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
            let created_at = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            embeddings.push(StoredEmbedding {
                id,
                speaker_id,
                speaker_label,
                vector: floats,
                source_session_id,
                created_at,
                model_version,
            });
        }
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory_db() -> Db {
        Db::open(":memory:", Crypto::new(None, None)).expect("open in-memory database")
    }

    fn test_jamie_archive() -> JamieArchive {
        use crate::jamie_import::{JamieExportMetadata, JamieMeeting, JamieTranscriptSegment};
        let started_at = DateTime::parse_from_rfc3339("2026-07-16T14:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        JamieArchive {
            metadata: JamieExportMetadata {
                user: Some("Test User".into()),
                export_date: Some(
                    DateTime::parse_from_rfc3339("2026-07-23T09:01:12Z")
                        .unwrap()
                        .with_timezone(&Utc),
                ),
                declared_total_meetings: Some(2),
                includes: vec!["Summaries".into(), "Transcripts".into(), "Tasks".into()],
                source_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .into(),
                source_size_bytes: 1_024,
            },
            meetings: vec![
                JamieMeeting {
                    source_fingerprint: "meeting-one".into(),
                    title: "Imported one".into(),
                    started_at: Some(started_at),
                    ended_at: Some(started_at + chrono::Duration::seconds(5)),
                    duration_ms: 5_000,
                    speaker_map: Vec::new(),
                    executive_summary: "Executive one".into(),
                    full_summary: "Full one".into(),
                    tasks: "[ ] Follow up".into(),
                    segments: vec![
                        JamieTranscriptSegment {
                            speaker_label: "Mv".into(),
                            start_ms: 0,
                            end_ms: 2_000,
                            text: "Hello".into(),
                        },
                        JamieTranscriptSegment {
                            speaker_label: "Anna".into(),
                            start_ms: 2_000,
                            end_ms: 4_000,
                            text: "Hi".into(),
                        },
                    ],
                    warnings: Vec::new(),
                },
                JamieMeeting {
                    source_fingerprint: "meeting-two".into(),
                    title: "Imported two".into(),
                    started_at: Some(started_at + chrono::Duration::days(1)),
                    ended_at: Some(
                        started_at + chrono::Duration::days(1) + chrono::Duration::seconds(3),
                    ),
                    duration_ms: 3_000,
                    speaker_map: Vec::new(),
                    executive_summary: String::new(),
                    full_summary: "Full two".into(),
                    tasks: String::new(),
                    segments: vec![
                        JamieTranscriptSegment {
                            speaker_label: "Mv".into(),
                            start_ms: 0,
                            end_ms: 1_000,
                            text: "Again".into(),
                        },
                        JamieTranscriptSegment {
                            speaker_label: "SPEAKER_00".into(),
                            start_ms: 1_000,
                            end_ms: 3_000,
                            text: "Meeting-local voice".into(),
                        },
                    ],
                    warnings: Vec::new(),
                },
            ],
            warnings: Vec::new(),
        }
    }

    fn test_jamie_draft(michael_id: &str) -> JamieImportDraft {
        JamieImportDraft {
            id: "aaaaaaaaaaaaaaaa".into(),
            source_path: "/private/test-export.txt".into(),
            source_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .into(),
            importer_version: JAMIE_IMPORTER_VERSION.into(),
            identity_decisions: vec![
                crate::jamie_import::JamieIdentityDecision {
                    alias: "Mv".into(),
                    action: "map_existing".into(),
                    target_speaker_id: Some(michael_id.into()),
                    display_name: Some("Michael Vartanyan".into()),
                },
                crate::jamie_import::JamieIdentityDecision {
                    alias: "Anna".into(),
                    action: "create_named".into(),
                    target_speaker_id: None,
                    display_name: Some("Anna Smith".into()),
                },
                crate::jamie_import::JamieIdentityDecision {
                    alias: "SPEAKER_00".into(),
                    action: "unresolved".into(),
                    target_speaker_id: None,
                    display_name: None,
                },
            ],
            excluded_meetings: Vec::new(),
            updated_at: Utc::now(),
        }
    }

    fn test_recap_payload() -> RecapPayload {
        let localized = crate::recap::LocalizedText {
            original: "Summary".into(),
            translated: "Summary".into(),
        };
        RecapPayload {
            target_language: "en".into(),
            meeting_title: "Weekly planning".into(),
            dominant_language: "en".into(),
            executive_summary: localized.clone(),
            full_summary: vec![crate::recap::SummarySection {
                heading: localized.clone(),
                body: localized,
                evidence_segment_ids: vec!["segment-1".into()],
            }],
            commitments: Vec::new(),
            actions_already_taken: Vec::new(),
            agenda_present: false,
            agenda_coverage: Vec::new(),
            translations: Vec::new(),
        }
    }

    #[test]
    fn voice_labels_are_monotonic() {
        let db = memory_db();
        db.insert_speaker(Some("Alice")).unwrap();
        db.insert_speaker(Some("VOICE9")).unwrap();
        assert_eq!(db.next_voice_label().unwrap(), "VOICE10");
    }

    #[test]
    fn archive_summaries_defer_transcript_decryption_until_one_session_is_opened() {
        let db = Db::open(":memory:", Crypto::new(Some("local test password"), None)).unwrap();
        let first = db
            .insert_session("Planning", "Alice: private roadmap phrase", 5_000)
            .unwrap();
        let second = db
            .insert_session("Review", "Bob: ordinary update", 7_000)
            .unwrap();

        let summaries = db.list_session_summaries().unwrap();
        assert_eq!(summaries.len(), 2);
        assert!(summaries.iter().any(|session| session.id == first));
        assert!(summaries.iter().any(|session| session.id == second));

        let selected = db.get_session(&first).unwrap().unwrap();
        assert_eq!(selected.transcript, "Alice: private roadmap phrase");
        assert!(db.get_session("missing").unwrap().is_none());
        assert_eq!(
            db.search_session_ids("private roadmap").unwrap(),
            vec![first]
        );
        assert_eq!(db.search_session_ids("Review").unwrap(), vec![second]);
    }

    #[test]
    fn large_archive_summary_payload_is_independent_of_transcript_size() {
        let db = memory_db();
        let transcript = format!(
            "archive-only-secret {}",
            "substantial transcript content ".repeat(1_200)
        );
        for index in 0..545 {
            db.insert_session(&format!("Meeting {index}"), &transcript, 60_000)
                .unwrap();
        }

        let summaries = db.list_session_summaries().unwrap();
        let serialized = serde_json::to_string(&summaries).unwrap();

        assert_eq!(summaries.len(), 545);
        assert!(!serialized.contains("archive-only-secret"));
        assert!(
            serialized.len() < 160_000,
            "summary payload was {} bytes",
            serialized.len()
        );
    }

    #[test]
    fn jamie_import_is_transactional_idempotent_and_rollback_preserves_mapped_people() {
        let path = std::env::temp_dir().join(format!(
            "recall-jamie-import-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let michael = db.insert_speaker(Some("Michael Vartanyan")).unwrap();
        let archive = test_jamie_archive();
        let draft = test_jamie_draft(&michael);

        let imported = db.import_jamie_archive(&archive, &draft).unwrap();

        assert_eq!(imported.imported_meetings, 2);
        assert_eq!(imported.already_imported_meetings, 0);
        assert_eq!(imported.imported_interventions, 4);
        assert_eq!(imported.created_people, 1);
        let import_id = imported.import_id.clone().unwrap();
        let import_backup = PathBuf::from(imported.backup_path.clone().unwrap());
        assert!(import_backup.is_file());
        assert_eq!(db.list_sessions().unwrap().len(), 2);
        let people = db.jamie_known_people().unwrap();
        assert_eq!(people.len(), 2);
        assert!(people.iter().any(|person| person.id == michael));
        let anna = people
            .iter()
            .find(|person| person.label == "Anna Smith")
            .unwrap();
        assert!(db
            .list_embeddings(crate::embedding::EMBEDDING_VERSION)
            .unwrap()
            .is_empty());
        let first = db
            .list_sessions()
            .unwrap()
            .into_iter()
            .find(|session| session.title == "Imported one")
            .unwrap();
        assert!(first.transcript.contains("Michael Vartanyan: Hello"));
        assert!(first.transcript.contains("Anna Smith: Hi"));
        let first_segments = db.list_segments(&first.id).unwrap();
        assert_eq!(
            first_segments[0].speaker_id.as_deref(),
            Some(michael.as_str())
        );
        assert_eq!(
            first_segments[1].speaker_id.as_deref(),
            Some(anna.id.as_str())
        );
        let artifact = db
            .load_imported_session_artifact(&first.id)
            .unwrap()
            .unwrap();
        assert_eq!(artifact.source_provider, "Jamie");
        assert_eq!(artifact.executive_summary, "Executive one");
        assert_eq!(artifact.tasks, "[ ] Follow up");

        let repeated = db.import_jamie_archive(&archive, &draft).unwrap();
        assert_eq!(repeated.imported_meetings, 0);
        assert_eq!(repeated.already_imported_meetings, 2);
        assert!(repeated.import_id.is_none());
        assert_eq!(db.list_sessions().unwrap().len(), 2);

        let rolled_back = db.rollback_import(&import_id).unwrap();

        assert_eq!(rolled_back.removed_meetings, 2);
        assert_eq!(rolled_back.removed_people, 1);
        assert_eq!(rolled_back.preserved_people, 0);
        let rollback_backup = PathBuf::from(&rolled_back.backup_path);
        assert!(rollback_backup.is_file());
        assert!(db.list_sessions().unwrap().is_empty());
        let remaining = db.jamie_known_people().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, michael);
        let batch = db.list_import_batches().unwrap().remove(0);
        assert_eq!(batch.status, "rolled_back");
        assert_eq!(batch.meeting_count, 2);
        drop(db);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(import_backup).unwrap();
        std::fs::remove_file(rollback_backup).unwrap();
    }

    #[test]
    fn deleting_an_imported_meeting_keeps_an_idempotency_tombstone() {
        let path = std::env::temp_dir().join(format!(
            "recall-jamie-delete-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let michael = db.insert_speaker(Some("Michael Vartanyan")).unwrap();
        let archive = test_jamie_archive();
        let draft = test_jamie_draft(&michael);
        let imported = db.import_jamie_archive(&archive, &draft).unwrap();
        let import_backup = PathBuf::from(imported.backup_path.unwrap());
        let import_id = imported.import_id.unwrap();
        let deleted = db
            .list_sessions()
            .unwrap()
            .into_iter()
            .find(|session| session.title == "Imported one")
            .unwrap();

        db.delete_session(&deleted.id).unwrap();

        assert!(db
            .load_imported_session_artifact(&deleted.id)
            .unwrap()
            .is_none());
        assert_eq!(db.imported_meeting_fingerprints("Jamie").unwrap().len(), 2);
        let repeated = db.import_jamie_archive(&archive, &draft).unwrap();
        assert_eq!(repeated.imported_meetings, 0);
        assert_eq!(repeated.already_imported_meetings, 2);
        assert_eq!(db.list_sessions().unwrap().len(), 1);

        let rolled_back = db.rollback_import(&import_id).unwrap();
        let rollback_backup = PathBuf::from(rolled_back.backup_path);
        assert_eq!(rolled_back.removed_meetings, 2);
        assert!(db.list_sessions().unwrap().is_empty());
        assert!(db
            .imported_meeting_fingerprints("Jamie")
            .unwrap()
            .is_empty());
        drop(db);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(import_backup).unwrap();
        std::fs::remove_file(rollback_backup).unwrap();
    }

    #[test]
    fn invalid_jamie_identity_mapping_writes_nothing() {
        let path = std::env::temp_dir().join(format!(
            "recall-jamie-invalid-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let michael = db.insert_speaker(Some("Michael Vartanyan")).unwrap();
        let archive = test_jamie_archive();
        let mut draft = test_jamie_draft(&michael);
        draft
            .identity_decisions
            .iter_mut()
            .find(|decision| decision.alias == "Mv")
            .unwrap()
            .target_speaker_id = Some("missing-person".into());

        let error = db.import_jamie_archive(&archive, &draft).unwrap_err();

        assert!(error.contains("review is incomplete"));
        assert!(db.list_sessions().unwrap().is_empty());
        assert!(db.list_import_batches().unwrap().is_empty());
        assert_eq!(db.jamie_known_people().unwrap().len(), 1);
        drop(db);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn unknown_interventions_can_be_grouped_into_one_reviewable_voice() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 2_000).unwrap();
        db.insert_speaker(Some("VOICE9")).unwrap();
        db.insert_segment(&session, 0, 1_000, None, Some("Unknown speaker"), "Hello")
            .unwrap();
        db.insert_segment(
            &session,
            1_000,
            2_000,
            None,
            Some("Unknown speaker"),
            "Again",
        )
        .unwrap();

        let (speaker_id, label, changed) = db
            .create_speaker_for_unattributed_segments(&session)
            .unwrap();

        assert_eq!(label, "VOICE10");
        assert_eq!(changed, 2);
        let segments = db.list_segments(&session).unwrap();
        assert!(segments
            .iter()
            .all(|segment| segment.speaker_id.as_deref() == Some(speaker_id.as_str())));
        assert!(segments
            .iter()
            .all(|segment| segment.speaker_label.as_deref() == Some("VOICE10")));
        assert!(db
            .create_speaker_for_unattributed_segments(&session)
            .is_err());
        assert_eq!(db.list_speakers().unwrap().len(), 2);
    }

    #[test]
    fn speaker_stats_report_when_a_voice_was_last_heard() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("VOICE1")).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&speaker), Some("VOICE1"), "Hello")
            .unwrap();

        let stats = db.list_speakers_with_stats().unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].conversation_count, 1);
        assert!(stats[0].last_seen_at.is_some());
    }

    #[test]
    fn failed_processing_keeps_the_draft_and_removes_partial_voice_artifacts() {
        let db = memory_db();
        let audio_path =
            std::env::temp_dir().join(format!("recall-processing-audio-{}.wav", Uuid::new_v4()));
        std::fs::write(&audio_path, b"retained audio").unwrap();
        let session_id = Uuid::new_v4().to_string();
        db.create_processing_session(
            &session_id,
            "run-1",
            "Draft meeting",
            "Speaker 1: live caption draft",
            42_000,
            &audio_path.to_string_lossy(),
        )
        .unwrap();
        let speaker = db.insert_speaker(Some("VOICE1")).unwrap();
        db.insert_embedding(
            &speaker,
            &session_id,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_sample(&speaker, "dGVzdA==", 16_000).unwrap();
        db.insert_segment(
            &session_id,
            0,
            1_000,
            Some(&speaker),
            Some("VOICE1"),
            "partial final segment",
        )
        .unwrap();

        db.fail_processing_session(&session_id, "upload timed out")
            .unwrap();

        let session = db.list_sessions().unwrap().remove(0);
        assert_eq!(session.transcript, "Speaker 1: live caption draft");
        assert_eq!(session.processing_status.as_deref(), Some("failed"));
        assert_eq!(
            session.processing_error.as_deref(),
            Some("upload timed out")
        );
        assert_eq!(session.processing_run_id.as_deref(), Some("run-1"));
        assert!(session.recoverable_audio);
        assert!(db.list_segments(&session_id).unwrap().is_empty());
        assert!(db.list_speakers().unwrap().is_empty());
        assert!(db
            .list_embeddings(crate::embedding::EMBEDDING_VERSION)
            .unwrap()
            .is_empty());

        db.restart_processing_session(&session_id, "run-2").unwrap();
        let restarted = db.list_sessions().unwrap().remove(0);
        assert_eq!(restarted.processing_status.as_deref(), Some("processing"));
        assert_eq!(restarted.processing_run_id.as_deref(), Some("run-2"));
        assert!(restarted.processing_error.is_none());

        db.finalize_processing_session(
            &session_id,
            "Final meeting",
            "Alice: final transcript",
            42_000,
        )
        .unwrap();
        assert_eq!(
            db.list_sessions().unwrap()[0].processing_status.as_deref(),
            Some("finalized")
        );
        db.complete_processing_session(&session_id).unwrap();
        let complete = db.list_sessions().unwrap().remove(0);
        assert_eq!(complete.title, "Final meeting");
        assert_eq!(complete.transcript, "Alice: final transcript");
        assert!(complete.processing_status.is_none());
        std::fs::remove_file(audio_path).unwrap();
    }

    #[test]
    fn reopening_marks_an_interrupted_processing_job_as_retryable() {
        let path = std::env::temp_dir().join(format!(
            "recall-processing-restart-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let audio_path = std::env::temp_dir().join(format!(
            "recall-processing-restart-audio-{}.wav",
            Uuid::new_v4()
        ));
        std::fs::write(&audio_path, b"retained audio").unwrap();
        let session_id = Uuid::new_v4().to_string();
        {
            let db = Db::open(&path, Crypto::new(None, None)).unwrap();
            db.create_processing_session(
                &session_id,
                "run-before-quit",
                "Draft",
                "Live-caption draft",
                12_000,
                &audio_path.to_string_lossy(),
            )
            .unwrap();
        }

        let reopened = Db::open(&path, Crypto::new(None, None)).unwrap();
        let session = reopened.list_sessions().unwrap().remove(0);
        assert_eq!(session.processing_status.as_deref(), Some("failed"));
        assert!(session
            .processing_error
            .as_deref()
            .unwrap()
            .contains("interrupted"));
        assert_eq!(session.transcript, "Live-caption draft");
        assert!(session.recoverable_audio);
        drop(reopened);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(audio_path).unwrap();
    }

    #[test]
    fn processing_job_migration_backs_up_and_preserves_existing_conversations() {
        let path = std::env::temp_dir().join(format!(
            "recall-processing-migration-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let backup = Db::processing_migration_backup_path(&path);
        {
            let db = Db::open(&path, Crypto::new(None, None)).unwrap();
            db.insert_session("Existing meeting", "Existing transcript", 9_000)
                .unwrap();
        }
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute("DROP TABLE processing_jobs", []).unwrap();
        }

        let migrated = Db::open(&path, Crypto::new(None, None)).unwrap();

        assert!(backup.is_file());
        let sessions = migrated.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "Existing meeting");
        assert_eq!(sessions[0].transcript, "Existing transcript");
        assert!(sessions[0].processing_status.is_none());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                std::fs::metadata(&backup).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        drop(migrated);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(backup).unwrap();
    }

    #[test]
    fn archive_import_migration_backs_up_and_preserves_existing_conversations() {
        let path = std::env::temp_dir().join(format!(
            "recall-import-migration-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let backup = Db::import_migration_backup_path(&path);
        {
            let db = Db::open(&path, Crypto::new(None, None)).unwrap();
            db.insert_session("Existing meeting", "Existing transcript", 9_000)
                .unwrap();
        }
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute("DROP TABLE import_created_speakers", [])
                .unwrap();
            conn.execute("DROP TABLE session_import_artifacts", [])
                .unwrap();
            conn.execute("DROP TABLE imported_sessions", []).unwrap();
            conn.execute("DROP TABLE import_batches", []).unwrap();
        }

        let migrated = Db::open(&path, Crypto::new(None, None)).unwrap();

        assert!(backup.is_file());
        let sessions = migrated.list_sessions().unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "Existing meeting");
        assert_eq!(sessions[0].transcript, "Existing transcript");
        assert!(migrated.list_import_batches().unwrap().is_empty());
        let backup_conn = Connection::open(&backup).unwrap();
        let integrity: String = backup_conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .unwrap();
        assert_eq!(integrity, "ok");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&backup).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        drop(backup_conn);
        drop(migrated);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(backup).unwrap();
    }

    #[test]
    fn voice_match_migration_preserves_legacy_suggestion_targets() {
        let path = std::env::temp_dir().join(format!(
            "recall-voice-match-migration-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let backup = Db::voice_match_migration_backup_path(&path);
        let recap_backup = Db::recap_migration_backup_path(&path);
        let processing_backup = Db::processing_migration_backup_path(&path);
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT,
                    duration_ms INTEGER DEFAULT 0,
                    transcript_nonce TEXT,
                    transcript_ct TEXT NOT NULL
                 );
                 CREATE TABLE voice_match_decisions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    provider_speakers_json TEXT NOT NULL,
                    resulting_speaker_id TEXT,
                    suggested_speaker_id TEXT,
                    runner_up_speaker_id TEXT,
                    best_score REAL,
                    runner_up_score REAL,
                    support_count INTEGER NOT NULL DEFAULT 0,
                    selected_duration_ms INTEGER NOT NULL DEFAULT 0,
                    selected_window_count INTEGER NOT NULL DEFAULT 0,
                    consistency_score REAL,
                    model_version TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution TEXT
                 );
                 INSERT INTO voice_match_decisions(
                    id, session_id, provider_speakers_json,
                    resulting_speaker_id, suggested_speaker_id,
                    support_count, selected_duration_ms, selected_window_count,
                    model_version, decision, reason, created_at
                 ) VALUES(
                    'decision-1', 'session-1', '[\"speaker_1\"]',
                    'voice-1', 'person-1',
                    1, 4000, 1,
                    'model-v1', 'suggested', 'legacy suggestion',
                    '2026-07-23T00:00:00Z'
                 );",
            )
            .unwrap();
        }

        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        assert!(backup.is_file());
        let conn = db.conn.lock().unwrap();
        let migrated_target: Option<String> = conn
            .query_row(
                "SELECT best_speaker_id
                   FROM voice_match_decisions
                  WHERE id='decision-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(migrated_target.as_deref(), Some("person-1"));
        drop(conn);
        drop(db);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&backup).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(backup).unwrap();
        std::fs::remove_file(recap_backup).unwrap();
        std::fs::remove_file(processing_backup).unwrap();
    }

    #[test]
    fn naming_a_voice_updates_segments_and_discards_its_sample() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("VOICE1")).unwrap();
        db.insert_sample(&speaker, "dGVzdA==", 16_000).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&speaker), Some("VOICE1"), "Hello")
            .unwrap();

        assert_eq!(
            db.session_ids_for_speakers(&[speaker.as_str()]).unwrap(),
            vec![session.clone()]
        );
        db.rename_speaker(&speaker, "Alice").unwrap();

        let segments = db.list_segments(&session).unwrap();
        assert_eq!(segments[0].speaker_label.as_deref(), Some("Alice"));
        assert!(db.list_samples(&speaker).unwrap().is_empty());
    }

    #[test]
    fn normalized_duplicate_name_routes_to_the_existing_profile() {
        let db = memory_db();
        let existing = db.insert_speaker(Some("José Smith")).unwrap();
        let provisional = db.insert_speaker(Some("VOICE2")).unwrap();

        let result = db
            .rename_speaker(&provisional, "  JOSE\u{301}   SMITH  ")
            .unwrap();

        assert_eq!(result.status, "conflict");
        assert_eq!(
            result.conflicting_speaker_id.as_deref(),
            Some(existing.as_str())
        );
        assert_eq!(
            db.list_speakers()
                .unwrap()
                .into_iter()
                .find(|speaker| speaker.id == provisional)
                .unwrap()
                .label
                .as_deref(),
            Some("VOICE2")
        );
    }

    #[test]
    fn direct_named_profile_creation_rejects_normalized_duplicates() {
        let db = memory_db();
        db.insert_speaker(Some("Michael Vartanyan")).unwrap();

        let error = db
            .insert_speaker(Some(" michael   vartanyan "))
            .unwrap_err();

        assert!(error.contains("already exists"));
        assert_eq!(db.list_speakers().unwrap().len(), 1);
    }

    #[test]
    fn legacy_duplicate_names_are_exposed_as_conflicts() {
        let db = memory_db();
        let now: DateTime<Utc> = SystemTime::now().into();
        {
            let conn = db.conn.lock().unwrap();
            conn.execute(
                "INSERT INTO speakers(id, label, created_at) VALUES('one', 'Alice', ?1)",
                params![now.to_rfc3339()],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO speakers(id, label, created_at) VALUES('two', ' alice ', ?1)",
                params![now.to_rfc3339()],
            )
            .unwrap();
        }

        let stats = db.list_speakers_with_stats().unwrap();

        assert_eq!(stats.len(), 2);
        assert!(stats.iter().all(|speaker| speaker.duplicate_name_conflict));
        assert!(stats
            .iter()
            .all(|speaker| speaker.duplicate_name_count == 2));
    }

    #[test]
    fn suggested_match_persists_and_acceptance_quarantines_incompatible_references() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 2_000).unwrap();
        let target = db.insert_speaker(Some("Alice")).unwrap();
        let source = db.insert_speaker(Some("VOICE7")).unwrap();
        db.insert_embedding(
            &target,
            &session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_embedding(
            &source,
            &session,
            &[0.96, 0.28],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_embedding(
            &source,
            &session,
            &[0.0, 1.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&source), Some("VOICE7"), "Hello")
            .unwrap();
        let provider_speakers = vec!["speaker_1".to_string()];
        db.insert_voice_match_decision(&VoiceMatchDecisionSave {
            session_id: &session,
            provider_speakers: &provider_speakers,
            resulting_speaker_id: Some(&source),
            best_speaker_id: Some(&target),
            runner_up_speaker_id: None,
            best_score: Some(0.96),
            runner_up_score: None,
            support_count: 1,
            selected_duration_ms: 4_000,
            selected_window_count: 1,
            consistency_score: Some(1.0),
            model_version: crate::embedding::EMBEDDING_VERSION,
            decision: "suggested",
            reason: "strong but ambiguous",
        })
        .unwrap();

        let before = db.list_speakers_with_stats().unwrap();
        let suggestion = before
            .iter()
            .find(|speaker| speaker.id == source)
            .and_then(|speaker| speaker.likely_match.as_ref())
            .unwrap();
        assert_eq!(suggestion.speaker_id, target);
        assert_eq!(suggestion.label, "Alice");

        let accepted = db.accept_voice_match_suggestion(&source, &target).unwrap();

        assert_eq!(accepted.activated_voiceprints, 1);
        assert_eq!(accepted.quarantined_voiceprints, 1);
        assert_eq!(
            db.list_embeddings(crate::embedding::EMBEDDING_VERSION)
                .unwrap()
                .len(),
            2
        );
        let (total, active): (i64, i64) = db
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(1), SUM(is_reference) FROM embeddings WHERE speaker_id=?1",
                params![target],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!((total, active), (3, 2));
        let segments = db.list_segments(&session).unwrap();
        assert_eq!(segments[0].speaker_id.as_deref(), Some(target.as_str()));
        assert_eq!(segments[0].speaker_label.as_deref(), Some("Alice"));
        let decisions = db.list_voice_match_decisions(&session).unwrap();
        assert_eq!(decisions.len(), 1);
        assert_eq!(
            decisions[0].resulting_speaker_id.as_deref(),
            Some(target.as_str())
        );
        assert_eq!(
            decisions[0].resolution.as_deref(),
            Some("accepted_suggestion")
        );
        assert!(decisions[0].resolved_at.is_some());
        assert_eq!(db.list_speakers().unwrap().len(), 1);
    }

    #[test]
    fn deleting_a_named_voice_used_by_history_is_blocked() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("Alice")).unwrap();
        db.insert_embedding(
            &speaker,
            &session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_sample(&speaker, "dGVzdA==", 16_000).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&speaker), Some("Alice"), "Hello")
            .unwrap();

        let error = db.delete_speaker(&speaker).unwrap_err();

        assert!(error.contains("used in 1 conversation"));
        assert_eq!(db.list_speakers().unwrap().len(), 1);
        assert_eq!(
            db.list_embeddings(crate::embedding::EMBEDDING_VERSION)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(db.list_samples(&speaker).unwrap().len(), 1);
        let segments = db.list_segments(&session).unwrap();
        assert_eq!(segments[0].speaker_id.as_deref(), Some(speaker.as_str()));
        assert_eq!(segments[0].speaker_label.as_deref(), Some("Alice"));
    }

    #[test]
    fn deleting_an_unused_named_voice_removes_its_private_artifacts() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("Alice")).unwrap();
        db.insert_embedding(
            &speaker,
            &session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_sample(&speaker, "dGVzdA==", 16_000).unwrap();

        db.delete_speaker(&speaker).unwrap();

        assert!(db.list_speakers().unwrap().is_empty());
        assert!(db
            .list_embeddings(crate::embedding::EMBEDDING_VERSION)
            .unwrap()
            .is_empty());
        assert!(db.list_samples(&speaker).unwrap().is_empty());
    }

    #[test]
    fn deleting_a_provisional_voice_can_unattribute_history() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("VOICE12")).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&speaker), Some("VOICE12"), "Hello")
            .unwrap();

        db.delete_speaker(&speaker).unwrap();

        assert!(db.list_speakers().unwrap().is_empty());
        let segments = db.list_segments(&session).unwrap();
        assert!(segments[0].speaker_id.is_none());
        assert!(segments[0].speaker_label.is_none());
    }

    #[test]
    fn merging_profiles_reassigns_history_and_combines_voiceprints() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 2_000).unwrap();
        let target = db.insert_speaker(Some("Alice")).unwrap();
        let source = db.insert_speaker(Some("VOICE2")).unwrap();
        db.insert_embedding(
            &target,
            &session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_embedding(
            &source,
            &session,
            &[0.9, 0.1],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_sample(&source, "dGVzdA==", 16_000).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&source), Some("VOICE2"), "Hello")
            .unwrap();

        db.merge_speakers(&source, &target, false).unwrap();

        assert_eq!(db.list_speakers().unwrap().len(), 1);
        let embeddings = db
            .list_embeddings(crate::embedding::EMBEDDING_VERSION)
            .unwrap();
        assert_eq!(embeddings.len(), 2);
        assert!(embeddings
            .iter()
            .all(|embedding| embedding.speaker_id == target));
        let segments = db.list_segments(&session).unwrap();
        assert_eq!(segments[0].speaker_id.as_deref(), Some(target.as_str()));
        assert_eq!(segments[0].speaker_label.as_deref(), Some("Alice"));
        assert!(db.list_samples(&target).unwrap().is_empty());
    }

    #[test]
    fn manual_merge_keeps_history_but_quarantines_a_conflicting_voiceprint() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 2_000).unwrap();
        let target = db.insert_speaker(Some("Alice")).unwrap();
        let source = db.insert_speaker(Some("VOICE2")).unwrap();
        db.insert_embedding(
            &target,
            &session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_embedding(
            &source,
            &session,
            &[0.0, 1.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&source), Some("VOICE2"), "Hello")
            .unwrap();

        let result = db.merge_speakers(&source, &target, false).unwrap();

        assert_eq!(result.activated_voiceprints, 0);
        assert_eq!(result.quarantined_voiceprints, 1);
        assert_eq!(
            db.list_embeddings(crate::embedding::EMBEDDING_VERSION)
                .unwrap()
                .len(),
            1
        );
        let segments = db.list_segments(&session).unwrap();
        assert_eq!(segments[0].speaker_id.as_deref(), Some(target.as_str()));
        assert_eq!(segments[0].speaker_label.as_deref(), Some("Alice"));
    }

    #[test]
    fn legacy_assignment_backs_up_and_promotes_an_import_owned_target() {
        let path = std::env::temp_dir().join(format!(
            "recall-legacy-identity-ownership-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let target = db.insert_speaker(Some("Imported Alice")).unwrap();
        let source = db.insert_speaker(Some("VOICE2")).unwrap();
        let local_session = db.insert_session("Local meeting", "", 1_000).unwrap();
        db.insert_segment(
            &local_session,
            0,
            1_000,
            Some(&source),
            Some("VOICE2"),
            "Local words",
        )
        .unwrap();
        db.conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO import_created_speakers(import_id, speaker_id)
                 VALUES('import-one', ?1)",
                params![target],
            )
            .unwrap();

        db.merge_speakers(&source, &target, false).unwrap();

        let owner_count: i64 = db
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(1) FROM import_created_speakers WHERE speaker_id=?1",
                params![target],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(owner_count, 0);
        let stem = path.file_stem().unwrap().to_string_lossy();
        let backup_prefix = format!("{stem}.pre-identity-merge-");
        let backup = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|candidate| {
                candidate
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|name| name.starts_with(&backup_prefix))
            })
            .expect("legacy assignment should create a verified backup");
        assert!(backup.is_file());
        drop(db);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(backup);
    }

    #[test]
    fn deleting_a_conversation_preserves_the_separate_voice_database() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("Alice")).unwrap();
        db.insert_embedding(
            &speaker,
            &session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&speaker), Some("Alice"), "Hello")
            .unwrap();

        assert_eq!(db.delete_session(&session).unwrap(), 0);

        assert!(db.list_sessions().unwrap().is_empty());
        assert_eq!(db.list_speakers().unwrap().len(), 1);
        assert_eq!(
            db.list_embeddings(crate::embedding::EMBEDDING_VERSION)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn agenda_and_recap_round_trip_and_are_deleted_with_the_conversation() {
        let db = memory_db();
        let session = db.insert_session("Before recap", "", 1_000).unwrap();
        let agenda = db
            .upsert_agenda(
                &session,
                "text",
                "Pasted agenda.txt",
                "text/plain",
                b"Introductions",
            )
            .unwrap();
        assert_eq!(
            agenda.metadata().text_content.as_deref(),
            Some("Introductions")
        );
        let recap = db
            .save_recap_and_title(RecapSave {
                session_id: &session,
                title: "Weekly planning",
                model: "test-model",
                prompt_version: crate::recap::PROMPT_VERSION,
                schema_version: crate::recap::SCHEMA_VERSION,
                source_fingerprint: "fingerprint",
                payload: &test_recap_payload(),
                input_tokens: 123,
                output_tokens: 45,
            })
            .unwrap();
        assert_eq!(recap.payload.meeting_title, "Weekly planning");
        assert_eq!(db.list_sessions().unwrap()[0].title, "Weekly planning");
        assert_eq!(
            db.load_agenda(&session).unwrap().unwrap().content,
            b"Introductions"
        );
        assert_eq!(
            db.load_recap(&session).unwrap().unwrap().source_fingerprint,
            "fingerprint"
        );
        db.update_recap_source_fingerprint(&session, "content-only-fingerprint")
            .unwrap();
        assert_eq!(
            db.load_recap(&session).unwrap().unwrap().source_fingerprint,
            "content-only-fingerprint"
        );

        db.delete_session(&session).unwrap();
        assert!(db.load_agenda(&session).unwrap().is_none());
        assert!(db.load_recap(&session).unwrap().is_none());
    }

    #[test]
    fn deleting_a_conversation_removes_its_orphan_provisional_voice() {
        let db = memory_db();
        let session = db.insert_session("Test", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("VOICE12")).unwrap();
        db.insert_embedding(
            &speaker,
            &session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_sample(&speaker, "dGVzdA==", 16_000).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&speaker), Some("VOICE12"), "Hello")
            .unwrap();

        assert_eq!(db.delete_session(&session).unwrap(), 1);
        assert!(db.list_speakers().unwrap().is_empty());
        assert!(db
            .list_embeddings(crate::embedding::EMBEDDING_VERSION)
            .unwrap()
            .is_empty());
        assert!(db.list_samples(&speaker).unwrap().is_empty());
    }

    #[test]
    fn deleting_one_conversation_keeps_a_provisional_voice_used_elsewhere() {
        let db = memory_db();
        let first = db.insert_session("First", "", 1_000).unwrap();
        let second = db.insert_session("Second", "", 1_000).unwrap();
        let speaker = db.insert_speaker(Some("VOICE4")).unwrap();
        db.insert_embedding(
            &speaker,
            &first,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_sample(&speaker, "dGVzdA==", 16_000).unwrap();
        for session in [&first, &second] {
            db.insert_segment(session, 0, 1_000, Some(&speaker), Some("VOICE4"), "Hello")
                .unwrap();
        }

        assert_eq!(db.delete_session(&first).unwrap(), 0);
        assert_eq!(db.list_speakers().unwrap().len(), 1);
        assert_eq!(db.list_samples(&speaker).unwrap().len(), 1);
        assert_eq!(
            db.list_embeddings(crate::embedding::EMBEDDING_VERSION)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            db.session_ids_for_speakers(&[speaker.as_str()]).unwrap(),
            vec![second]
        );
    }

    #[test]
    fn migration_keeps_only_the_oldest_legacy_embedding_as_a_reference() {
        let path = std::env::temp_dir().join(format!(
            "recall-reference-migration-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let backup = Db::reference_migration_backup_path(&path);
        let vector = general_purpose::STANDARD.encode(bytemuck::cast_slice(&[1.0_f32, 0.0]));
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE speakers (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    created_at TEXT NOT NULL
                 );
                 CREATE TABLE embeddings (
                    id TEXT PRIMARY KEY,
                    speaker_id TEXT,
                    vector_nonce TEXT,
                    vector_ct TEXT NOT NULL,
                    source_session_id TEXT,
                    created_at TEXT NOT NULL,
                    model_version TEXT
                 );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO speakers(id, label, created_at) VALUES('s1', 'Alice', '2026-01-01T00:00:00Z')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO embeddings VALUES('older', 's1', '', ?1, 'session-1', '2026-01-01T00:00:00Z', ?2)",
                params![vector, crate::embedding::EMBEDDING_VERSION],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO embeddings VALUES('newer', 's1', '', ?1, 'session-2', '2026-01-02T00:00:00Z', ?2)",
                params![vector, crate::embedding::EMBEDDING_VERSION],
            )
            .unwrap();
        }

        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        assert!(backup.is_file());
        let references = db
            .list_embeddings(crate::embedding::EMBEDDING_VERSION)
            .unwrap();
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].id, "older");
        let conn = db.conn.lock().unwrap();
        let total: i64 = conn
            .query_row("SELECT COUNT(1) FROM embeddings", [], |row| row.get(0))
            .unwrap();
        assert_eq!(total, 2);
        let marker: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key='voice_reference_migration_v1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(marker, "complete");
        drop(conn);
        drop(db);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(backup);
    }

    #[test]
    fn opening_a_legacy_database_adds_columns_without_losing_transcript() {
        let path = std::env::temp_dir().join(format!("recall-db-test-{}.sqlite", Uuid::new_v4()));
        let backup = Db::migration_backup_path(&path);
        let recap_backup = Db::recap_migration_backup_path(&path);
        let processing_backup = Db::processing_migration_backup_path(&path);
        let voice_match_backup = Db::voice_match_migration_backup_path(&path);
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
                 CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    transcript_nonce TEXT,
                    transcript_ct TEXT NOT NULL
                 );
                 CREATE TABLE speakers (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    created_at TEXT NOT NULL
                 );
                 CREATE TABLE embeddings (
                    id TEXT PRIMARY KEY,
                    speaker_id TEXT,
                    vector_nonce TEXT,
                    vector_ct TEXT NOT NULL,
                    source_session_id TEXT,
                    created_at TEXT NOT NULL
                 );
                 CREATE TABLE speaker_samples (
                    id TEXT PRIMARY KEY,
                    speaker_id TEXT NOT NULL,
                    sample_b64 TEXT NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                 );
                 CREATE TABLE segments (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    speaker_label TEXT,
                    text_nonce TEXT,
                    text_ct TEXT NOT NULL
                 );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO sessions(id, created_at, transcript_nonce, transcript_ct)
                 VALUES(?1, ?2, '', ?3)",
                params![
                    "legacy-session",
                    Utc::now().to_rfc3339(),
                    general_purpose::STANDARD.encode("Legacy transcript")
                ],
            )
            .unwrap();
        }

        {
            let db = Db::open(&path, Crypto::new(None, None)).unwrap();
            assert!(backup.is_file());
            assert!(voice_match_backup.is_file());
            let sessions = db.list_sessions().unwrap();
            assert_eq!(sessions.len(), 1);
            assert_eq!(sessions[0].id, "legacy-session");
            assert_eq!(sessions[0].title, "");
            assert_eq!(sessions[0].duration_ms, 0);
            assert_eq!(sessions[0].transcript, "Legacy transcript");
        }
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(backup).unwrap();
        std::fs::remove_file(recap_backup).unwrap();
        std::fs::remove_file(processing_backup).unwrap();
        std::fs::remove_file(voice_match_backup).unwrap();
    }

    #[test]
    fn identity_indexes_and_paginated_natural_order_are_available() {
        let db = memory_db();
        let voice_ten = db.insert_speaker(Some("VOICE10")).unwrap();
        let voice_two = db.insert_speaker(Some("VOICE2")).unwrap();
        db.insert_speaker(Some("Alice")).unwrap();
        let session = db.insert_session("Index test", "", 2_000).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&voice_ten), Some("VOICE10"), "Ten")
            .unwrap();
        db.insert_segment(
            &session,
            1_000,
            2_000,
            Some(&voice_two),
            Some("VOICE2"),
            "Two",
        )
        .unwrap();

        let first_page = db
            .list_identity_profiles("", "all", 1, 2)
            .expect("list first page");
        assert_eq!(first_page.total, 3);
        assert_eq!(first_page.page_count, 2);
        assert_eq!(
            first_page
                .items
                .iter()
                .map(|profile| profile.label.as_str())
                .collect::<Vec<_>>(),
            vec!["Alice", "VOICE2"]
        );
        assert_eq!(
            db.list_identity_profiles("", "provisional", 1, 100)
                .unwrap()
                .items
                .iter()
                .map(|profile| profile.label.as_str())
                .collect::<Vec<_>>(),
            vec!["VOICE2", "VOICE10"]
        );

        let conn = db.conn.lock().unwrap();
        let indexes = [
            "sessions_created_at_idx",
            "segments_speaker_session_idx",
            "segments_session_start_idx",
            "embeddings_speaker_model_reference_idx",
            "speaker_samples_speaker_idx",
        ];
        for index in indexes {
            let exists: bool = conn
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM sqlite_master
                         WHERE type='index' AND name=?1
                     )",
                    params![index],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(exists, "missing index {index}");
        }
        let plan: String = conn
            .query_row(
                "EXPLAIN QUERY PLAN
                 SELECT DISTINCT session_id FROM segments WHERE speaker_id=?1",
                params![voice_two],
                |row| row.get(3),
            )
            .unwrap();
        assert!(plan.contains("segments_speaker_session_idx"), "{plan}");
        let session_order_plan = {
            let mut stmt = conn
                .prepare(
                    "EXPLAIN QUERY PLAN
                     SELECT s.id, s.created_at, COALESCE(s.title, ''),
                            COALESCE(s.duration_ms, 0),
                            p.status, p.error, p.run_id, p.audio_path
                       FROM sessions s
                       LEFT JOIN processing_jobs p ON p.session_id = s.id
                      ORDER BY s.created_at DESC",
                )
                .unwrap();
            stmt.query_map([], |row| row.get::<_, String>(3))
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(
            session_order_plan.contains("sessions_created_at_idx"),
            "{session_order_plan}"
        );
        assert!(
            !session_order_plan.contains("USE TEMP B-TREE"),
            "{session_order_plan}"
        );
    }

    #[test]
    fn identical_unassigned_labels_remain_separate_conversation_scoped_groups() {
        let db = memory_db();
        let first = db.insert_session("First meeting", "", 1_000).unwrap();
        let second = db.insert_session("Second meeting", "", 1_000).unwrap();
        for session_id in [&first, &second] {
            db.insert_segment(session_id, 0, 1_000, None, Some("Speaker 1"), "Hello")
                .unwrap();
        }

        let page = db
            .list_unassigned_identities("Speaker 1", "generic", 1, 100)
            .unwrap();
        assert_eq!(page.total, 2);
        assert_ne!(page.items[0].key.session_id, page.items[1].key.session_id);
        assert!(page
            .items
            .iter()
            .all(|item| item.key.speaker_label.as_deref() == Some("Speaker 1")));
    }

    #[test]
    fn identity_consolidation_is_backed_up_atomic_and_quarantines_incompatible_vectors() {
        let path = std::env::temp_dir().join(format!(
            "recall-identity-consolidation-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let target = db.insert_speaker(Some("Alice")).unwrap();
        let source = db.insert_speaker(Some("VOICE12")).unwrap();
        let target_session = db.insert_session("Target", "", 1_000).unwrap();
        let source_session = db.insert_session("Source", "", 1_000).unwrap();
        let unassigned_session = db.insert_session("Unassigned", "", 1_000).unwrap();
        let untouched_session = db.insert_session("Untouched", "", 1_000).unwrap();
        db.insert_segment(
            &target_session,
            0,
            1_000,
            Some(&target),
            Some("Alice"),
            "Target words",
        )
        .unwrap();
        db.insert_segment(
            &source_session,
            0,
            1_000,
            Some(&source),
            Some("VOICE12"),
            "Source words",
        )
        .unwrap();
        db.insert_segment(
            &unassigned_session,
            0,
            1_000,
            None,
            Some("Speaker 1"),
            "Assigned words",
        )
        .unwrap();
        db.insert_segment(
            &untouched_session,
            0,
            1_000,
            None,
            Some("Speaker 1"),
            "Remain separate",
        )
        .unwrap();
        db.insert_embedding(
            &target,
            &target_session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_embedding(
            &source,
            &source_session,
            &[1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_embedding(
            &source,
            &source_session,
            &[-1.0, 0.0],
            crate::embedding::EMBEDDING_VERSION,
        )
        .unwrap();
        db.insert_sample(&source, "dGVzdA==", 16_000).unwrap();
        db.save_recap_and_title(RecapSave {
            session_id: &source_session,
            title: "Source",
            model: "test-model",
            prompt_version: crate::recap::PROMPT_VERSION,
            schema_version: crate::recap::SCHEMA_VERSION,
            source_fingerprint: "before-identity-change",
            payload: &test_recap_payload(),
            input_tokens: 10,
            output_tokens: 5,
        })
        .unwrap();
        let request = IdentityConsolidationRequest {
            profile_ids: vec![target.clone(), source.clone()],
            unassigned_groups: vec![UnassignedIdentityKey {
                session_id: unassigned_session.clone(),
                speaker_label: Some("Speaker 1".into()),
            }],
            target_speaker_id: Some(target.clone()),
            final_label: "Alice Example".into(),
        };

        let preview = db.preview_identity_consolidation(&request).unwrap();
        assert_eq!(preview.affected_conversation_count, 3);
        assert_eq!(preview.affected_intervention_count, 3);
        assert_eq!(preview.stale_recap_count, 1);
        assert_eq!(preview.samples_to_delete, 1);
        let result = db
            .consolidate_identities(&request, &preview.affected_session_ids)
            .unwrap();

        assert_eq!(result.target_speaker_id, target);
        assert_eq!(result.target_label, "Alice Example");
        assert_eq!(result.activated_voiceprints, 1);
        assert_eq!(result.quarantined_voiceprints, 1);
        assert_eq!(result.deleted_samples, 1);
        assert!(!db
            .list_speakers()
            .unwrap()
            .iter()
            .any(|speaker| speaker.id == source));
        assert!(db.list_samples(&target).unwrap().is_empty());
        for session_id in [&target_session, &source_session, &unassigned_session] {
            let segments = db.list_segments(session_id).unwrap();
            assert!(segments.iter().all(|segment| {
                segment.speaker_id.as_deref() == Some(target.as_str())
                    && segment.speaker_label.as_deref() == Some("Alice Example")
            }));
        }
        let untouched = db.list_segments(&untouched_session).unwrap();
        assert!(untouched[0].speaker_id.is_none());
        assert_eq!(untouched[0].speaker_label.as_deref(), Some("Speaker 1"));
        let sessions = db.list_sessions().unwrap();
        for session_id in [&target_session, &source_session, &unassigned_session] {
            let transcript = &sessions
                .iter()
                .find(|session| &session.id == session_id)
                .unwrap()
                .transcript;
            assert!(transcript.starts_with("Alice Example:"));
        }
        assert_eq!(
            db.load_recap(&source_session)
                .unwrap()
                .unwrap()
                .source_fingerprint,
            "before-identity-change"
        );
        let conn = db.conn.lock().unwrap();
        let active: i64 = conn
            .query_row(
                "SELECT COUNT(1) FROM embeddings
                  WHERE speaker_id=?1 AND is_reference=1",
                params![target],
                |row| row.get(0),
            )
            .unwrap();
        let inactive: i64 = conn
            .query_row(
                "SELECT COUNT(1) FROM embeddings
                  WHERE speaker_id=?1 AND is_reference=0",
                params![target],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!((active, inactive), (2, 1));
        drop(conn);

        let backup = PathBuf::from(result.backup_path);
        assert!(backup.is_file());
        let backup_conn = Connection::open(&backup).unwrap();
        let integrity: String = backup_conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .unwrap();
        assert_eq!(integrity, "ok");
        let source_still_in_backup: bool = backup_conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM speakers WHERE id=?1)",
                params![source],
                |row| row.get(0),
            )
            .unwrap();
        assert!(source_still_in_backup);
        drop(backup_conn);
        drop(db);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(backup);
    }

    #[test]
    fn identity_consolidation_promotes_import_owned_people_after_local_history_is_assigned() {
        let path = std::env::temp_dir().join(format!(
            "recall-identity-import-ownership-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let target = db.insert_speaker(Some("Imported Alice")).unwrap();
        let source = db.insert_speaker(Some("Imported Alicia")).unwrap();
        let imported_target_session = db.insert_session("Imported target", "", 1_000).unwrap();
        let imported_source_session = db.insert_session("Imported source", "", 1_000).unwrap();
        let local_session = db.insert_session("Local meeting", "", 1_000).unwrap();
        db.insert_segment(
            &imported_target_session,
            0,
            1_000,
            Some(&target),
            Some("Imported Alice"),
            "Imported target words",
        )
        .unwrap();
        db.insert_segment(
            &imported_source_session,
            0,
            1_000,
            Some(&source),
            Some("Imported Alicia"),
            "Imported source words",
        )
        .unwrap();
        db.insert_segment(
            &local_session,
            0,
            1_000,
            None,
            Some("Speaker 1"),
            "Local words",
        )
        .unwrap();
        {
            let conn = db.conn.lock().unwrap();
            for (session_id, fingerprint) in [
                (&imported_target_session, "target-fingerprint"),
                (&imported_source_session, "source-fingerprint"),
            ] {
                conn.execute(
                    "INSERT INTO imported_sessions(
                        source_provider, source_meeting_sha256, import_id, session_id
                     ) VALUES('Jamie', ?1, 'import-one', ?2)",
                    params![fingerprint, session_id],
                )
                .unwrap();
            }
            for speaker_id in [&target, &source] {
                conn.execute(
                    "INSERT INTO import_created_speakers(import_id, speaker_id)
                     VALUES('import-one', ?1)",
                    params![speaker_id],
                )
                .unwrap();
            }
        }
        let request = IdentityConsolidationRequest {
            profile_ids: vec![target.clone(), source],
            unassigned_groups: vec![UnassignedIdentityKey {
                session_id: local_session,
                speaker_label: Some("Speaker 1".into()),
            }],
            target_speaker_id: Some(target.clone()),
            final_label: "Alice".into(),
        };
        let preview = db.preview_identity_consolidation(&request).unwrap();
        assert_eq!(preview.imported_source_profile_count, 1);
        let result = db
            .consolidate_identities(&request, &preview.affected_session_ids)
            .unwrap();

        let owner_count: i64 = db
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(1) FROM import_created_speakers WHERE speaker_id=?1",
                params![target],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(owner_count, 0);
        drop(db);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(result.backup_path);
    }

    #[test]
    fn an_unassigned_group_can_create_a_name_only_person_without_claiming_similar_groups() {
        let path = std::env::temp_dir().join(format!(
            "recall-identity-new-person-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let selected_session = db.insert_session("Selected", "", 1_000).unwrap();
        let other_session = db.insert_session("Other", "", 1_000).unwrap();
        for session_id in [&selected_session, &other_session] {
            db.insert_segment(session_id, 0, 1_000, None, Some("Speaker 1"), "Hello")
                .unwrap();
        }
        let request = IdentityConsolidationRequest {
            profile_ids: Vec::new(),
            unassigned_groups: vec![UnassignedIdentityKey {
                session_id: selected_session.clone(),
                speaker_label: Some("Speaker 1".into()),
            }],
            target_speaker_id: None,
            final_label: "New Person".into(),
        };
        let preview = db.preview_identity_consolidation(&request).unwrap();
        assert!(preview.creates_new_person);
        let result = db
            .consolidate_identities(&request, &preview.affected_session_ids)
            .unwrap();

        let selected = db.list_segments(&selected_session).unwrap();
        assert_eq!(
            selected[0].speaker_id.as_deref(),
            Some(result.target_speaker_id.as_str())
        );
        assert_eq!(selected[0].speaker_label.as_deref(), Some("New Person"));
        let other = db.list_segments(&other_session).unwrap();
        assert!(other[0].speaker_id.is_none());
        assert_eq!(other[0].speaker_label.as_deref(), Some("Speaker 1"));
        assert!(db
            .list_embeddings(crate::embedding::EMBEDDING_VERSION)
            .unwrap()
            .is_empty());
        drop(db);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(result.backup_path);
    }

    #[test]
    fn identity_consolidation_rejects_duplicate_names_and_changed_preview_scope() {
        let path = std::env::temp_dir().join(format!(
            "recall-identity-revalidation-test-{}.sqlite",
            Uuid::new_v4()
        ));
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();
        let alice = db.insert_speaker(Some("Alice")).unwrap();
        let bob = db.insert_speaker(Some("Bob")).unwrap();
        let voice = db.insert_speaker(Some("VOICE2")).unwrap();
        let session = db.insert_session("Meeting", "", 1_000).unwrap();
        db.insert_segment(&session, 0, 1_000, Some(&voice), Some("VOICE2"), "Hello")
            .unwrap();
        let conflicting = IdentityConsolidationRequest {
            profile_ids: vec![voice.clone(), bob.clone()],
            unassigned_groups: Vec::new(),
            target_speaker_id: Some(voice.clone()),
            final_label: " alice ".into(),
        };
        assert!(db
            .preview_identity_consolidation(&conflicting)
            .unwrap_err()
            .contains("Alice"));

        let valid = IdentityConsolidationRequest {
            profile_ids: vec![voice.clone(), bob.clone()],
            unassigned_groups: Vec::new(),
            target_speaker_id: Some(bob),
            final_label: "Bob Example".into(),
        };
        let preview = db.preview_identity_consolidation(&valid).unwrap();
        let mut changed_scope = preview.affected_session_ids.clone();
        changed_scope.push("another-session".into());
        let error = db
            .consolidate_identities(&valid, &changed_scope)
            .unwrap_err();
        assert!(error.contains("impact preview"));
        assert!(db
            .list_speakers()
            .unwrap()
            .iter()
            .any(|speaker| speaker.id == voice));
        assert!(db
            .list_speakers()
            .unwrap()
            .iter()
            .any(|speaker| speaker.id == alice));
        drop(db);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    #[ignore = "requires RECALL_IDENTITY_BENCHMARK_DB pointing to a disposable database copy"]
    fn benchmark_identity_manager_on_disposable_snapshot() {
        let path = std::env::var("RECALL_IDENTITY_BENCHMARK_DB")
            .expect("set RECALL_IDENTITY_BENCHMARK_DB to a disposable database copy");
        let db = Db::open(&path, Crypto::new(None, None)).unwrap();

        db.list_identity_profiles("", "all", 1, 100).unwrap();
        let started = std::time::Instant::now();
        let profiles = db.list_identity_profiles("", "all", 1, 100).unwrap();
        let profiles_elapsed = started.elapsed();

        db.list_unassigned_identities("", "all", 1, 100).unwrap();
        let started = std::time::Instant::now();
        let unassigned = db.list_unassigned_identities("", "all", 1, 100).unwrap();
        let unassigned_elapsed = started.elapsed();

        let ordinary_profile = {
            let conn = db.conn.lock().unwrap();
            conn.query_row(
                "SELECT s.id
                   FROM speakers s
                   JOIN segments sg ON sg.speaker_id=s.id
                  GROUP BY s.id
                 HAVING COUNT(1) BETWEEN 1 AND 20
                  ORDER BY COUNT(1) DESC
                  LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap()
        };
        let started = std::time::Instant::now();
        db.rename_speaker(
            &ordinary_profile,
            &format!("Benchmark rename {}", Uuid::new_v4()),
        )
        .unwrap();
        let rename_elapsed = started.elapsed();

        let largest_profiles = {
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT s.id
                       FROM speakers s
                       JOIN segments sg ON sg.speaker_id=s.id
                      GROUP BY s.id
                      ORDER BY COUNT(1) DESC
                      LIMIT 2",
                )
                .unwrap();
            stmt.query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .map(Result::unwrap)
                .collect::<Vec<_>>()
        };
        assert_eq!(largest_profiles.len(), 2);
        let request = IdentityConsolidationRequest {
            profile_ids: largest_profiles.clone(),
            unassigned_groups: Vec::new(),
            target_speaker_id: Some(largest_profiles[1].clone()),
            final_label: format!("Benchmark merge {}", Uuid::new_v4()),
        };
        let preview = db.preview_identity_consolidation(&request).unwrap();
        let started = std::time::Instant::now();
        let result = db
            .consolidate_identities(&request, &preview.affected_session_ids)
            .unwrap();
        let merge_elapsed = started.elapsed();
        println!(
            "identity benchmark: profiles={} {:?}; unassigned={} {:?}; rename={:?}; largest merge conversations={} interventions={} {:?}",
            profiles.total,
            profiles_elapsed,
            unassigned.total,
            unassigned_elapsed,
            rename_elapsed,
            preview.affected_conversation_count,
            preview.affected_intervention_count,
            merge_elapsed,
        );
        let _ = std::fs::remove_file(result.backup_path);
    }
}

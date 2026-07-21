use std::{
    collections::HashSet,
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
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use uuid::Uuid;
use zeroize::Zeroize;

use crate::recap::RecapPayload;

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
}

#[derive(Debug, Clone, Serialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub title: String,
    pub duration_ms: i64,
    pub transcript: String,
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

impl Db {
    pub fn open(path: impl AsRef<Path>, crypto: Crypto) -> Result<Self, String> {
        let path = path.as_ref();
        Self::backup_before_migration(path)?;
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        let db = Db {
            conn: std::sync::Mutex::new(conn),
            crypto,
        };
        db.init_schema()?;
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
                eprintln!(
                    "[database] backed up the pre-recap database to {}",
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
        eprintln!(
            "[database] backed up the pre-migration database to {}",
            backup.display()
        );
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
                 );",
            )
            .map_err(|e| e.to_string())?;

        Self::add_column_if_missing(&conn_guard, "segments", "speaker_id", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "segments", "speaker_label", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "sessions", "title", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "sessions", "duration_ms", "INTEGER DEFAULT 0")?;
        Self::add_column_if_missing(&conn_guard, "embeddings", "model_version", "TEXT")?;
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

    pub fn list_sessions(&self) -> Result<Vec<Session>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, created_at, COALESCE(title, ''), COALESCE(duration_ms, 0), transcript_nonce, transcript_ct FROM sessions ORDER BY created_at DESC")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let created_at: String = row.get(1)?;
                let title: String = row.get(2)?;
                let duration_ms: i64 = row.get(3)?;
                let nonce: String = row.get(4)?;
                let ct: String = row.get(5)?;
                Ok((id, created_at, title, duration_ms, nonce, ct))
            })
            .map_err(|e| e.to_string())?;

        let mut sessions = Vec::new();
        for row in rows {
            let (id, created_at, title, duration_ms, nonce, ct) = row.map_err(|e| e.to_string())?;
            let ts = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            let transcript_bytes = self.crypto.decrypt(&nonce, &ct)?;
            let transcript = String::from_utf8(transcript_bytes).unwrap_or_default();
            sessions.push(Session {
                id,
                created_at: ts,
                title,
                duration_ms,
                transcript,
            });
        }
        Ok(sessions)
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
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
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
                "SELECT s.id, s.label, s.created_at,
                        (SELECT COUNT(1) FROM speaker_samples sm WHERE sm.speaker_id = s.id) as sample_count,
                        (SELECT COUNT(1) FROM embeddings e WHERE e.speaker_id = s.id AND e.model_version = ?1 AND e.is_reference = 1) as embedding_count,
                        (SELECT COUNT(DISTINCT sg.session_id) FROM segments sg WHERE sg.speaker_id = s.id) as conversation_count,
                        (SELECT MAX(se.created_at)
                           FROM segments sg
                           JOIN sessions se ON se.id = sg.session_id
                          WHERE sg.speaker_id = s.id) as last_seen_at
                 FROM speakers s
                 ORDER BY COALESCE(last_seen_at, s.created_at) DESC",
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

        let mut speakers = Vec::new();
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
            speakers.push(SpeakerStats {
                id,
                label,
                created_at,
                last_seen_at,
                sample_count: sample_count as usize,
                embedding_count: embedding_count as usize,
                conversation_count: conversation_count as usize,
            });
        }
        Ok(speakers)
    }

    pub fn session_ids_for_speakers(&self, speaker_ids: &[&str]) -> Result<Vec<String>, String> {
        if speaker_ids.is_empty() {
            return Ok(Vec::new());
        }
        let wanted = speaker_ids.iter().copied().collect::<HashSet<_>>();
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT DISTINCT session_id, speaker_id
                 FROM segments
                 WHERE speaker_id IS NOT NULL",
            )
            .map_err(|error| error.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|error| error.to_string())?;
        let mut sessions = HashSet::new();
        for row in rows {
            let (session_id, speaker_id) = row.map_err(|error| error.to_string())?;
            if wanted.contains(speaker_id.as_str()) {
                sessions.insert(session_id);
            }
        }
        let mut sessions = sessions.into_iter().collect::<Vec<_>>();
        sessions.sort();
        Ok(sessions)
    }

    pub fn rename_speaker(&self, speaker_id: &str, new_label: &str) -> Result<(), String> {
        let label = new_label.trim();
        if label.is_empty() {
            return Err("Speaker name cannot be empty".into());
        }
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        let changed = tx
            .execute(
                "UPDATE speakers SET label=?1 WHERE id=?2",
                params![label, speaker_id],
            )
            .map_err(|e| e.to_string())?;
        if changed == 0 {
            return Err("Speaker profile not found".into());
        }
        tx.execute(
            "UPDATE segments SET speaker_label=?1 WHERE speaker_id=?2",
            params![label, speaker_id],
        )
        .map_err(|e| e.to_string())?;
        // Discard samples after naming/renaming for privacy.
        tx.execute(
            "DELETE FROM speaker_samples WHERE speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn delete_speaker(&self, speaker_id: &str) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
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
        tx.execute("DELETE FROM speakers WHERE id=?1", params![speaker_id])
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn merge_speakers(
        &self,
        source_id: &str,
        target_id: &str,
        replace_target_voiceprints: bool,
    ) -> Result<(), String> {
        if source_id == target_id {
            return Err("Source and target speaker profiles must be different".into());
        }
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|e| e.to_string())?;
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
        if replace_target_voiceprints {
            tx.execute(
                "DELETE FROM embeddings WHERE speaker_id=?1",
                params![target_id],
            )
            .map_err(|e| e.to_string())?;
        }
        tx.execute(
            "UPDATE embeddings SET speaker_id=?1 WHERE speaker_id=?2",
            params![target_id, source_id],
        )
        .map_err(|e| e.to_string())?;
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
        tx.execute("DELETE FROM speakers WHERE id=?1", params![source_id])
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn insert_embedding(
        &self,
        speaker_id: &str,
        session_id: &str,
        vector: &[f32],
        model_version: &str,
    ) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let bytes: &[u8] = bytemuck::cast_slice(vector);
        let (nonce, ct) = self.crypto.encrypt(bytes);
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO embeddings(id, speaker_id, vector_nonce, vector_ct, source_session_id, created_at, model_version, is_reference) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 1)",
                params![id, speaker_id, nonce, ct, session_id, now.to_rfc3339(), model_version],
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

    fn test_recap_payload() -> RecapPayload {
        let localized = crate::recap::LocalizedText {
            original: "Summary".into(),
            english: "Summary".into(),
        };
        RecapPayload {
            meeting_title_english: "Weekly planning".into(),
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
        assert_eq!(recap.payload.meeting_title_english, "Weekly planning");
        assert_eq!(db.list_sessions().unwrap()[0].title, "Weekly planning");
        assert_eq!(
            db.load_agenda(&session).unwrap().unwrap().content,
            b"Introductions"
        );
        assert_eq!(
            db.load_recap(&session).unwrap().unwrap().source_fingerprint,
            "fingerprint"
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
    }
}

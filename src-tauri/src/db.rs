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
    pub sample_count: usize,
    pub embedding_count: usize,
    pub conversation_count: usize,
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
        let mut needs_migration = false;
        for (table, column) in [
            ("sessions", "title"),
            ("sessions", "duration_ms"),
            ("segments", "speaker_id"),
            ("embeddings", "model_version"),
        ] {
            if Self::table_exists(&conn, table)? && !Self::column_exists(&conn, table, column)? {
                needs_migration = true;
                break;
            }
        }
        drop(conn);
        if !needs_migration {
            return Ok(());
        }
        let backup = Self::migration_backup_path(path);
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
                    model_version TEXT
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
                 );",
            )
            .map_err(|e| e.to_string())?;

        Self::add_column_if_missing(&conn_guard, "segments", "speaker_id", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "segments", "speaker_label", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "sessions", "title", "TEXT")?;
        Self::add_column_if_missing(&conn_guard, "sessions", "duration_ms", "INTEGER DEFAULT 0")?;
        Self::add_column_if_missing(&conn_guard, "embeddings", "model_version", "TEXT")?;
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

    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let mut conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let tx = conn.transaction().map_err(|error| error.to_string())?;
        tx.execute(
            "DELETE FROM segments WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|error| error.to_string())?;
        let changed = tx
            .execute("DELETE FROM sessions WHERE id=?1", params![session_id])
            .map_err(|error| error.to_string())?;
        if changed == 0 {
            return Err("Conversation not found".into());
        }
        tx.commit().map_err(|error| error.to_string())?;
        Ok(())
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
                        (SELECT COUNT(1) FROM embeddings e WHERE e.speaker_id = s.id AND e.model_version = ?1) as embedding_count,
                        (SELECT COUNT(DISTINCT sg.session_id) FROM segments sg WHERE sg.speaker_id = s.id) as conversation_count
                 FROM speakers s
                 ORDER BY s.created_at ASC",
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
                Ok((
                    id,
                    label,
                    created_at,
                    sample_count,
                    embedding_count,
                    conversation_count,
                ))
            })
            .map_err(|e| e.to_string())?;

        let mut speakers = Vec::new();
        for row in rows {
            let (id, label, created_at, sample_count, embedding_count, conversation_count) =
                row.map_err(|e| e.to_string())?;
            let created_at = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            speakers.push(SpeakerStats {
                id,
                label,
                created_at,
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
        let changed = tx
            .execute("DELETE FROM speakers WHERE id=?1", params![speaker_id])
            .map_err(|e| e.to_string())?;
        if changed == 0 {
            return Err("Speaker profile not found".into());
        }
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
            .map(|label| !label.starts_with("VOICE"))
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
                "INSERT INTO embeddings(id, speaker_id, vector_nonce, vector_ct, source_session_id, created_at, model_version) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
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
                 WHERE e.model_version = ?1",
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

    #[test]
    fn voice_labels_are_monotonic() {
        let db = memory_db();
        db.insert_speaker(Some("Alice")).unwrap();
        db.insert_speaker(Some("VOICE9")).unwrap();
        assert_eq!(db.next_voice_label().unwrap(), "VOICE10");
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

        db.delete_session(&session).unwrap();

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
    fn opening_a_legacy_database_adds_columns_without_losing_transcript() {
        let path = std::env::temp_dir().join(format!("recall-db-test-{}.sqlite", Uuid::new_v4()));
        let backup = Db::migration_backup_path(&path);
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
    }
}

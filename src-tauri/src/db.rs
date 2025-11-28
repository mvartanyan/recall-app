use std::{collections::HashSet, path::Path, time::SystemTime};

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
            let salt_obj = SaltString::from_b64(&salt).unwrap_or_else(|_| SaltString::generate(&mut OsRng));
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
    pub encrypted: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub transcript: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Speaker {
    pub id: String,
    pub label: Option<String>,
    pub created_at: DateTime<Utc>,
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
}

impl Db {
    pub fn open(path: impl AsRef<Path>, crypto: Crypto) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        let encrypted = crypto.key.is_some();
        let db = Db {
            conn: std::sync::Mutex::new(conn),
            crypto,
            encrypted,
        };
        db.init_schema()?;
        db.persist_salt_if_missing()?;
        Ok(db)
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

    pub fn insert_session(&self, transcript: &str) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let (nonce, ct) = self.crypto.encrypt(transcript.as_bytes());
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO sessions(id, created_at, transcript_nonce, transcript_ct) VALUES(?1, ?2, ?3, ?4)",
                params![id, now.to_rfc3339(), nonce, ct],
            )
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        conn.execute("DELETE FROM sessions WHERE id=?1", params![session_id])
            .map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM segments WHERE session_id=?1",
            params![session_id],
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "DELETE FROM embeddings WHERE source_session_id=?1",
            params![session_id],
        )
        .map_err(|e| e.to_string())?;
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

    pub fn list_sessions(&self) -> Result<Vec<Session>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare("SELECT id, created_at, transcript_nonce, transcript_ct FROM sessions ORDER BY created_at DESC")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let created_at: String = row.get(1)?;
                let nonce: String = row.get(2)?;
                let ct: String = row.get(3)?;
                Ok((id, created_at, nonce, ct))
            })
            .map_err(|e| e.to_string())?;

        let mut sessions = Vec::new();
        for row in rows {
            let (id, created_at, nonce, ct) = row.map_err(|e| e.to_string())?;
            let ts = DateTime::parse_from_rfc3339(&created_at)
                .map_err(|e| e.to_string())?
                .with_timezone(&Utc);
            let transcript_bytes = self.crypto.decrypt(&nonce, &ct)?;
            let transcript = String::from_utf8(transcript_bytes).unwrap_or_default();
            sessions.push(Session {
                id,
                created_at: ts,
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
                Ok((id, session_id, start_ms, end_ms, speaker_id, speaker_label, nonce, ct))
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

    pub fn rename_speaker(&self, speaker_id: &str, new_label: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        conn.execute(
            "UPDATE speakers SET label=?1 WHERE id=?2",
            params![new_label, speaker_id],
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE segments SET speaker_label=?1 WHERE speaker_id=?2",
            params![new_label, speaker_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn delete_speaker(&self, speaker_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        conn.execute(
            "DELETE FROM embeddings WHERE speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|e| e.to_string())?;
        conn.execute("DELETE FROM speakers WHERE id=?1", params![speaker_id])
            .map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE segments SET speaker_id=NULL, speaker_label=NULL WHERE speaker_id=?1",
            params![speaker_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn insert_embedding(
        &self,
        speaker_id: &str,
        session_id: &str,
        vector: &[f32],
    ) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let now: DateTime<Utc> = SystemTime::now().into();
        let bytes: &[u8] = bytemuck::cast_slice(vector);
        let (nonce, ct) = self.crypto.encrypt(bytes);
        self.conn
            .lock()
            .map_err(|_| "lock poisoned".to_string())?
            .execute(
                "INSERT INTO embeddings(id, speaker_id, vector_nonce, vector_ct, source_session_id, created_at) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![id, speaker_id, nonce, ct, session_id, now.to_rfc3339()],
            )
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn list_embeddings(&self) -> Result<Vec<StoredEmbedding>, String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        let mut stmt = conn
            .prepare(
                "SELECT e.id, e.speaker_id, s.label, e.vector_nonce, e.vector_ct, e.source_session_id, e.created_at
                 FROM embeddings e
                 LEFT JOIN speakers s ON e.speaker_id = s.id",
            )
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let speaker_id: String = row.get(1)?;
                let speaker_label: Option<String> = row.get(2)?;
                let nonce: String = row.get(3)?;
                let ct: String = row.get(4)?;
                let source_session_id: String = row.get(5)?;
                let created_at: String = row.get(6)?;
                Ok((
                    id,
                    speaker_id,
                    speaker_label,
                    nonce,
                    ct,
                    source_session_id,
                    created_at,
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
            });
        }
        Ok(embeddings)
    }
}

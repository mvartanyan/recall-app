use std::{path::Path, time::SystemTime};

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::{password_hash::SaltString, Argon2};
use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use rand::RngCore;
use rusqlite::{params, Connection, OptionalExtension};
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
            let salt_obj = SaltString::new(&salt).expect("valid salt");
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

#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub transcript: String,
}

impl Db {
    pub fn open(path: impl AsRef<Path>, crypto: Crypto) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        let db = Db {
            conn: std::sync::Mutex::new(conn),
            crypto,
        };
        db.init_schema()?;
        if let Some(salt) = db.crypto.salt() {
            db.save_salt(&salt)?;
        }
        Ok(db)
    }

    fn init_schema(&self) -> Result<(), String> {
        let conn = self.conn.lock().map_err(|_| "lock poisoned".to_string())?;
        conn
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
                    text_nonce TEXT,
                    text_ct TEXT NOT NULL
                 );",
            )
            .map_err(|e| e.to_string())?;
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
}

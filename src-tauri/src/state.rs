use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::AppConfig;
use crate::db::{Crypto, Db};
use crate::ProgressEvent;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Mutex<Option<Arc<Db>>>>,
    pub data_dir: PathBuf,
    pub config_path: PathBuf,
    pub config: Arc<Mutex<AppConfig>>,
    pub embedder: Arc<Mutex<Option<crate::embedding::Embedder>>>,
    pub progress: Arc<Mutex<HashMap<String, Vec<ProgressEvent>>>>,
    pub model_path: PathBuf,
}

impl AppState {
    pub fn new(data_dir: PathBuf, model_path: PathBuf) -> Self {
        let config_path = data_dir.join("config.json");
        let config = AppConfig::load(&config_path);
        Self {
            db: Arc::new(Mutex::new(None)),
            data_dir,
            config_path,
            config: Arc::new(Mutex::new(config)),
            embedder: Arc::new(Mutex::new(None)),
            progress: Arc::new(Mutex::new(HashMap::new())),
            model_path,
        }
    }

    #[allow(dead_code)]
    pub fn save_config(&self) -> Result<(), String> {
        let cfg = self
            .config
            .lock()
            .map_err(|_| "config lock".to_string())?
            .clone();
        cfg.save(&self.config_path)
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("recall.db")
    }

    pub fn open_db(&self, crypto: Crypto) -> Result<(), String> {
        std::fs::create_dir_all(&self.data_dir).map_err(|e| e.to_string())?;
        let db_path = self.db_path();
        let db = Arc::new(Db::open(db_path, crypto)?);
        let mut guard = self.db.lock().map_err(|_| "db lock".to_string())?;
        *guard = Some(db);
        Ok(())
    }

    pub fn unlock_db(&self, crypto: Crypto) -> Result<(), String> {
        std::fs::create_dir_all(&self.data_dir).map_err(|error| error.to_string())?;
        let candidate = Arc::new(Db::open(self.db_path(), crypto)?);
        candidate
            .list_sessions()
            .map_err(|_| "The database password is incorrect or the data is damaged".to_string())?;
        let mut guard = self.db.lock().map_err(|_| "DB lock poisoned".to_string())?;
        *guard = Some(candidate);
        Ok(())
    }

    pub fn load_embedder(&self) -> Result<(), String> {
        if !self.model_path.is_file() {
            return Err(format!(
                "Speaker model is missing at {}",
                self.model_path.display()
            ));
        }
        let embedder = crate::embedding::Embedder::new(self.model_path.to_string_lossy().as_ref())?;
        let mut guard = self
            .embedder
            .lock()
            .map_err(|_| "embedder lock".to_string())?;
        *guard = Some(embedder);
        Ok(())
    }

    pub fn db_handle(&self) -> Result<Arc<Db>, String> {
        self.db
            .lock()
            .map_err(|_| "DB lock poisoned".to_string())?
            .clone()
            .ok_or_else(|| "Database not initialized".to_string())
    }
}

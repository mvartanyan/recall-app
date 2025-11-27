use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::AppConfig;
use crate::db::{Crypto, Db};

pub struct AppState {
    pub db: Arc<Mutex<Option<Db>>>,
    pub data_dir: PathBuf,
    pub config_path: PathBuf,
    pub config: Arc<Mutex<AppConfig>>,
    pub embedder: Arc<Mutex<Option<crate::embedding::Embedder>>>,
}

impl AppState {
    pub fn new(data_dir: PathBuf) -> Self {
        let config_path = data_dir.join("config.json");
        let config = AppConfig::load(&config_path);
        Self {
            db: Arc::new(Mutex::new(None)),
            data_dir,
            config_path,
            config: Arc::new(Mutex::new(config)),
            embedder: Arc::new(Mutex::new(None)),
        }
    }

    pub fn save_config(&self) -> Result<(), String> {
        let cfg = self.config.lock().map_err(|_| "config lock".to_string())?.clone();
        cfg.save(&self.config_path)
    }

    pub fn open_db(&self, crypto: Crypto) -> Result<(), String> {
        std::fs::create_dir_all(&self.data_dir).map_err(|e| e.to_string())?;
        let db_path = self.data_dir.join("recall.db");
        let db = Db::open(db_path, crypto)?;
        let mut guard = self.db.lock().map_err(|_| "db lock".to_string())?;
        *guard = Some(db);
        Ok(())
    }

    pub fn load_embedder(&self) -> Result<(), String> {
        let model_path = self.data_dir.join("models").join("spkrec-ecapa-voxceleb.onnx");
        if !model_path.exists() {
            return Err("ONNX model missing".into());
        }
        let embedder = crate::embedding::Embedder::new(model_path.to_string_lossy().as_ref())?;
        let mut guard = self.embedder.lock().map_err(|_| "embedder lock".to_string())?;
        *guard = Some(embedder);
        Ok(())
    }
}

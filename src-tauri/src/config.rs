use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use std::{fs, path::PathBuf};

#[skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub api_base: Option<String>,
    pub encryption_enabled: bool,
}

impl AppConfig {
    pub fn load(path: &PathBuf) -> Self {
        if let Ok(content) = fs::read_to_string(path) {
            if let Ok(cfg) = serde_json::from_str::<AppConfig>(&content) {
                return cfg;
            }
        }
        AppConfig::default()
    }

    pub fn save(&self, path: &PathBuf) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, content).map_err(|e| e.to_string())
    }
}

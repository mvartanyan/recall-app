use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub encryption_enabled: bool,
    pub selected_input_device: Option<String>,
    pub language_hints: Vec<String>,
    pub live_transcription: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            encryption_enabled: false,
            selected_input_device: None,
            language_hints: vec!["en", "fr", "de", "es", "ru"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            live_transcription: true,
        }
    }
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

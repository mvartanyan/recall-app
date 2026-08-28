use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub encryption_enabled: bool,
    pub selected_input_device: Option<String>,
    pub language_hints: Vec<String>,
    pub live_transcription: bool,
    pub openai_model: String,
    pub preferred_language: String,
    pub no_translation_languages: Vec<String>,
    pub onboarding_version: Option<String>,
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
            openai_model: "gpt-5.6-terra".to_string(),
            preferred_language: "en".to_string(),
            no_translation_languages: Vec::new(),
            onboarding_version: None,
        }
    }
}

impl AppConfig {
    pub fn load(path: &PathBuf) -> Self {
        if let Ok(content) = fs::read_to_string(path) {
            let had_legacy_expected_speakers = serde_json::from_str::<serde_json::Value>(&content)
                .ok()
                .and_then(|value| value.get("expected_speakers").cloned())
                .is_some();
            if let Ok(mut cfg) = serde_json::from_str::<AppConfig>(&content) {
                let previous_preferred = cfg.preferred_language.clone();
                let previous_exclusions = cfg.no_translation_languages.clone();
                cfg.preferred_language = normalized_base_language(&cfg.preferred_language)
                    .unwrap_or_else(|| "en".to_string());
                cfg.no_translation_languages = cfg
                    .no_translation_languages
                    .iter()
                    .filter_map(|language| normalized_base_language(language))
                    .filter(|language| language != &cfg.preferred_language)
                    .collect();
                cfg.no_translation_languages.sort();
                cfg.no_translation_languages.dedup();
                if cfg.preferred_language != previous_preferred
                    || cfg.no_translation_languages != previous_exclusions
                    || had_legacy_expected_speakers
                {
                    let _ = cfg.save(path);
                }
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

fn normalized_base_language(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase().replace('_', "-");
    normalized
        .split('-')
        .next()
        .filter(|language| !language.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_config_defaults_to_english_and_removes_the_preferred_exclusion() {
        let path =
            std::env::temp_dir().join(format!("recall-config-{}.json", uuid::Uuid::new_v4()));
        fs::write(
            &path,
            r#"{
                "language_hints": ["de"],
                "expected_speakers": 4,
                "no_translation_languages": ["EN-us", "de-DE", "de"]
            }"#,
        )
        .unwrap();
        let config = AppConfig::load(&path);
        assert_eq!(config.preferred_language, "en");
        assert_eq!(config.no_translation_languages, vec!["de"]);
        let saved = fs::read_to_string(&path).unwrap();
        assert!(saved.contains("\"preferred_language\": \"en\""));
        assert!(!saved.contains("expected_speakers"));
        fs::remove_file(path).unwrap();
    }
}

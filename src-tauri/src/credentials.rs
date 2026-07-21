use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

pub const SONIOX_KEY_FILENAME: &str = "soniox-api-key";
pub const OPENAI_KEY_FILENAME: &str = "openai-api-key";

pub fn save_api_key(path: &Path, api_key: &str) -> Result<(), String> {
    save_provider_api_key(path, api_key, "Soniox")
}

pub fn save_openai_api_key(path: &Path, api_key: &str) -> Result<(), String> {
    save_provider_api_key(path, api_key, "OpenAI")
}

fn save_provider_api_key(path: &Path, api_key: &str, provider: &str) -> Result<(), String> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(format!("{provider} API key cannot be empty"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("{provider} key path has no parent directory"))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("Could not create Recall's data directory: {error}"))?;

    let temporary_path = path.with_extension("tmp");
    let result = (|| {
        let mut options = OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options
            .open(&temporary_path)
            .map_err(|error| format!("Could not create the local {provider} key file: {error}"))?;
        #[cfg(unix)]
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("Could not restrict the {provider} key file: {error}"))?;
        file.write_all(api_key.as_bytes())
            .map_err(|error| format!("Could not write the local {provider} key file: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("Could not finish writing the {provider} key: {error}"))?;
        fs::rename(&temporary_path, path)
            .map_err(|error| format!("Could not install the local {provider} key file: {error}"))?;
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|error| format!("Could not restrict the {provider} key file: {error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result
}

pub fn load_api_key(path: &Path) -> Result<String, String> {
    load_provider_api_key(path, "Soniox")
}

pub fn load_openai_api_key(path: &Path) -> Result<String, String> {
    load_provider_api_key(path, "OpenAI")
}

fn load_provider_api_key(path: &Path, provider: &str) -> Result<String, String> {
    let value = fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            format!("{provider} API key is not configured")
        } else {
            format!("Could not read the local {provider} API key: {error}")
        }
    })?;
    let value = value.trim();
    if value.is_empty() {
        Err(format!("The local {provider} API key file is empty"))
    } else {
        Ok(value.to_string())
    }
}

pub fn has_api_key(path: &Path) -> bool {
    load_api_key(path).is_ok()
}

pub fn has_openai_api_key(path: &Path) -> bool {
    load_openai_api_key(path).is_ok()
}

pub fn delete_api_key(path: &Path) -> Result<(), String> {
    delete_provider_api_key(path, "Soniox")
}

pub fn delete_openai_api_key(path: &Path) -> Result<(), String> {
    delete_provider_api_key(path, "OpenAI")
}

fn delete_provider_api_key(path: &Path, provider: &str) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "Could not remove the local {provider} API key: {error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_key_round_trip_trims_and_removes_value() {
        let directory =
            std::env::temp_dir().join(format!("recall-key-test-{}", uuid::Uuid::new_v4()));
        let path = directory.join(SONIOX_KEY_FILENAME);
        save_api_key(&path, "  test-key  ").unwrap();
        assert_eq!(load_api_key(&path).unwrap(), "test-key");
        assert!(has_api_key(&path));
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        delete_api_key(&path).unwrap();
        assert!(!has_api_key(&path));
        let _ = fs::remove_dir(directory);
    }

    #[test]
    fn empty_key_is_rejected_without_creating_a_file() {
        let directory =
            std::env::temp_dir().join(format!("recall-key-test-{}", uuid::Uuid::new_v4()));
        let path = directory.join(SONIOX_KEY_FILENAME);
        assert!(save_api_key(&path, "  ").is_err());
        assert!(!path.exists());
    }

    #[test]
    fn openai_key_uses_the_same_user_only_storage_contract() {
        let directory =
            std::env::temp_dir().join(format!("recall-openai-key-test-{}", uuid::Uuid::new_v4()));
        let path = directory.join(OPENAI_KEY_FILENAME);
        save_openai_api_key(&path, "  openai-test-key  ").unwrap();
        assert_eq!(load_openai_api_key(&path).unwrap(), "openai-test-key");
        assert!(has_openai_api_key(&path));
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        delete_openai_api_key(&path).unwrap();
        assert!(!has_openai_api_key(&path));
        let _ = fs::remove_dir(directory);
    }
}

const SERVICE: &str = "com.example.recall.soniox";
const ACCOUNT: &str = "api-key";

fn entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE, ACCOUNT)
        .map_err(|error| format!("Could not open macOS Keychain: {error}"))
}

pub fn save_api_key(api_key: &str) -> Result<(), String> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err("Soniox API key cannot be empty".into());
    }
    entry()?
        .set_password(api_key)
        .map_err(|error| format!("Could not save the Soniox key in macOS Keychain: {error}"))
}

pub fn load_api_key() -> Result<String, String> {
    match entry()?.get_password() {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => Err("The Soniox key stored in Keychain is empty".into()),
        Err(keyring::Error::NoEntry) => Err("Soniox API key is not configured".into()),
        Err(error) => Err(format!(
            "Could not read the Soniox key from macOS Keychain: {error}"
        )),
    }
}

pub fn has_api_key() -> bool {
    load_api_key().is_ok()
}

pub fn delete_api_key() -> Result<(), String> {
    match entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(format!(
            "Could not remove the Soniox key from macOS Keychain: {error}"
        )),
    }
}

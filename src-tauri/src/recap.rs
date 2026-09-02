use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const PROMPT_VERSION: &str = "recall-recap-v6";
pub const SCHEMA_VERSION: &str = "recall-recap-schema-v5";

pub const BUILTIN_EXECUTIVE_SUMMARY_ID: &str = "builtin-executive-summary";
pub const BUILTIN_FULL_SUMMARY_ID: &str = "builtin-full-summary";
pub const BUILTIN_ACTIONS_ID: &str = "builtin-actions";

pub const DEFAULT_EXECUTIVE_SUMMARY_PROMPT: &str = "Write a concise account of the meeting's purpose, conclusions, decisions, material risks, disagreements, and open questions.";
pub const DEFAULT_FULL_SUMMARY_PROMPT: &str = "Write a detailed, sectioned account of the topics discussed, arguments, rationale, decisions, dependencies, risks, and next steps.";
pub const DEFAULT_ACTIONS_PROMPT: &str = "Capture explicit future commitments and actions reported as already completed. Include the participant, stated timing, and uncertainty. Exclude suggestions and possibilities.";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StandardRecapPrompts {
    pub executive_summary: String,
    pub full_summary: String,
    pub actions: String,
}

impl Default for StandardRecapPrompts {
    fn default() -> Self {
        Self {
            executive_summary: DEFAULT_EXECUTIVE_SUMMARY_PROMPT.to_string(),
            full_summary: DEFAULT_FULL_SUMMARY_PROMPT.to_string(),
            actions: DEFAULT_ACTIONS_PROMPT.to_string(),
        }
    }
}

fn default_target_language() -> String {
    "en".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalizedText {
    pub original: String,
    #[serde(alias = "english")]
    pub translated: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SummarySection {
    pub heading: LocalizedText,
    pub body: LocalizedText,
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionItem {
    pub participant: String,
    pub statement: LocalizedText,
    pub stated_timing: LocalizedText,
    pub uncertainty: LocalizedText,
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgendaCoverageItem {
    pub agenda_item: LocalizedText,
    pub status: String,
    pub statement: LocalizedText,
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslationAnnotation {
    pub segment_id: String,
    pub source_excerpt: String,
    pub language: String,
    #[serde(alias = "english_translation")]
    pub translated_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecapPayload {
    #[serde(default = "default_target_language")]
    pub target_language: String,
    #[serde(alias = "meeting_title_english")]
    pub meeting_title: String,
    pub dominant_language: String,
    pub executive_summary: LocalizedText,
    pub full_summary: Vec<SummarySection>,
    pub commitments: Vec<ActionItem>,
    pub actions_already_taken: Vec<ActionItem>,
    pub agenda_present: bool,
    pub agenda_coverage: Vec<AgendaCoverageItem>,
    pub translations: Vec<TranslationAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CustomRecapPayload {
    pub target_language: String,
    pub content_markdown: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecapSourceSegment {
    pub id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker_id: Option<String>,
    pub speaker_label: String,
    pub text: String,
}

#[derive(Debug, Clone, Copy)]
pub struct AgendaFingerprint<'a> {
    pub source_kind: &'a str,
    pub filename: &'a str,
    pub mime_type: &'a str,
    pub content: &'a [u8],
}

#[derive(Serialize)]
struct FingerprintInput<'a> {
    segments: &'a [RecapSourceSegment],
    agenda: Option<FingerprintAgenda<'a>>,
}

#[derive(Serialize)]
struct LegacyFingerprintInput<'a> {
    segments: &'a [RecapSourceSegment],
    agenda: Option<FingerprintAgenda<'a>>,
    no_translation_languages: Vec<String>,
}

#[derive(Serialize)]
struct FingerprintAgenda<'a> {
    source_kind: &'a str,
    filename: &'a str,
    mime_type: &'a str,
    content_sha256: String,
}

pub fn source_fingerprint(
    segments: &[RecapSourceSegment],
    agenda: Option<AgendaFingerprint<'_>>,
) -> Result<String, String> {
    let agenda = agenda.map(|value| FingerprintAgenda {
        source_kind: value.source_kind,
        filename: value.filename,
        mime_type: value.mime_type,
        content_sha256: hex_sha256(value.content),
    });
    let input = FingerprintInput { segments, agenda };
    let bytes = serde_json::to_vec(&input)
        .map_err(|error| format!("Could not fingerprint recap sources: {error}"))?;
    Ok(hex_sha256(&bytes))
}

pub fn legacy_source_fingerprint(
    segments: &[RecapSourceSegment],
    agenda: Option<AgendaFingerprint<'_>>,
    no_translation_languages: &[String],
) -> Result<String, String> {
    let agenda = agenda.map(|value| FingerprintAgenda {
        source_kind: value.source_kind,
        filename: value.filename,
        mime_type: value.mime_type,
        content_sha256: hex_sha256(value.content),
    });
    let mut languages = no_translation_languages
        .iter()
        .map(|language| language.trim().to_lowercase())
        .filter(|language| !language.is_empty() && language != "en")
        .collect::<Vec<_>>();
    languages.sort();
    languages.dedup();
    let input = LegacyFingerprintInput {
        segments,
        agenda,
        no_translation_languages: languages,
    };
    let bytes = serde_json::to_vec(&input)
        .map_err(|error| format!("Could not fingerprint legacy recap sources: {error}"))?;
    Ok(hex_sha256(&bytes))
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn validate_payload(
    payload: &RecapPayload,
    valid_segment_ids: &HashSet<String>,
    agenda_present: bool,
) -> Result<(), String> {
    if payload.target_language.trim().is_empty() {
        return Err("The LLM provider returned no target language".into());
    }
    if payload.meeting_title.trim().is_empty() {
        return Err("The LLM provider returned an empty meeting title".into());
    }
    if payload.dominant_language.trim().is_empty() {
        return Err("The LLM provider returned no dominant meeting language".into());
    }
    validate_localized(&payload.executive_summary, "executive summary")?;
    if payload.full_summary.is_empty() {
        return Err("The LLM provider returned no full-summary sections".into());
    }
    for (index, section) in payload.full_summary.iter().enumerate() {
        validate_localized(&section.heading, &format!("summary heading {}", index + 1))?;
        validate_localized(&section.body, &format!("summary section {}", index + 1))?;
        validate_required_evidence(
            &section.evidence_segment_ids,
            valid_segment_ids,
            "full-summary section",
        )?;
    }
    for item in payload
        .commitments
        .iter()
        .chain(payload.actions_already_taken.iter())
    {
        if item.participant.trim().is_empty() {
            return Err("The LLM provider returned an action without a participant".into());
        }
        validate_localized(&item.statement, "action statement")?;
        validate_required_evidence(&item.evidence_segment_ids, valid_segment_ids, "action item")?;
    }
    if payload.agenda_present != agenda_present {
        return Err("The LLM provider returned inconsistent agenda metadata".into());
    }
    if !agenda_present && !payload.agenda_coverage.is_empty() {
        return Err("The LLM provider returned agenda coverage when no agenda was supplied".into());
    }
    if agenda_present && payload.agenda_coverage.is_empty() {
        return Err("The LLM provider returned no agenda coverage for the supplied agenda".into());
    }
    for item in &payload.agenda_coverage {
        if !matches!(
            item.status.as_str(),
            "covered" | "partial" | "not-covered" | "unreadable"
        ) {
            return Err(format!(
                "The LLM provider returned an unsupported agenda status: {}",
                item.status
            ));
        }
        validate_localized(&item.agenda_item, "agenda item")?;
        validate_localized(&item.statement, "agenda coverage statement")?;
        if matches!(item.status.as_str(), "covered" | "partial") {
            validate_required_evidence(
                &item.evidence_segment_ids,
                valid_segment_ids,
                "covered agenda item",
            )?;
        } else {
            validate_evidence(&item.evidence_segment_ids, valid_segment_ids)?;
        }
    }
    for translation in &payload.translations {
        if !valid_segment_ids.contains(&translation.segment_id) {
            return Err(format!(
                "The LLM translation references an unknown segment: {}",
                translation.segment_id
            ));
        }
        if translation.source_excerpt.trim().is_empty()
            || translation.language.trim().is_empty()
            || translation.translated_text.trim().is_empty()
        {
            return Err("The LLM provider returned an incomplete translation annotation".into());
        }
    }
    Ok(())
}

pub fn validate_custom_recap_payload(
    payload: &CustomRecapPayload,
    target_language: &str,
) -> Result<(), String> {
    if payload.target_language != target_language {
        return Err(format!(
            "The LLM provider returned target language {} instead of {}",
            payload.target_language, target_language
        ));
    }
    if payload.content_markdown.trim().is_empty() {
        return Err("The LLM provider returned an empty custom recap".into());
    }
    Ok(())
}

fn validate_localized(value: &LocalizedText, field: &str) -> Result<(), String> {
    if value.original.trim().is_empty() || value.translated.trim().is_empty() {
        Err(format!("The LLM provider returned an incomplete {field}"))
    } else {
        Ok(())
    }
}

fn validate_evidence(ids: &[String], valid_segment_ids: &HashSet<String>) -> Result<(), String> {
    for id in ids {
        if !valid_segment_ids.contains(id) {
            return Err(format!("The LLM cited an unknown transcript segment: {id}"));
        }
    }
    Ok(())
}

fn validate_required_evidence(
    ids: &[String],
    valid_segment_ids: &HashSet<String>,
    field: &str,
) -> Result<(), String> {
    if ids.is_empty() {
        return Err(format!(
            "The LLM provider returned a {field} without transcript evidence"
        ));
    }
    validate_evidence(ids, valid_segment_ids)
}

pub fn analysis_response_schema(valid_segment_ids: &[String], target_language: &str) -> Value {
    let localized = || {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "original": { "type": "string" },
                "translated": { "type": "string" }
            },
            "required": ["original", "translated"]
        })
    };
    let evidence = || {
        json!({
            "type": "array",
            "items": { "$ref": "#/$defs/segment_id" },
            "minItems": 1
        })
    };
    let optional_evidence =
        || json!({ "type": "array", "items": { "$ref": "#/$defs/segment_id" } });
    let summary_section = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "heading": localized(),
            "body": localized(),
            "evidence_segment_ids": evidence()
        },
        "required": ["heading", "body", "evidence_segment_ids"]
    });
    let action_item = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "participant": { "type": "string" },
            "statement": localized(),
            "stated_timing": localized(),
            "uncertainty": localized(),
            "evidence_segment_ids": evidence()
        },
        "required": [
            "participant",
            "statement",
            "stated_timing",
            "uncertainty",
            "evidence_segment_ids"
        ]
    });
    let agenda_item = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "agenda_item": localized(),
            "status": {
                "type": "string",
                "enum": ["covered", "partial", "not-covered", "unreadable"]
            },
            "statement": localized(),
            "evidence_segment_ids": optional_evidence()
        },
        "required": ["agenda_item", "status", "statement", "evidence_segment_ids"]
    });
    json!({
        "type": "object",
        "additionalProperties": false,
        "$defs": {
            "segment_id": {
                "type": "string",
                "enum": valid_segment_ids
            }
        },
        "properties": {
            "target_language": { "type": "string", "enum": [target_language] },
            "meeting_title": { "type": "string" },
            "dominant_language": { "type": "string" },
            "executive_summary": localized(),
            "full_summary": { "type": "array", "items": summary_section },
            "commitments": { "type": "array", "items": action_item.clone() },
            "actions_already_taken": { "type": "array", "items": action_item },
            "agenda_present": { "type": "boolean" },
            "agenda_coverage": { "type": "array", "items": agenda_item }
        },
        "required": [
            "target_language",
            "meeting_title",
            "dominant_language",
            "executive_summary",
            "full_summary",
            "commitments",
            "actions_already_taken",
            "agenda_present",
            "agenda_coverage"
        ]
    })
}

pub fn translation_response_schema(valid_segment_ids: &[String]) -> Value {
    let translation = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "segment_id": { "$ref": "#/$defs/segment_id" },
            "source_excerpt": { "type": "string", "enum": [""] },
            "language": { "type": "string" },
            "translated_text": { "type": "string" }
        },
        "required": ["segment_id", "source_excerpt", "language", "translated_text"]
    });
    json!({
        "type": "object",
        "additionalProperties": false,
        "$defs": {
            "segment_id": {
                "type": "string",
                "enum": valid_segment_ids
            }
        },
        "properties": {
            "translations": {
                "type": "array",
                "items": translation,
                "minItems": valid_segment_ids.len(),
                "maxItems": valid_segment_ids.len()
            }
        },
        "required": ["translations"]
    })
}

pub fn custom_recap_response_schema(target_language: &str) -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "target_language": { "type": "string", "enum": [target_language] },
            "content_markdown": { "type": "string" }
        },
        "required": ["target_language", "content_markdown"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_changes_for_transcript_or_agenda() {
        let segment = RecapSourceSegment {
            id: "seg-1".into(),
            start_ms: 0,
            end_ms: 1_000,
            speaker_id: Some("speaker-1".into()),
            speaker_label: "Alice".into(),
            text: "Bonjour".into(),
        };
        let baseline = source_fingerprint(std::slice::from_ref(&segment), None).unwrap();
        let with_agenda = source_fingerprint(
            std::slice::from_ref(&segment),
            Some(AgendaFingerprint {
                source_kind: "text",
                filename: "Agenda.txt",
                mime_type: "text/plain",
                content: b"Introductions",
            }),
        )
        .unwrap();
        let mut changed = segment;
        changed.text = "Bonsoir".into();
        let with_edit = source_fingerprint(&[changed], None).unwrap();
        assert_ne!(baseline, with_agenda);
        assert_ne!(baseline, with_edit);
    }

    #[test]
    fn translation_preferences_do_not_participate_in_the_source_fingerprint() {
        let segments = Vec::new();
        assert_eq!(source_fingerprint(&segments, None).unwrap().len(), 64);
        assert_ne!(
            legacy_source_fingerprint(&segments, None, &[]).unwrap(),
            legacy_source_fingerprint(&segments, None, &["de".into()]).unwrap()
        );
    }

    #[test]
    fn shipped_recap_types_have_stable_distinct_ids_and_default_prompts() {
        let ids = HashSet::from([
            BUILTIN_EXECUTIVE_SUMMARY_ID,
            BUILTIN_FULL_SUMMARY_ID,
            BUILTIN_ACTIONS_ID,
        ]);
        assert_eq!(ids.len(), 3);

        let prompts = StandardRecapPrompts::default();
        assert_eq!(prompts.executive_summary, DEFAULT_EXECUTIVE_SUMMARY_PROMPT);
        assert_eq!(prompts.full_summary, DEFAULT_FULL_SUMMARY_PROMPT);
        assert_eq!(prompts.actions, DEFAULT_ACTIONS_PROMPT);
        assert!(prompts.executive_summary.contains("material risks"));
        assert!(prompts.full_summary.contains("dependencies"));
        assert!(prompts
            .actions
            .contains("Exclude suggestions and possibilities"));
    }

    #[test]
    fn strict_schema_closes_every_object_shape() {
        fn inspect(value: &Value) {
            if value.get("type") == Some(&Value::String("object".into())) {
                assert_eq!(value.get("additionalProperties"), Some(&Value::Bool(false)));
                let properties = value
                    .get("properties")
                    .and_then(Value::as_object)
                    .expect("object properties");
                let required = value
                    .get("required")
                    .and_then(Value::as_array)
                    .expect("required properties");
                assert_eq!(properties.len(), required.len());
            }
            match value {
                Value::Array(values) => values.iter().for_each(inspect),
                Value::Object(values) => values.values().for_each(inspect),
                _ => {}
            }
        }
        let ids = ["segment-1".into()];
        inspect(&analysis_response_schema(&ids, "de"));
        inspect(&translation_response_schema(&ids));
        inspect(&custom_recap_response_schema("de"));
    }

    #[test]
    fn custom_recap_schema_and_validation_require_language_and_markdown() {
        let schema = custom_recap_response_schema("de");
        assert_eq!(
            schema.pointer("/properties/target_language/enum"),
            Some(&json!(["de"]))
        );
        assert_eq!(
            schema.pointer("/required"),
            Some(&json!(["target_language", "content_markdown"]))
        );
        assert!(validate_custom_recap_payload(
            &CustomRecapPayload {
                target_language: "de".into(),
                content_markdown: "# Risikoanalyse".into(),
            },
            "de"
        )
        .is_ok());
        assert!(validate_custom_recap_payload(
            &CustomRecapPayload {
                target_language: "en".into(),
                content_markdown: "# Risks".into(),
            },
            "de"
        )
        .unwrap_err()
        .contains("instead of de"));
        assert!(validate_custom_recap_payload(
            &CustomRecapPayload {
                target_language: "de".into(),
                content_markdown: "\n".into(),
            },
            "de"
        )
        .unwrap_err()
        .contains("empty custom recap"));
    }

    #[test]
    fn generated_claims_require_transcript_evidence() {
        let valid = HashSet::from(["segment-1".to_string()]);
        assert!(validate_required_evidence(&[], &valid, "action item")
            .unwrap_err()
            .contains("without transcript evidence"));
        assert!(validate_required_evidence(&["segment-1".into()], &valid, "action item").is_ok());
    }

    #[test]
    fn strict_schema_requires_evidence_for_summaries_and_actions() {
        let schema = analysis_response_schema(&["segment-1".into()], "de");
        assert_eq!(
            schema
                .pointer("/properties/full_summary/items/properties/evidence_segment_ids/minItems"),
            Some(&Value::from(1))
        );
        assert_eq!(
            schema
                .pointer("/properties/commitments/items/properties/evidence_segment_ids/minItems"),
            Some(&Value::from(1))
        );
        assert!(schema
            .pointer("/properties/agenda_coverage/items/properties/evidence_segment_ids/minItems")
            .is_none());
    }

    #[test]
    fn strict_schema_limits_every_segment_reference_to_supplied_ids() {
        let valid_ids = vec!["segment-1".to_string(), "segment-2".to_string()];
        let analysis_schema = analysis_response_schema(&valid_ids, "de");
        assert_eq!(
            analysis_schema.pointer("/$defs/segment_id/enum"),
            Some(&json!(["segment-1", "segment-2"]))
        );
        assert_eq!(
            analysis_schema.pointer(
                "/properties/full_summary/items/properties/evidence_segment_ids/items/$ref"
            ),
            Some(&Value::String("#/$defs/segment_id".into()))
        );
        let translation_schema = translation_response_schema(&valid_ids);
        assert_eq!(
            translation_schema.pointer("/properties/translations/items/properties/segment_id/$ref"),
            Some(&Value::String("#/$defs/segment_id".into()))
        );
        assert_eq!(
            translation_schema.pointer("/properties/translations/minItems"),
            Some(&Value::from(2))
        );
        assert_eq!(
            translation_schema.pointer("/properties/translations/maxItems"),
            Some(&Value::from(2))
        );
        assert_eq!(
            translation_schema
                .pointer("/properties/translations/items/properties/source_excerpt/enum"),
            Some(&json!([""]))
        );
    }

    #[test]
    fn legacy_english_recaps_deserialize_with_explicit_target_language_metadata() {
        let legacy = json!({
            "meeting_title_english": "Planning meeting",
            "dominant_language": "fr",
            "executive_summary": { "original": "Planification", "english": "Planning" },
            "full_summary": [{
                "heading": { "original": "Plan", "english": "Plan" },
                "body": { "original": "Accord", "english": "Agreement" },
                "evidence_segment_ids": ["segment-1"]
            }],
            "commitments": [],
            "actions_already_taken": [],
            "agenda_present": false,
            "agenda_coverage": [],
            "translations": [{
                "segment_id": "segment-1",
                "source_excerpt": "Bonjour",
                "language": "fr",
                "english_translation": "Hello"
            }]
        });
        let payload: RecapPayload = serde_json::from_value(legacy).unwrap();
        assert_eq!(payload.target_language, "en");
        assert_eq!(payload.meeting_title, "Planning meeting");
        assert_eq!(payload.executive_summary.translated, "Planning");
        assert_eq!(payload.translations[0].translated_text, "Hello");
        let current = serde_json::to_value(payload).unwrap();
        assert_eq!(current["target_language"], "en");
        assert!(current.get("meeting_title_english").is_none());
        assert!(current["executive_summary"].get("english").is_none());
    }
}

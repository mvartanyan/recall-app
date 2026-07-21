use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

pub const PROMPT_VERSION: &str = "recall-recap-v3";
pub const SCHEMA_VERSION: &str = "recall-recap-schema-v3";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocalizedText {
    pub original: String,
    pub english: String,
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
    pub english_translation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecapPayload {
    pub meeting_title_english: String,
    pub dominant_language: String,
    pub executive_summary: LocalizedText,
    pub full_summary: Vec<SummarySection>,
    pub commitments: Vec<ActionItem>,
    pub actions_already_taken: Vec<ActionItem>,
    pub agenda_present: bool,
    pub agenda_coverage: Vec<AgendaCoverageItem>,
    pub translations: Vec<TranslationAnnotation>,
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
    let input = FingerprintInput {
        segments,
        agenda,
        no_translation_languages: languages,
    };
    let bytes = serde_json::to_vec(&input)
        .map_err(|error| format!("Could not fingerprint recap sources: {error}"))?;
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
    if payload.meeting_title_english.trim().is_empty() {
        return Err("OpenAI returned an empty meeting title".into());
    }
    if payload.dominant_language.trim().is_empty() {
        return Err("OpenAI returned no dominant meeting language".into());
    }
    validate_localized(&payload.executive_summary, "executive summary")?;
    if payload.full_summary.is_empty() {
        return Err("OpenAI returned no full-summary sections".into());
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
            return Err("OpenAI returned an action without a participant".into());
        }
        validate_localized(&item.statement, "action statement")?;
        validate_required_evidence(&item.evidence_segment_ids, valid_segment_ids, "action item")?;
    }
    if payload.agenda_present != agenda_present {
        return Err("OpenAI returned inconsistent agenda metadata".into());
    }
    if !agenda_present && !payload.agenda_coverage.is_empty() {
        return Err("OpenAI returned agenda coverage when no agenda was supplied".into());
    }
    if agenda_present && payload.agenda_coverage.is_empty() {
        return Err("OpenAI returned no agenda coverage for the supplied agenda".into());
    }
    for item in &payload.agenda_coverage {
        if !matches!(
            item.status.as_str(),
            "covered" | "partial" | "not-covered" | "unreadable"
        ) {
            return Err(format!(
                "OpenAI returned an unsupported agenda status: {}",
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
                "OpenAI translation references an unknown segment: {}",
                translation.segment_id
            ));
        }
        if translation.source_excerpt.trim().is_empty()
            || translation.language.trim().is_empty()
            || translation.english_translation.trim().is_empty()
        {
            return Err("OpenAI returned an incomplete translation annotation".into());
        }
    }
    Ok(())
}

fn validate_localized(value: &LocalizedText, field: &str) -> Result<(), String> {
    if value.original.trim().is_empty() || value.english.trim().is_empty() {
        Err(format!("OpenAI returned an incomplete {field}"))
    } else {
        Ok(())
    }
}

fn validate_evidence(ids: &[String], valid_segment_ids: &HashSet<String>) -> Result<(), String> {
    for id in ids {
        if !valid_segment_ids.contains(id) {
            return Err(format!("OpenAI cited an unknown transcript segment: {id}"));
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
            "OpenAI returned a {field} without transcript evidence"
        ));
    }
    validate_evidence(ids, valid_segment_ids)
}

pub fn response_schema(valid_segment_ids: &[String]) -> Value {
    let localized = || {
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "original": { "type": "string" },
                "english": { "type": "string" }
            },
            "required": ["original", "english"]
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
    let translation = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "segment_id": { "$ref": "#/$defs/segment_id" },
            "source_excerpt": { "type": "string" },
            "language": { "type": "string" },
            "english_translation": { "type": "string" }
        },
        "required": ["segment_id", "source_excerpt", "language", "english_translation"]
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
            "meeting_title_english": { "type": "string" },
            "dominant_language": { "type": "string" },
            "executive_summary": localized(),
            "full_summary": { "type": "array", "items": summary_section },
            "commitments": { "type": "array", "items": action_item.clone() },
            "actions_already_taken": { "type": "array", "items": action_item },
            "agenda_present": { "type": "boolean" },
            "agenda_coverage": { "type": "array", "items": agenda_item },
            "translations": {
                "type": "array",
                "items": translation,
                "minItems": valid_segment_ids.len(),
                "maxItems": valid_segment_ids.len()
            }
        },
        "required": [
            "meeting_title_english",
            "dominant_language",
            "executive_summary",
            "full_summary",
            "commitments",
            "actions_already_taken",
            "agenda_present",
            "agenda_coverage",
            "translations"
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_changes_for_transcript_agenda_or_translation_policy() {
        let segment = RecapSourceSegment {
            id: "seg-1".into(),
            start_ms: 0,
            end_ms: 1_000,
            speaker_id: Some("speaker-1".into()),
            speaker_label: "Alice".into(),
            text: "Bonjour".into(),
        };
        let baseline = source_fingerprint(std::slice::from_ref(&segment), None, &[]).unwrap();
        let with_policy =
            source_fingerprint(std::slice::from_ref(&segment), None, &["fr".into()]).unwrap();
        let with_agenda = source_fingerprint(
            std::slice::from_ref(&segment),
            Some(AgendaFingerprint {
                source_kind: "text",
                filename: "Agenda.txt",
                mime_type: "text/plain",
                content: b"Introductions",
            }),
            &[],
        )
        .unwrap();
        let mut changed = segment;
        changed.text = "Bonsoir".into();
        let with_edit = source_fingerprint(&[changed], None, &[]).unwrap();
        assert_ne!(baseline, with_policy);
        assert_ne!(baseline, with_agenda);
        assert_ne!(baseline, with_edit);
    }

    #[test]
    fn english_is_implicit_and_does_not_change_translation_policy_hash() {
        let segments = Vec::new();
        assert_eq!(
            source_fingerprint(&segments, None, &[]).unwrap(),
            source_fingerprint(&segments, None, &["en".into(), "EN".into()]).unwrap()
        );
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
        inspect(&response_schema(&["segment-1".into()]));
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
        let schema = response_schema(&["segment-1".into()]);
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
        let schema = response_schema(&valid_ids);
        assert_eq!(
            schema.pointer("/$defs/segment_id/enum"),
            Some(&json!(["segment-1", "segment-2"]))
        );
        assert_eq!(
            schema.pointer(
                "/properties/full_summary/items/properties/evidence_segment_ids/items/$ref"
            ),
            Some(&Value::String("#/$defs/segment_id".into()))
        );
        assert_eq!(
            schema.pointer("/properties/translations/items/properties/segment_id/$ref"),
            Some(&Value::String("#/$defs/segment_id".into()))
        );
        assert_eq!(
            schema.pointer("/properties/translations/minItems"),
            Some(&Value::from(2))
        );
        assert_eq!(
            schema.pointer("/properties/translations/maxItems"),
            Some(&Value::from(2))
        );
    }
}

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

pub const JAMIE_IMPORTER_VERSION: &str = "jamie-text-v1";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieExportMetadata {
    pub user: Option<String>,
    pub export_date: Option<DateTime<Utc>>,
    pub declared_total_meetings: Option<usize>,
    pub includes: Vec<String>,
    pub source_sha256: String,
    pub source_size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieWarning {
    pub code: String,
    pub message: String,
    pub blocking: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieSpeakerMapEntry {
    pub source_label: String,
    pub display_label: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieTranscriptSegment {
    pub speaker_label: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieMeeting {
    pub source_fingerprint: String,
    pub title: String,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub duration_ms: i64,
    pub speaker_map: Vec<JamieSpeakerMapEntry>,
    pub executive_summary: String,
    pub full_summary: String,
    pub tasks: String,
    pub segments: Vec<JamieTranscriptSegment>,
    pub warnings: Vec<JamieWarning>,
}

impl JamieMeeting {
    pub fn has_blocking_warnings(&self) -> bool {
        self.warnings.iter().any(|warning| warning.blocking)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieArchive {
    pub metadata: JamieExportMetadata,
    pub meetings: Vec<JamieMeeting>,
    pub warnings: Vec<JamieWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JamieKnownPerson {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JamieIdentityDecision {
    pub alias: String,
    pub action: String,
    pub target_speaker_id: Option<String>,
    pub display_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JamieImportDraft {
    pub id: String,
    pub source_path: String,
    pub source_sha256: String,
    pub importer_version: String,
    pub identity_decisions: Vec<JamieIdentityDecision>,
    pub excluded_meetings: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieIdentityPreview {
    pub alias: String,
    pub generic: bool,
    pub intervention_count: usize,
    pub meeting_count: usize,
    pub excerpts: Vec<String>,
    pub decision: JamieIdentityDecision,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieMeetingPreview {
    pub source_fingerprint: String,
    pub title: String,
    pub started_at: Option<DateTime<Utc>>,
    pub duration_ms: i64,
    pub intervention_count: usize,
    pub speaker_count: usize,
    pub has_executive_summary: bool,
    pub has_full_summary: bool,
    pub has_tasks: bool,
    pub included: bool,
    pub already_imported: bool,
    pub warnings: Vec<JamieWarning>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct JamieImportPreview {
    pub draft: JamieImportDraft,
    pub metadata: JamieExportMetadata,
    pub known_people: Vec<JamieKnownPerson>,
    pub meetings: Vec<JamieMeetingPreview>,
    pub identities: Vec<JamieIdentityPreview>,
    pub archive_warnings: Vec<JamieWarning>,
    pub validation_errors: Vec<String>,
    pub ready_to_import: bool,
    pub included_meeting_count: usize,
    pub existing_meeting_count: usize,
    pub total_intervention_count: usize,
}

pub fn initial_import_draft(
    source_path: &Path,
    archive: &JamieArchive,
    known_people: &[JamieKnownPerson],
) -> JamieImportDraft {
    let michael = known_people
        .iter()
        .filter(|person| normalized_person_name(&person.label) == "michael vartanyan")
        .collect::<Vec<_>>();
    let mut identity_decisions = all_alias_counts(archive)
        .into_keys()
        .map(|alias| {
            if is_generic_speaker_label(&alias) {
                JamieIdentityDecision {
                    alias,
                    action: "unresolved".into(),
                    target_speaker_id: None,
                    display_name: None,
                }
            } else if alias == "Mv" && michael.len() == 1 {
                JamieIdentityDecision {
                    alias,
                    action: "proposed_map".into(),
                    target_speaker_id: Some(michael[0].id.clone()),
                    display_name: Some(michael[0].label.clone()),
                }
            } else {
                JamieIdentityDecision {
                    alias,
                    action: "review".into(),
                    target_speaker_id: None,
                    display_name: None,
                }
            }
        })
        .collect::<Vec<_>>();
    identity_decisions.sort_by(|left, right| {
        left.alias
            .to_lowercase()
            .cmp(&right.alias.to_lowercase())
            .then_with(|| left.alias.cmp(&right.alias))
    });
    JamieImportDraft {
        id: archive.metadata.source_sha256[..16].to_string(),
        source_path: source_path.to_string_lossy().to_string(),
        source_sha256: archive.metadata.source_sha256.clone(),
        importer_version: JAMIE_IMPORTER_VERSION.into(),
        identity_decisions,
        excluded_meetings: Vec::new(),
        updated_at: Utc::now(),
    }
}

pub fn merge_saved_draft(
    mut initial: JamieImportDraft,
    saved: JamieImportDraft,
) -> JamieImportDraft {
    if initial.source_sha256 != saved.source_sha256
        || saved.importer_version != JAMIE_IMPORTER_VERSION
    {
        return initial;
    }
    let saved_decisions = saved
        .identity_decisions
        .into_iter()
        .map(|decision| (decision.alias.clone(), decision))
        .collect::<HashMap<_, _>>();
    for decision in &mut initial.identity_decisions {
        if let Some(saved) = saved_decisions.get(&decision.alias) {
            *decision = saved.clone();
        }
    }
    initial.excluded_meetings = saved.excluded_meetings;
    initial.updated_at = saved.updated_at;
    initial
}

pub fn import_draft_path(data_dir: &Path, source_sha256: &str) -> PathBuf {
    let source_prefix = source_sha256.get(..16).unwrap_or(source_sha256);
    data_dir
        .join("imports")
        .join(format!("jamie-{source_prefix}.json"))
}

pub fn load_import_draft(path: &Path) -> Result<Option<JamieImportDraft>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let content = fs::read(path)
        .map_err(|error| format!("Could not read the Jamie import draft: {error}"))?;
    serde_json::from_slice(&content)
        .map(Some)
        .map_err(|error| format!("Could not decode the Jamie import draft: {error}"))
}

pub fn save_import_draft(path: &Path, draft: &JamieImportDraft) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "Jamie import draft path has no parent".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("Could not create the import-draft directory: {error}"))?;
    let temporary = path.with_extension("json.tmp");
    let content = serde_json::to_vec_pretty(draft)
        .map_err(|error| format!("Could not encode the Jamie import draft: {error}"))?;
    let mut file = File::create(&temporary)
        .map_err(|error| format!("Could not create the Jamie import draft: {error}"))?;
    file.write_all(&content)
        .map_err(|error| format!("Could not save the Jamie import draft: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("Could not flush the Jamie import draft: {error}"))?;
    restrict_file_permissions(&temporary)?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("Could not replace the Jamie import draft: {error}"))?;
    restrict_file_permissions(path)
}

pub fn build_import_preview(
    archive: &JamieArchive,
    draft: &JamieImportDraft,
    known_people: &[JamieKnownPerson],
    existing_meeting_fingerprints: &HashSet<String>,
) -> JamieImportPreview {
    let decision_map = draft
        .identity_decisions
        .iter()
        .map(|decision| (decision.alias.as_str(), decision))
        .collect::<HashMap<_, _>>();
    let excluded = draft
        .excluded_meetings
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut identity_accumulator = HashMap::<String, (usize, HashSet<String>, Vec<String>)>::new();
    let meetings = archive
        .meetings
        .iter()
        .map(|meeting| {
            let mut speakers = HashSet::new();
            for segment in &meeting.segments {
                speakers.insert(segment.speaker_label.clone());
                let entry = identity_accumulator
                    .entry(segment.speaker_label.clone())
                    .or_insert_with(|| (0, HashSet::new(), Vec::new()));
                entry.0 += 1;
                entry.1.insert(meeting.source_fingerprint.clone());
                if entry.2.len() < 3 {
                    entry.2.push(bounded_excerpt(&segment.text, 180));
                }
            }
            JamieMeetingPreview {
                source_fingerprint: meeting.source_fingerprint.clone(),
                title: meeting.title.clone(),
                started_at: meeting.started_at,
                duration_ms: meeting.duration_ms,
                intervention_count: meeting.segments.len(),
                speaker_count: speakers.len(),
                has_executive_summary: !meeting.executive_summary.trim().is_empty(),
                has_full_summary: !meeting.full_summary.trim().is_empty(),
                has_tasks: !meeting.tasks.trim().is_empty(),
                included: !excluded.contains(meeting.source_fingerprint.as_str()),
                already_imported: existing_meeting_fingerprints
                    .contains(&meeting.source_fingerprint),
                warnings: meeting.warnings.clone(),
            }
        })
        .collect::<Vec<_>>();
    let mut identities = identity_accumulator
        .into_iter()
        .map(
            |(alias, (intervention_count, meetings, excerpts))| JamieIdentityPreview {
                generic: is_generic_speaker_label(&alias),
                decision: decision_map
                    .get(alias.as_str())
                    .map(|decision| (*decision).clone())
                    .unwrap_or_else(|| JamieIdentityDecision {
                        alias: alias.clone(),
                        action: "review".into(),
                        target_speaker_id: None,
                        display_name: None,
                    }),
                alias,
                intervention_count,
                meeting_count: meetings.len(),
                excerpts,
            },
        )
        .collect::<Vec<_>>();
    identities.sort_by(|left, right| {
        left.alias
            .to_lowercase()
            .cmp(&right.alias.to_lowercase())
            .then_with(|| left.alias.cmp(&right.alias))
    });
    let validation_errors = validate_import_draft(archive, draft, known_people);
    let included_meeting_count = meetings.iter().filter(|meeting| meeting.included).count();
    let existing_meeting_count = meetings
        .iter()
        .filter(|meeting| existing_meeting_fingerprints.contains(&meeting.source_fingerprint))
        .count();
    let statistics = archive_statistics(archive);
    JamieImportPreview {
        draft: draft.clone(),
        metadata: archive.metadata.clone(),
        known_people: known_people.to_vec(),
        meetings,
        identities,
        archive_warnings: archive.warnings.clone(),
        ready_to_import: validation_errors.is_empty() && included_meeting_count > 0,
        validation_errors,
        included_meeting_count,
        existing_meeting_count,
        total_intervention_count: *statistics.get("segments").unwrap_or(&0),
    }
}

pub fn validate_import_draft(
    archive: &JamieArchive,
    draft: &JamieImportDraft,
    known_people: &[JamieKnownPerson],
) -> Vec<String> {
    let mut errors = archive
        .warnings
        .iter()
        .filter(|warning| warning.blocking)
        .map(|warning| warning.message.clone())
        .collect::<Vec<_>>();
    if draft.source_sha256 != archive.metadata.source_sha256 {
        errors.push("The saved draft belongs to a different source file.".into());
    }
    if draft.importer_version != JAMIE_IMPORTER_VERSION {
        errors.push("The saved draft uses an incompatible importer version.".into());
    }
    let known_ids = known_people
        .iter()
        .map(|person| person.id.as_str())
        .collect::<HashSet<_>>();
    let known_names = known_people
        .iter()
        .map(|person| normalized_person_name(&person.label))
        .collect::<HashSet<_>>();
    let decisions = draft
        .identity_decisions
        .iter()
        .map(|decision| (decision.alias.as_str(), decision))
        .collect::<HashMap<_, _>>();
    let stable_aliases = stable_alias_counts(archive);
    for alias in stable_aliases.keys() {
        if is_generic_speaker_label(alias) {
            continue;
        }
        let Some(decision) = decisions.get(alias.as_str()) else {
            errors.push(format!("{alias}: no identity decision is saved."));
            continue;
        };
        match decision.action.as_str() {
            "map_existing" => {
                if !decision
                    .target_speaker_id
                    .as_deref()
                    .map(|id| known_ids.contains(id))
                    .unwrap_or(false)
                {
                    errors.push(format!("{alias}: choose an existing person again."));
                }
            }
            "create_named" => {
                let name = decision.display_name.as_deref().unwrap_or_default();
                let normalized = normalized_person_name(name);
                if normalized.is_empty() {
                    errors.push(format!("{alias}: enter a name for the new person."));
                } else if known_names.contains(&normalized) {
                    errors.push(format!(
                        "{alias}: {name:?} already exists; map the alias to that person instead."
                    ));
                }
            }
            "unresolved" => {}
            "proposed_map" => {
                errors.push(format!(
                    "{alias}: review and accept or change the proposed person mapping."
                ));
            }
            _ => errors.push(format!(
                "{alias}: choose how this source identity should import."
            )),
        }
    }

    let excluded = draft
        .excluded_meetings
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    for meeting in &archive.meetings {
        if excluded.contains(meeting.source_fingerprint.as_str()) {
            continue;
        }
        if meeting.has_blocking_warnings() {
            for warning in meeting.warnings.iter().filter(|warning| warning.blocking) {
                errors.push(format!("{}: {}", meeting.title, warning.message));
            }
        }
    }
    errors.sort();
    errors.dedup();
    errors
}

fn normalized_person_name(value: &str) -> String {
    value
        .nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn bounded_excerpt(value: &str, max_chars: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let mut result = compact.chars().take(max_chars).collect::<String>();
    result.push('…');
    result
}

#[cfg(unix)]
fn restrict_file_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|error| format!("Could not restrict import-draft permissions: {error}"))
}

#[cfg(not(unix))]
fn restrict_file_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[derive(Default)]
struct HeaderFields {
    user: Option<String>,
    export_date: Option<DateTime<Utc>>,
    declared_total_meetings: Option<usize>,
    includes: Vec<String>,
}

pub fn parse_jamie_export(path: &Path) -> Result<JamieArchive, String> {
    let file = File::open(path).map_err(|error| {
        format!(
            "Could not open the Jamie export {}: {error}",
            path.display()
        )
    })?;
    let mut reader = BufReader::new(file);
    let mut source_hasher = Sha256::new();
    let mut source_size_bytes = 0u64;
    let mut header = HeaderFields::default();
    let mut archive_warnings = Vec::new();
    let mut meetings = Vec::new();
    let mut current_block: Option<Vec<String>> = None;
    let mut line = String::new();

    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|error| format!("Could not read the Jamie export: {error}"))?;
        if bytes == 0 {
            break;
        }
        source_hasher.update(line.as_bytes());
        source_size_bytes += bytes as u64;
        let clean = line.trim_end_matches(['\r', '\n']).to_string();

        if clean.starts_with("MEETING:") {
            if let Some(block) = current_block.take() {
                meetings.push(parse_meeting_block(block, meetings.len() + 1));
            }
            current_block = Some(vec![clean]);
            continue;
        }

        if let Some(block) = current_block.as_mut() {
            block.push(clean);
        } else {
            parse_header_line(&clean, &mut header, &mut archive_warnings);
        }
    }

    if let Some(block) = current_block {
        meetings.push(parse_meeting_block(block, meetings.len() + 1));
    }

    if meetings.is_empty() {
        return Err("The selected file does not contain any Jamie MEETING blocks".into());
    }
    if let Some(declared) = header.declared_total_meetings {
        if declared != meetings.len() {
            archive_warnings.push(JamieWarning {
                code: "meeting_count_mismatch".into(),
                message: format!(
                    "The export header declares {declared} meetings, but {} blocks were parsed.",
                    meetings.len()
                ),
                blocking: true,
            });
        }
    }

    Ok(JamieArchive {
        metadata: JamieExportMetadata {
            user: header.user,
            export_date: header.export_date,
            declared_total_meetings: header.declared_total_meetings,
            includes: header.includes,
            source_sha256: format!("{:x}", source_hasher.finalize()),
            source_size_bytes,
        },
        meetings,
        warnings: archive_warnings,
    })
}

fn parse_header_line(line: &str, header: &mut HeaderFields, warnings: &mut Vec<JamieWarning>) {
    if let Some(value) = line.strip_prefix("User:") {
        header.user = nonempty(value);
    } else if let Some(value) = line.strip_prefix("Export Date:") {
        match parse_rfc3339(value) {
            Ok(parsed) => header.export_date = Some(parsed),
            Err(message) => warnings.push(JamieWarning {
                code: "invalid_export_date".into(),
                message,
                blocking: false,
            }),
        }
    } else if let Some(value) = line.strip_prefix("Total Meetings:") {
        match value.trim().parse::<usize>() {
            Ok(total) => header.declared_total_meetings = Some(total),
            Err(_) => warnings.push(JamieWarning {
                code: "invalid_declared_meeting_count".into(),
                message: format!("Could not parse Total Meetings value {:?}", value.trim()),
                blocking: false,
            }),
        }
    } else if let Some(value) = line.strip_prefix("Includes:") {
        header.includes = value.split(',').filter_map(nonempty).collect::<Vec<_>>();
    }
}

fn parse_meeting_block(lines: Vec<String>, ordinal: usize) -> JamieMeeting {
    let title = lines
        .first()
        .and_then(|line| line.strip_prefix("MEETING:"))
        .and_then(nonempty)
        .unwrap_or_else(|| format!("Imported meeting {ordinal}"));
    let started_at = find_prefixed(&lines, "Date:")
        .map(parse_rfc3339)
        .transpose();
    let ended_at = find_prefixed(&lines, "End:").map(parse_rfc3339).transpose();
    let mut warnings = Vec::new();
    let started_at = match started_at {
        Ok(value) => value,
        Err(message) => {
            warnings.push(JamieWarning {
                code: "invalid_start_date".into(),
                message,
                blocking: true,
            });
            None
        }
    };
    let ended_at = match ended_at {
        Ok(value) => value,
        Err(message) => {
            warnings.push(JamieWarning {
                code: "invalid_end_date".into(),
                message,
                blocking: true,
            });
            None
        }
    };
    if started_at.is_none() {
        warnings.push(JamieWarning {
            code: "missing_start_date".into(),
            message: format!("{title}: missing Date field"),
            blocking: true,
        });
    }

    let sections = section_indices(&lines);
    let speaker_map = sections
        .get("SPEAKERS")
        .map(|start| {
            let end = next_section_index(&sections, *start, lines.len());
            parse_speaker_map(&lines[(start + 1)..end])
        })
        .unwrap_or_default();
    if !sections.contains_key("SPEAKERS") {
        warnings.push(JamieWarning {
            code: "missing_speaker_map".into(),
            message: format!(
                "{title}: no SPEAKERS map; transcript labels will remain meeting-local unless reviewed"
            ),
            blocking: false,
        });
    }

    let summary_lines = section_slice(&lines, &sections, "SUMMARY");
    let (executive_summary, full_summary) = parse_summary(summary_lines);
    if !sections.contains_key("SUMMARY") {
        warnings.push(JamieWarning {
            code: "missing_summary".into(),
            message: format!("{title}: missing SUMMARY section"),
            blocking: false,
        });
    }

    let transcript_lines = section_slice(&lines, &sections, "TRANSCRIPT");
    let (segments, transcript_warnings) = parse_transcript(transcript_lines, &title);
    warnings.extend(transcript_warnings);
    if !sections.contains_key("TRANSCRIPT") {
        warnings.push(JamieWarning {
            code: "missing_transcript".into(),
            message: format!("{title}: missing TRANSCRIPT section"),
            blocking: true,
        });
    } else if segments.is_empty() {
        warnings.push(JamieWarning {
            code: "empty_transcript".into(),
            message: format!("{title}: no timestamped transcript interventions were parsed"),
            blocking: true,
        });
    }

    let tasks = trim_section(section_slice(&lines, &sections, "TASKS")).join("\n");
    if !sections.contains_key("TASKS") {
        warnings.push(JamieWarning {
            code: "missing_tasks".into(),
            message: format!("{title}: missing TASKS section"),
            blocking: false,
        });
    }

    let duration_ms = match (started_at, ended_at) {
        (Some(start), Some(end)) if end >= start => (end - start).num_milliseconds(),
        (Some(_), Some(_)) => {
            warnings.push(JamieWarning {
                code: "end_before_start".into(),
                message: format!("{title}: End precedes Date"),
                blocking: true,
            });
            segments.last().map(|segment| segment.end_ms).unwrap_or(0)
        }
        _ => segments.last().map(|segment| segment.end_ms).unwrap_or(0),
    };
    let source_fingerprint = meeting_fingerprint(
        &title,
        started_at,
        ended_at,
        &speaker_map,
        &executive_summary,
        &full_summary,
        &tasks,
        &segments,
    );

    JamieMeeting {
        source_fingerprint,
        title,
        started_at,
        ended_at,
        duration_ms,
        speaker_map,
        executive_summary,
        full_summary,
        tasks,
        segments,
        warnings,
    }
}

fn section_indices(lines: &[String]) -> BTreeMap<&'static str, usize> {
    let mut sections = BTreeMap::new();
    for (index, line) in lines.iter().enumerate() {
        match line.trim() {
            "SPEAKERS:" => {
                sections.entry("SPEAKERS").or_insert(index);
            }
            "SUMMARY:" => {
                sections.entry("SUMMARY").or_insert(index);
            }
            "TRANSCRIPT:" => {
                sections.entry("TRANSCRIPT").or_insert(index);
            }
            "TASKS:" => {
                sections.entry("TASKS").or_insert(index);
            }
            _ => {}
        }
    }
    sections
}

fn next_section_index(
    sections: &BTreeMap<&'static str, usize>,
    current: usize,
    fallback: usize,
) -> usize {
    sections
        .values()
        .copied()
        .filter(|candidate| *candidate > current)
        .min()
        .unwrap_or(fallback)
}

fn section_slice<'a>(
    lines: &'a [String],
    sections: &BTreeMap<&'static str, usize>,
    name: &str,
) -> &'a [String] {
    let Some(start) = sections.get(name).copied() else {
        return &[];
    };
    let end = next_section_index(sections, start, lines.len());
    &lines[(start + 1)..end]
}

fn parse_speaker_map(lines: &[String]) -> Vec<JamieSpeakerMapEntry> {
    lines
        .iter()
        .filter_map(|line| {
            let entry = line.trim().strip_prefix("- ")?;
            let (source_label, display_label) = entry.split_once(':')?;
            Some(JamieSpeakerMapEntry {
                source_label: source_label.trim().to_string(),
                display_label: display_label.trim().to_string(),
            })
        })
        .filter(|entry| !entry.source_label.is_empty() && !entry.display_label.is_empty())
        .collect()
}

fn parse_summary(lines: &[String]) -> (String, String) {
    let trimmed = trim_section(lines);
    if trimmed.is_empty() {
        return (String::new(), String::new());
    }
    let executive_index = trimmed
        .iter()
        .position(|line| line.trim().eq_ignore_ascii_case("## Executive Summary"));
    let full_index = trimmed
        .iter()
        .position(|line| line.trim().eq_ignore_ascii_case("## Full Summary"));
    match (executive_index, full_index) {
        (Some(executive), Some(full)) if executive < full => (
            trim_section(&trimmed[(executive + 1)..full]).join("\n"),
            trim_section(&trimmed[(full + 1)..]).join("\n"),
        ),
        (None, Some(full)) => (
            String::new(),
            trim_section(&trimmed[(full + 1)..]).join("\n"),
        ),
        (Some(executive), None) => (
            trim_section(&trimmed[(executive + 1)..]).join("\n"),
            String::new(),
        ),
        _ => (String::new(), trimmed.join("\n")),
    }
}

fn parse_transcript(
    lines: &[String],
    title: &str,
) -> (Vec<JamieTranscriptSegment>, Vec<JamieWarning>) {
    let mut timestamps = Vec::new();
    let mut warnings = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if let Some(raw) = line.trim().strip_prefix("###### ") {
            match parse_timestamp_range(raw) {
                Some((start_ms, end_ms)) => timestamps.push((index, start_ms, end_ms)),
                None if raw.contains(" - ") => warnings.push(JamieWarning {
                    code: "malformed_transcript_timestamp".into(),
                    message: format!("{title}: could not parse transcript timestamp {raw:?}"),
                    blocking: true,
                }),
                None => {}
            }
        }
    }

    let mut segments = Vec::new();
    for (position, (timestamp_index, start_ms, end_ms)) in timestamps.iter().copied().enumerate() {
        let speaker_index = previous_nonempty(lines, timestamp_index);
        let Some(speaker_index) = speaker_index else {
            warnings.push(JamieWarning {
                code: "missing_transcript_speaker".into(),
                message: format!(
                    "{title}: transcript intervention at {} has no preceding speaker label",
                    display_timestamp(start_ms)
                ),
                blocking: true,
            });
            continue;
        };
        let text_end = timestamps
            .get(position + 1)
            .and_then(|(next_timestamp, _, _)| previous_nonempty(lines, *next_timestamp))
            .unwrap_or(lines.len());
        let text_start = timestamp_index + 1;
        if text_end < text_start {
            continue;
        }
        let text = trim_section(&lines[text_start..text_end]).join("\n");
        if text.trim().is_empty() {
            warnings.push(JamieWarning {
                code: "empty_transcript_intervention".into(),
                message: format!(
                    "{title}: transcript intervention at {} is empty",
                    display_timestamp(start_ms)
                ),
                blocking: false,
            });
            continue;
        }
        let speaker_label = lines[speaker_index].trim().to_string();
        if speaker_label.is_empty() {
            continue;
        }
        segments.push(JamieTranscriptSegment {
            speaker_label,
            start_ms,
            end_ms: end_ms.max(start_ms),
            text,
        });
    }
    (segments, warnings)
}

fn parse_timestamp_range(value: &str) -> Option<(i64, i64)> {
    let (start, end) = value.split_once(" - ")?;
    Some((parse_timestamp(start)?, parse_timestamp(end)?))
}

fn parse_timestamp(value: &str) -> Option<i64> {
    let parts = value
        .trim()
        .split(':')
        .map(str::parse::<i64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let seconds = match parts.as_slice() {
        [minutes, seconds] if *minutes >= 0 && (0..60).contains(seconds) => minutes * 60 + seconds,
        [hours, minutes, seconds]
            if *hours >= 0 && (0..60).contains(minutes) && (0..60).contains(seconds) =>
        {
            hours * 3_600 + minutes * 60 + seconds
        }
        _ => return None,
    };
    Some(seconds * 1_000)
}

pub fn is_generic_speaker_label(label: &str) -> bool {
    let normalized = label
        .trim()
        .to_lowercase()
        .replace(['_', '-'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if matches!(
        normalized.as_str(),
        "" | "unknown" | "unknown speaker" | "unnamed voice"
    ) {
        return true;
    }
    normalized
        .strip_prefix("speaker ")
        .map(|suffix| !suffix.is_empty() && suffix.chars().all(|value| value.is_ascii_digit()))
        .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
fn meeting_fingerprint(
    title: &str,
    started_at: Option<DateTime<Utc>>,
    ended_at: Option<DateTime<Utc>>,
    speaker_map: &[JamieSpeakerMapEntry],
    executive_summary: &str,
    full_summary: &str,
    tasks: &str,
    segments: &[JamieTranscriptSegment],
) -> String {
    let mut hasher = Sha256::new();
    hash_part(&mut hasher, title.as_bytes());
    hash_part(
        &mut hasher,
        started_at
            .map(|value| value.to_rfc3339())
            .unwrap_or_default()
            .as_bytes(),
    );
    hash_part(
        &mut hasher,
        ended_at
            .map(|value| value.to_rfc3339())
            .unwrap_or_default()
            .as_bytes(),
    );
    for speaker in speaker_map {
        hash_part(&mut hasher, speaker.source_label.as_bytes());
        hash_part(&mut hasher, speaker.display_label.as_bytes());
    }
    hash_part(&mut hasher, executive_summary.as_bytes());
    hash_part(&mut hasher, full_summary.as_bytes());
    hash_part(&mut hasher, tasks.as_bytes());
    for segment in segments {
        hash_part(&mut hasher, segment.speaker_label.as_bytes());
        hash_part(&mut hasher, &segment.start_ms.to_le_bytes());
        hash_part(&mut hasher, &segment.end_ms.to_le_bytes());
        hash_part(&mut hasher, segment.text.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hash_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn find_prefixed<'a>(lines: &'a [String], prefix: &str) -> Option<&'a str> {
    lines
        .iter()
        .find_map(|line| line.strip_prefix(prefix).map(str::trim))
        .filter(|value| !value.is_empty())
}

fn parse_rfc3339(value: &str) -> Result<DateTime<Utc>, String> {
    DateTime::parse_from_rfc3339(value.trim())
        .map(|value| value.with_timezone(&Utc))
        .map_err(|error| format!("Could not parse date {:?}: {error}", value.trim()))
}

fn nonempty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn previous_nonempty(lines: &[String], before: usize) -> Option<usize> {
    (0..before)
        .rev()
        .find(|index| !lines[*index].trim().is_empty())
}

fn trim_section(lines: &[String]) -> Vec<String> {
    let mut start = 0;
    let mut end = lines.len();
    while start < end && (lines[start].trim().is_empty() || is_separator(lines[start].trim())) {
        start += 1;
    }
    while end > start && (lines[end - 1].trim().is_empty() || is_separator(lines[end - 1].trim())) {
        end -= 1;
    }
    lines[start..end].to_vec()
}

fn is_separator(line: &str) -> bool {
    line.len() >= 20 && line.chars().all(|value| matches!(value, '-' | '='))
}

fn display_timestamp(milliseconds: i64) -> String {
    let total_seconds = milliseconds.max(0) / 1_000;
    format!("{:02}:{:02}", total_seconds / 60, total_seconds % 60)
}

pub fn archive_statistics(archive: &JamieArchive) -> BTreeMap<String, usize> {
    let mut values = BTreeMap::new();
    values.insert("meetings".into(), archive.meetings.len());
    values.insert(
        "meetings_without_speaker_map".into(),
        archive
            .meetings
            .iter()
            .filter(|meeting| meeting.speaker_map.is_empty())
            .count(),
    );
    values.insert(
        "blocking_meetings".into(),
        archive
            .meetings
            .iter()
            .filter(|meeting| meeting.has_blocking_warnings())
            .count(),
    );
    values.insert(
        "segments".into(),
        archive
            .meetings
            .iter()
            .map(|meeting| meeting.segments.len())
            .sum(),
    );
    values.insert(
        "meetings_with_executive_summary".into(),
        archive
            .meetings
            .iter()
            .filter(|meeting| !meeting.executive_summary.trim().is_empty())
            .count(),
    );
    values.insert(
        "meetings_with_full_summary".into(),
        archive
            .meetings
            .iter()
            .filter(|meeting| !meeting.full_summary.trim().is_empty())
            .count(),
    );
    values.insert(
        "meetings_with_tasks".into(),
        archive
            .meetings
            .iter()
            .filter(|meeting| !meeting.tasks.trim().is_empty())
            .count(),
    );
    values
}

pub fn stable_alias_counts(archive: &JamieArchive) -> BTreeMap<String, usize> {
    all_alias_counts(archive)
        .into_iter()
        .filter(|(alias, _)| !is_generic_speaker_label(alias))
        .collect()
}

fn all_alias_counts(archive: &JamieArchive) -> BTreeMap<String, usize> {
    let mut counts = HashMap::<String, usize>::new();
    for meeting in &archive.meetings {
        for segment in &meeting.segments {
            *counts.entry(segment.speaker_label.clone()).or_default() += 1;
        }
    }
    counts.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::Write,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    fn temp_export(content: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("recall-jamie-{nonce}.txt"));
        let mut file = File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        path
    }

    const COMPLETE_EXPORT: &str = r#"==========================================================
MEETING EXPORT
==========================================================
User: Test User (test@example.com)
Export Date: 2026-07-23T09:01:12.342Z
Total Meetings: 1
Includes: Summaries, Transcripts, Tasks
==========================================================

----------------------------------------------------------
MEETING: Weekly Привет
----------------------------------------------------------
Date: 2026-07-16T14:00:38.000Z
End: 2026-07-16T15:01:41.000Z

SPEAKERS:
  - Speaker 0: Mv
  - Speaker 1: Anna

SUMMARY:
## Executive Summary

Short summary.

## Full Summary

Long
multiline summary.

TRANSCRIPT:
Mv

###### 00:02 - 00:04

Hello.

Anna

###### 59:59 - 01:00:03

Привет.
Second paragraph.

TASKS:
  [ ] Send notes (Assigned to: Mv)
"#;

    #[test]
    fn parses_complete_multilingual_export_without_mutating_it() {
        let path = temp_export(COMPLETE_EXPORT);
        let archive = parse_jamie_export(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(archive.metadata.declared_total_meetings, Some(1));
        assert_eq!(archive.meetings.len(), 1);
        let meeting = &archive.meetings[0];
        assert_eq!(meeting.title, "Weekly Привет");
        assert_eq!(meeting.duration_ms, 3_663_000);
        assert_eq!(meeting.speaker_map.len(), 2);
        assert_eq!(meeting.executive_summary, "Short summary.");
        assert_eq!(meeting.full_summary, "Long\nmultiline summary.");
        assert_eq!(meeting.segments.len(), 2);
        assert_eq!(meeting.segments[1].start_ms, 3_599_000);
        assert_eq!(meeting.segments[1].end_ms, 3_603_000);
        assert_eq!(meeting.segments[1].text, "Привет.\nSecond paragraph.");
        assert!(meeting.tasks.contains("Send notes"));
        assert!(!meeting.has_blocking_warnings());
    }

    #[test]
    fn missing_speaker_map_is_reviewable_but_does_not_shift_meeting_boundaries() {
        let export = COMPLETE_EXPORT
            .replace("Total Meetings: 1", "Total Meetings: 2")
            .replace("SPEAKERS:\n  - Speaker 0: Mv\n  - Speaker 1: Anna\n\n", "")
            + r#"
----------------------------------------------------------
MEETING: Second
----------------------------------------------------------
Date: 2026-07-17T10:00:00Z
End: 2026-07-17T10:00:05Z
SUMMARY:
## Executive Summary
Second summary.
TRANSCRIPT:
SPEAKER_00

###### 00:00 - 00:05

Second meeting.
TASKS:
"#;
        let path = temp_export(&export);
        let archive = parse_jamie_export(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(archive.meetings.len(), 2);
        assert!(archive.meetings[0]
            .warnings
            .iter()
            .any(|warning| warning.code == "missing_speaker_map" && !warning.blocking));
        assert_eq!(archive.meetings[1].segments[0].text, "Second meeting.");
    }

    #[test]
    fn malformed_timestamp_blocks_only_its_meeting() {
        let export = COMPLETE_EXPORT.replace("###### 00:02 - 00:04", "###### 00:wrong - 00:04");
        let path = temp_export(&export);
        let archive = parse_jamie_export(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert!(archive.meetings[0].has_blocking_warnings());
        assert_eq!(archive.meetings[0].segments.len(), 1);
        assert!(archive.meetings[0]
            .warnings
            .iter()
            .any(|warning| warning.code == "malformed_transcript_timestamp"));
    }

    #[test]
    fn count_mismatch_is_reported_without_dropping_parsed_blocks() {
        let export = COMPLETE_EXPORT.replace("Total Meetings: 1", "Total Meetings: 3");
        let path = temp_export(&export);
        let archive = parse_jamie_export(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(archive.meetings.len(), 1);
        assert!(archive
            .warnings
            .iter()
            .any(|warning| warning.code == "meeting_count_mismatch" && warning.blocking));
    }

    #[test]
    fn generic_labels_are_meeting_local_and_case_distinct_aliases_stay_distinct() {
        assert!(is_generic_speaker_label("SPEAKER_00"));
        assert!(is_generic_speaker_label("Speaker 0"));
        assert!(is_generic_speaker_label("unknown speaker"));
        assert!(!is_generic_speaker_label("Mv"));
        assert!(!is_generic_speaker_label("MV"));

        let path = temp_export(COMPLETE_EXPORT);
        let archive = parse_jamie_export(&path).unwrap();
        fs::remove_file(path).unwrap();
        let counts = stable_alias_counts(&archive);
        assert_eq!(counts.get("Mv"), Some(&1));
        assert_eq!(counts.get("Anna"), Some(&1));
    }

    #[test]
    fn timestamps_support_long_minutes_and_hours() {
        assert_eq!(parse_timestamp("123:45"), Some(7_425_000));
        assert_eq!(parse_timestamp("01:02:03"), Some(3_723_000));
        assert_eq!(parse_timestamp("00:99"), None);
        assert_eq!(parse_timestamp("01:60:00"), None);
    }

    #[test]
    fn exact_mv_gets_a_reviewable_michael_proposal_while_other_aliases_do_not() {
        let export = COMPLETE_EXPORT.replace("Anna\n\n###### 59:59", "MV\n\n###### 59:59");
        let path = temp_export(&export);
        let archive = parse_jamie_export(&path).unwrap();
        let known = vec![JamieKnownPerson {
            id: "michael-id".into(),
            label: "Michael Vartanyan".into(),
        }];
        let draft = initial_import_draft(&path, &archive, &known);
        fs::remove_file(path).unwrap();

        let mv = draft
            .identity_decisions
            .iter()
            .find(|decision| decision.alias == "Mv")
            .unwrap();
        assert_eq!(mv.action, "proposed_map");
        assert_eq!(mv.target_speaker_id.as_deref(), Some("michael-id"));
        let upper = draft
            .identity_decisions
            .iter()
            .find(|decision| decision.alias == "MV")
            .unwrap();
        assert_eq!(upper.action, "review");
    }

    #[test]
    fn validation_requires_identity_review_and_routes_name_collisions_to_mapping() {
        let path = temp_export(COMPLETE_EXPORT);
        let archive = parse_jamie_export(&path).unwrap();
        let known = vec![JamieKnownPerson {
            id: "michael-id".into(),
            label: "Michael Vartanyan".into(),
        }];
        let mut draft = initial_import_draft(&path, &archive, &known);
        let initial_errors = validate_import_draft(&archive, &draft, &known);
        assert!(initial_errors
            .iter()
            .any(|error| error.contains("review and accept")));
        assert!(initial_errors.iter().any(|error| error.contains("Anna")));

        for decision in &mut draft.identity_decisions {
            match decision.alias.as_str() {
                "Mv" => decision.action = "map_existing".into(),
                "Anna" => {
                    decision.action = "create_named".into();
                    decision.display_name = Some("  michael   vartanyan ".into());
                }
                _ => {}
            }
        }
        let collision = validate_import_draft(&archive, &draft, &known);
        assert!(collision
            .iter()
            .any(|error| error.contains("already exists")));
        draft
            .identity_decisions
            .iter_mut()
            .find(|decision| decision.alias == "Anna")
            .unwrap()
            .display_name = Some("Anna Smith".into());
        assert!(validate_import_draft(&archive, &draft, &known).is_empty());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn draft_round_trip_keeps_decisions_without_meeting_content() {
        let source = temp_export(COMPLETE_EXPORT);
        let archive = parse_jamie_export(&source).unwrap();
        let draft = initial_import_draft(&source, &archive, &[]);
        let draft_path = std::env::temp_dir().join(format!(
            "recall-jamie-draft-{}.json",
            &archive.metadata.source_sha256[..16]
        ));

        save_import_draft(&draft_path, &draft).unwrap();
        let saved = load_import_draft(&draft_path).unwrap().unwrap();

        assert_eq!(saved, draft);
        let persisted = fs::read_to_string(&draft_path).unwrap();
        assert!(!persisted.contains("Short summary."));
        assert!(!persisted.contains("Hello."));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&draft_path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        fs::remove_file(source).unwrap();
        fs::remove_file(draft_path).unwrap();
    }
}

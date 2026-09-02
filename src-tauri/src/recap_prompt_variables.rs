use chrono::{DateTime, FixedOffset, Local, Utc};
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RecapPromptVariableDefinition {
    pub token: String,
    pub label: String,
    pub description: String,
    pub example: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecapPromptVariableContext {
    meeting_datetime: DateTime<FixedOffset>,
}

impl RecapPromptVariableContext {
    pub fn from_desktop_local(meeting_created_at: DateTime<Utc>) -> Self {
        let local_datetime = meeting_created_at.with_timezone(&Local);
        Self {
            meeting_datetime: local_datetime.fixed_offset(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_fixed_offset(
        meeting_created_at: DateTime<Utc>,
        offset: FixedOffset,
    ) -> Self {
        Self {
            meeting_datetime: meeting_created_at.with_timezone(&offset),
        }
    }
}

type VariableResolver = fn(&RecapPromptVariableContext) -> String;

struct RecapPromptVariableSpec {
    token: &'static str,
    label: &'static str,
    description: &'static str,
    example: &'static str,
    resolve: VariableResolver,
}

const RECAP_PROMPT_VARIABLES: &[RecapPromptVariableSpec] = &[
    RecapPromptVariableSpec {
        token: "{{meeting_date}}",
        label: "Meeting date",
        description: "Meeting date from Recall's persisted conversation timestamp in the desktop's local timezone, formatted as YYYY/MM/DD.",
        example: "2026/09/01",
        resolve: resolve_meeting_date,
    },
    RecapPromptVariableSpec {
        token: "{{meeting_time}}",
        label: "Meeting time",
        description: "Meeting time from Recall's persisted conversation timestamp in the desktop's local timezone, formatted as HH:mm.",
        example: "09:30",
        resolve: resolve_meeting_time,
    },
    RecapPromptVariableSpec {
        token: "{{meeting_datetime}}",
        label: "Meeting date and time",
        description: "Meeting date and time from Recall's persisted conversation timestamp in the desktop's local timezone, including its UTC offset.",
        example: "2026/09/01 09:30 UTC+02:00",
        resolve: resolve_meeting_datetime,
    },
];

pub fn recap_prompt_variable_definitions() -> Vec<RecapPromptVariableDefinition> {
    RECAP_PROMPT_VARIABLES
        .iter()
        .map(|variable| RecapPromptVariableDefinition {
            token: variable.token.to_string(),
            label: variable.label.to_string(),
            description: variable.description.to_string(),
            example: variable.example.to_string(),
        })
        .collect()
}

pub fn expand_recap_prompt(template: &str, context: &RecapPromptVariableContext) -> String {
    RECAP_PROMPT_VARIABLES
        .iter()
        .fold(template.to_string(), |expanded, variable| {
            expanded.replace(variable.token, &(variable.resolve)(context))
        })
}

fn resolve_meeting_date(context: &RecapPromptVariableContext) -> String {
    context.meeting_datetime.format("%Y/%m/%d").to_string()
}

fn resolve_meeting_time(context: &RecapPromptVariableContext) -> String {
    context.meeting_datetime.format("%H:%M").to_string()
}

fn resolve_meeting_datetime(context: &RecapPromptVariableContext) -> String {
    format!(
        "{} UTC{}",
        context.meeting_datetime.format("%Y/%m/%d %H:%M"),
        context.meeting_datetime.format("%:z")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(value: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(value)
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn registry_metadata_and_expansion_share_the_same_tokens() {
        let definitions = recap_prompt_variable_definitions();
        assert_eq!(
            definitions
                .iter()
                .map(|definition| definition.token.as_str())
                .collect::<Vec<_>>(),
            vec![
                "{{meeting_date}}",
                "{{meeting_time}}",
                "{{meeting_datetime}}"
            ]
        );
        assert!(definitions.iter().all(|definition| {
            !definition.label.is_empty()
                && !definition.description.is_empty()
                && !definition.example.is_empty()
        }));

        let context = RecapPromptVariableContext::from_fixed_offset(
            utc("2026-09-01T07:30:45Z"),
            FixedOffset::east_opt(2 * 60 * 60).unwrap(),
        );
        let template = RECAP_PROMPT_VARIABLES
            .iter()
            .map(|variable| variable.token)
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            expand_recap_prompt(&template, &context),
            "2026/09/01 09:30 2026/09/01 09:30 UTC+02:00"
        );
    }

    #[test]
    fn fixed_offset_formatting_handles_date_rollbacks_and_negative_offsets() {
        let context = RecapPromptVariableContext::from_fixed_offset(
            utc("2026-09-01T02:30:00Z"),
            FixedOffset::west_opt(7 * 60 * 60 + 30 * 60).unwrap(),
        );

        assert_eq!(
            expand_recap_prompt(
                "{{meeting_date}}|{{meeting_time}}|{{meeting_datetime}}",
                &context
            ),
            "2026/08/31|19:00|2026/08/31 19:00 UTC-07:30"
        );
    }

    #[test]
    fn unknown_tokens_remain_literal() {
        let context = RecapPromptVariableContext::from_fixed_offset(
            utc("2026-09-01T07:30:00Z"),
            FixedOffset::east_opt(0).unwrap(),
        );

        assert_eq!(
            expand_recap_prompt(
                "Date {{meeting_date}}, owner {{meeting_owner}}, malformed {{meeting_date }}, again {{meeting_date}}.",
                &context
            ),
            "Date 2026/09/01, owner {{meeting_owner}}, malformed {{meeting_date }}, again 2026/09/01."
        );
    }
}

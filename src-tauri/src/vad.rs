use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

use crate::embedding;

pub const VAD_SAMPLE_RATE: u32 = 16_000;
const VAD_WINDOW_SIZE: usize = 512;
const VAD_BUFFER_SECONDS: f32 = 600.0;
const MERGE_GAP_MS: u64 = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechInterval {
    pub start_ms: u64,
    pub end_ms: u64,
}

impl SpeechInterval {
    pub fn duration_ms(self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    pub fn intersection(self, start_ms: u64, end_ms: u64) -> Option<Self> {
        let start_ms = self.start_ms.max(start_ms);
        let end_ms = self.end_ms.min(end_ms);
        (end_ms > start_ms).then_some(Self { start_ms, end_ms })
    }
}

pub struct VadDetector {
    detector: VoiceActivityDetector,
}

impl VadDetector {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model_path.into()),
                threshold: 0.25,
                min_silence_duration: 0.5,
                min_speech_duration: 0.5,
                window_size: VAD_WINDOW_SIZE as i32,
                max_speech_duration: 10.0,
            },
            sample_rate: VAD_SAMPLE_RATE as i32,
            num_threads: 1,
            provider: Some("cpu".into()),
            debug: false,
            ..VadModelConfig::default()
        };
        let detector = VoiceActivityDetector::create(&config, VAD_BUFFER_SECONDS)
            .ok_or_else(|| "Failed to initialize the local Silero VAD model".to_string())?;
        Ok(Self { detector })
    }

    pub fn speech_intervals(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<SpeechInterval>, String> {
        if pcm.is_empty() || sample_rate == 0 {
            return Err("Audio is empty".into());
        }
        let samples = embedding::resample_linear(pcm, sample_rate, VAD_SAMPLE_RATE);
        self.detector.reset();
        self.detector.clear();

        let mut intervals = Vec::new();
        for chunk in samples.chunks(VAD_WINDOW_SIZE) {
            self.detector.accept_waveform(chunk);
            drain_intervals(&self.detector, &mut intervals);
        }
        self.detector.flush();
        drain_intervals(&self.detector, &mut intervals);
        self.detector.reset();
        self.detector.clear();
        Ok(merge_intervals(intervals))
    }
}

fn drain_intervals(detector: &VoiceActivityDetector, intervals: &mut Vec<SpeechInterval>) {
    while let Some(segment) = detector.front() {
        let start_samples = segment.start().max(0) as u64;
        let sample_count = segment.n().max(0) as u64;
        let start_ms = start_samples.saturating_mul(1_000) / VAD_SAMPLE_RATE as u64;
        let end_ms = start_samples
            .saturating_add(sample_count)
            .saturating_mul(1_000)
            / VAD_SAMPLE_RATE as u64;
        if end_ms > start_ms {
            intervals.push(SpeechInterval { start_ms, end_ms });
        }
        drop(segment);
        detector.pop();
    }
}

fn merge_intervals(mut intervals: Vec<SpeechInterval>) -> Vec<SpeechInterval> {
    intervals.sort_by_key(|interval| interval.start_ms);
    let mut merged: Vec<SpeechInterval> = Vec::new();
    for interval in intervals {
        if let Some(previous) = merged.last_mut() {
            if interval.start_ms <= previous.end_ms.saturating_add(MERGE_GAP_MS) {
                previous.end_ms = previous.end_ms.max(interval.end_ms);
                continue;
            }
        }
        merged.push(interval);
    }
    merged
}

pub fn intersections(
    intervals: &[SpeechInterval],
    start_ms: u64,
    end_ms: u64,
) -> Vec<SpeechInterval> {
    intervals
        .iter()
        .filter_map(|interval| interval.intersection(start_ms, end_ms))
        .collect()
}

pub fn total_duration(intervals: &[SpeechInterval]) -> u64 {
    intervals
        .iter()
        .map(|interval| interval.duration_ms())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn intersections_preserve_timing_and_total_speech() {
        let intervals = vec![
            SpeechInterval {
                start_ms: 500,
                end_ms: 1_500,
            },
            SpeechInterval {
                start_ms: 2_000,
                end_ms: 4_000,
            },
        ];
        let clipped = intersections(&intervals, 1_000, 3_000);
        assert_eq!(
            clipped,
            vec![
                SpeechInterval {
                    start_ms: 1_000,
                    end_ms: 1_500,
                },
                SpeechInterval {
                    start_ms: 2_000,
                    end_ms: 3_000,
                },
            ]
        );
        assert_eq!(total_duration(&clipped), 1_500);
    }

    #[test]
    fn packaged_vad_rejects_digital_silence() {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/silero_vad.onnx");
        assert!(model.is_file(), "packaged VAD model is missing");
        let detector = VadDetector::new(model.to_string_lossy().as_ref()).unwrap();
        let intervals = detector
            .speech_intervals(&vec![0.0; VAD_SAMPLE_RATE as usize * 2], VAD_SAMPLE_RATE)
            .unwrap();
        assert!(intervals.is_empty());
    }

    #[test]
    fn packaged_vad_rejects_keyboard_like_impulses() {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models/silero_vad.onnx");
        let detector = VadDetector::new(model.to_string_lossy().as_ref()).unwrap();
        let mut samples = vec![0.0; VAD_SAMPLE_RATE as usize * 4];
        for pulse_start in (2_000..samples.len()).step_by(6_400) {
            for (index, sample) in samples.iter_mut().skip(pulse_start).take(80).enumerate() {
                *sample = if index % 2 == 0 { 0.9 } else { -0.9 };
            }
        }
        let intervals = detector
            .speech_intervals(&samples, VAD_SAMPLE_RATE)
            .unwrap();
        assert!(
            intervals.is_empty(),
            "keyboard-like impulses must not become voice samples: {intervals:?}"
        );
    }
}

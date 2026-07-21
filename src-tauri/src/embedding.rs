use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

pub const EMBEDDING_VERSION: &str = "wespeaker-ecapa512-lm-v2";
const MODEL_SAMPLE_RATE: u32 = 16_000;

pub struct Embedder {
    extractor: SpeakerEmbeddingExtractor,
}

impl Embedder {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model_path.into()),
            num_threads: 2,
            debug: false,
            provider: Some("cpu".into()),
        };
        let extractor = SpeakerEmbeddingExtractor::create(&config)
            .ok_or_else(|| "Failed to initialize the local speaker model".to_string())?;
        Ok(Self { extractor })
    }

    pub fn embed(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>, String> {
        if pcm.is_empty() || sample_rate == 0 {
            return Err("Speaker sample is empty".into());
        }
        let samples = resample_linear(pcm, sample_rate, MODEL_SAMPLE_RATE);
        let stream = self
            .extractor
            .create_stream()
            .ok_or_else(|| "Failed to create speaker embedding stream".to_string())?;
        stream.accept_waveform(MODEL_SAMPLE_RATE as i32, &samples);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return Err("Speaker sample is too short for fingerprinting".into());
        }
        self.extractor
            .compute(&stream)
            .ok_or_else(|| "Speaker fingerprint computation failed".to_string())
    }
}

fn resample_linear(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return input.to_vec();
    }
    let output_len = ((input.len() as u64 * target_rate as u64) / source_rate as u64) as usize;
    let mut output = Vec::with_capacity(output_len);
    let ratio = source_rate as f64 / target_rate as f64;
    for index in 0..output_len {
        let source_position = index as f64 * ratio;
        let left = source_position.floor() as usize;
        let right = (left + 1).min(input.len() - 1);
        let fraction = (source_position - left as f64) as f32;
        output.push(input[left] * (1.0 - fraction) + input[right] * fraction);
    }
    output
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_changes_length_and_preserves_endpoints() {
        let input = vec![0.0, 0.5, 1.0, 0.5];
        let output = resample_linear(&input, 4, 8);
        assert_eq!(output.len(), 8);
        assert!((output[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_rejects_mismatched_vectors() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }
}

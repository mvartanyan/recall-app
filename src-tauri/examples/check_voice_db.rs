use std::io::Cursor;

use base64::{engine::general_purpose, Engine as _};
use rusqlite::Connection;
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

const MODEL_SAMPLE_RATE: u32 = 16_000;

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
    dot / (left_norm * right_norm)
}

fn resample_linear(input: &[f32], source_rate: u32) -> Vec<f32> {
    if source_rate == MODEL_SAMPLE_RATE {
        return input.to_vec();
    }
    let output_len =
        ((input.len() as u64 * MODEL_SAMPLE_RATE as u64) / source_rate as u64) as usize;
    let ratio = source_rate as f64 / MODEL_SAMPLE_RATE as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * ratio;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let fraction = (position - left as f64) as f32;
            input[left] * (1.0 - fraction) + input[right] * fraction
        })
        .collect()
}

fn decode_wav(encoded: &str) -> Result<(Vec<f32>, u32), String> {
    let bytes = general_purpose::STANDARD
        .decode(encoded)
        .map_err(|error| error.to_string())?;
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err("Only 16-bit PCM preview samples are supported".into());
    }
    let channels = spec.channels.max(1) as usize;
    let interleaved = reader
        .samples::<i16>()
        .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or_else(|| "model path is required".to_string())?;
    let db_path = args
        .next()
        .ok_or_else(|| "database path is required".to_string())?;
    let config = SpeakerEmbeddingExtractorConfig {
        model: Some(model),
        num_threads: 2,
        debug: false,
        provider: Some("cpu".into()),
    };
    let extractor = SpeakerEmbeddingExtractor::create(&config)
        .ok_or_else(|| "Could not create speaker extractor".to_string())?;
    let connection = Connection::open(db_path).map_err(|error| error.to_string())?;
    let mut statement = connection
        .prepare(
            "SELECT s.label, sm.sample_b64
               FROM speaker_samples sm
               JOIN speakers s ON s.id = sm.speaker_id
              WHERE s.label GLOB 'VOICE[0-9]*'
              ORDER BY sm.created_at",
        )
        .map_err(|error| error.to_string())?;
    let samples = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;

    let mut embeddings = Vec::new();
    for (label, encoded) in samples {
        let (pcm, sample_rate) = decode_wav(&encoded)?;
        let pcm = resample_linear(&pcm, sample_rate);
        let stream = extractor
            .create_stream()
            .ok_or_else(|| format!("Could not create stream for {label}"))?;
        stream.accept_waveform(MODEL_SAMPLE_RATE as i32, &pcm);
        stream.input_finished();
        if !extractor.is_ready(&stream) {
            return Err(format!("Sample for {label} is too short"));
        }
        let embedding = extractor
            .compute(&stream)
            .ok_or_else(|| format!("Could not compute embedding for {label}"))?;
        embeddings.push((label, embedding));
    }

    println!("embedding_dim={}", extractor.dim());
    print!("speaker");
    for (label, _) in &embeddings {
        print!("\t{label}");
    }
    println!();
    for (left_label, left) in &embeddings {
        print!("{left_label}");
        for (_, right) in &embeddings {
            print!("\t{:.4}", cosine(left, right));
        }
        println!();
    }
    Ok(())
}

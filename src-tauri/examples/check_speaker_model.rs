use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig, Wave};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let aa: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let bb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (aa * bb)
}

fn main() {
    let model = std::env::args().nth(1).expect("model path");
    let wavs: Vec<String> = std::env::args().skip(2).collect();
    assert!(!wavs.is_empty(), "at least one wav path");
    let config = SpeakerEmbeddingExtractorConfig {
        model: Some(model.into()),
        num_threads: 2,
        debug: false,
        provider: Some("cpu".into()),
    };
    let extractor = SpeakerEmbeddingExtractor::create(&config).expect("create extractor");
    let mut embeddings = Vec::new();
    for wav in &wavs {
        let wave = Wave::read(wav).expect("read wav");
        let stream = extractor.create_stream().expect("create stream");
        stream.accept_waveform(wave.sample_rate(), wave.samples());
        stream.input_finished();
        assert!(extractor.is_ready(&stream), "audio too short: {wav}");
        embeddings.push(extractor.compute(&stream).expect("compute embedding"));
    }
    println!("embedding_dim={}", extractor.dim());
    for left in 0..embeddings.len() {
        for right in left..embeddings.len() {
            println!(
                "similarity[{left},{right}]={:.4}",
                cosine(&embeddings[left], &embeddings[right])
            );
        }
    }
}

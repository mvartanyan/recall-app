use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

pub struct Embedder {
    session: Session,
}

impl Embedder {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .commit_from_file(model_path)
            .map_err(|e| e.to_string())?;
        Ok(Self { session })
    }

    pub fn embed(&mut self, pcm: &[f32]) -> Result<Vec<f32>, String> {
        // Produce simple framed features of shape [1, frames, 80] by chunking PCM into 80-sample frames.
        let feat_dim: usize = 80;
        let mut frames: Vec<f32> = Vec::new();
        let mut idx = 0;
        while idx < pcm.len() {
            let end = std::cmp::min(idx + feat_dim, pcm.len());
            let mut frame = Vec::with_capacity(feat_dim);
            frame.extend_from_slice(&pcm[idx..end]);
            if frame.len() < feat_dim {
                frame.resize(feat_dim, 0.0);
            }
            frames.extend_from_slice(&frame);
            idx += feat_dim;
        }
        if frames.is_empty() {
            frames.resize(feat_dim, 0.0);
        }
        let num_frames = (frames.len() / feat_dim) as i64;

        let input = Tensor::from_array(([1i64, num_frames, feat_dim as i64], frames))
            .map_err(|e| format!("tensor error: {e}"))?;
        let outputs = self
            .session
            .run(ort::inputs![input])
            .map_err(|e| format!("ort run error: {e}"))?;
        let output = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract error: {e}"))?;
        Ok(output.iter().cloned().collect())
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

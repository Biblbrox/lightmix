#[cfg(feature = "viz-rerun")]
use burn::tensor::backend::Backend;
#[cfg(feature = "viz-rerun")]
use rerun::{Color, Position3D};

#[cfg(feature = "viz-rerun")]
use crate::data::batch::Batch;

#[cfg(feature = "viz-rerun")]
pub fn class_color(idx: usize) -> [u8; 3] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    idx.hash(&mut hasher);
    let hash = hasher.finish();

    // HSV -> RGB conversion for a nice spread
    let hue = (hash as f64) / (u64::MAX as f64) * 360.0;
    hsv_to_rgb(hue, 0.85, 0.9)
}

#[cfg(feature = "viz-rerun")]
pub fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h {
        0.0..=60.0 => (c, x, 0.0),
        60.0..=120.0 => (x, c, 0.0),
        120.0..=180.0 => (0.0, c, x),
        180.0..=240.0 => (0.0, x, c),
        240.0..=300.0 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

#[cfg(feature = "viz-rerun")]
pub fn log_3d_sample<B: Backend>(
    output: &burn::train::ClassificationOutput<B>,
    batch: &Batch<B>,
    epoch: usize,
    iteration: i64,
    rec: &rerun::RecordingStream,
) {
    use rand::{RngExt, rng};
    use rerun::Points3D;

    let mut rng = rng();

    rec.set_time_sequence("epoch", epoch as i64);
    rec.set_time_sequence("batch", iteration);

    // batch.data is flat [batch_size * 1024 * 3] due to batcher flatten::<1>(0, -1)
    const N: usize = 1024;
    const C: usize = 3;

    // Pick random sample from batch
    let output = output.output.clone();
    let logits_shape: [usize; 1] = output.shape().dims();
    let batch_size = logits_shape[0];
    let sample_idx = rng.random_range(0..batch_size);
    let output = output.slice_dim(0, sample_idx..sample_idx + 1);

    // Convert tensor to bytes and reinterpret as f32
    let data_tensor = batch.data.clone().to_data();
    let sample_points: &[f32] = data_tensor.as_slice().unwrap();

    let logits_total: usize = logits_shape.iter().product();

    let data = output.clone().to_data();
    let logits_f32: &[f32] = data.as_slice().unwrap();

    // Labels from batch (already flattened [batch_size])
    let data = batch.targets.clone().to_data();
    let labels: &[i32] = data.as_slice().unwrap();

    if labels.is_empty() {
        return;
    }

    let coords: Vec<Position3D> = (0..N)
        .map(|i| {
            Position3D::new(
                sample_points[i * C],
                sample_points[i * C + 1],
                sample_points[i * C + 2],
            )
        })
        .collect();

    // Find argmax for this sample's logits
    let num_classes = logits_total / batch_size;
    let sample_logits = &logits_f32[0..num_classes];
    let pred_idx = sample_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let label_idx = labels[sample_idx] as usize;
    let correct = pred_idx == label_idx;

    // Ground truth color (class-based palette)
    let gt_rgb = class_color(label_idx);
    let gt_colors = vec![Color::from_rgb(gt_rgb[0], gt_rgb[1], gt_rgb[2]); N];

    // Prediction color: green if correct, red if wrong
    let pred_rgb = if correct { [0, 255, 0] } else { [255, 0, 0] };
    let pred_colors = vec![Color::from_rgb(pred_rgb[0], pred_rgb[1], pred_rgb[2]); N];

    // Log ground truth (class-colored) and prediction (correct/wrong)
    rec.log(
        "ground_truth",
        &Points3D::new(coords.clone()).with_colors(gt_colors),
    )
    .ok();

    rec.log(
        "prediction",
        &Points3D::new(coords).with_colors(pred_colors),
    )
    .ok();
}

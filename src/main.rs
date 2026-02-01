#![allow(clippy::manual_retain)]

use std::{env, time::Instant};

use ndarray::Array;
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};
use std::fs::File;
use std::io::{BufReader, Read};

#[rustfmt::skip]
pub const CIFAR100_CLASS_LABELS: [&str; 100] = [
"beaver", "dolphin", "otter", "seal", "whale",
"aquarium_fish", "flatfish", "ray", "shark", "trout",
"orchid", "poppy", "rose", "sunflower", "tulip",
"bottle", "bowl", "can", "cup", "plate",
"apple", "mushroom", "orange", "pear", "sweet_pepper",
"clock", "keyboard", "lamp", "telephone", "television",
"bed", "chair", "couch", "table", "wardrobe",
"bee", "beetle", "butterfly", "caterpillar", "cockroach",
"bear", "leopard", "lion", "tiger", "wolf",
"bridge", "castle", "house", "road", "skyscraper",
"cloud", "forest", "mountain", "plain", "sea",
"camel", "cattle", "chimpanzee", "elephant", "kangaroo",
"fox", "porcupine", "possum", "raccoon", "skunk",
"crab", "lobster", "snail", "spider", "worm",
"baby", "boy", "girl", "man", "woman",
"crocodile", "dinosaur", "lizard", "snake", "turtle",
"hamster", "mouse", "rabbit", "shrew", "squirrel",
"maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree",
"bicycle", "bus", "motorcycle", "pickup_truck", "train",
"lawn_mower", "rocket", "streetcar", "tank", "tractor",
];

const CIFAR_IMAGE_SIZE: usize = 32 * 32 * 3;
const CIFAR_RECORD_SIZE: usize = CIFAR_IMAGE_SIZE + 2;

fn load_cifar100_test(path: &str) -> std::io::Result<Vec<(Vec<f32>, u8)>> {
    let mut f = BufReader::new(File::open(path)?);
    let mut dataset = Vec::new();

    loop {
        let mut buf = [0u8; CIFAR_RECORD_SIZE];
        if f.read_exact(&mut buf).is_err() {
            break;
        }

        let label = buf[1]; // fine label

        let mut img = vec![0f32; CIFAR_IMAGE_SIZE];

        // Normalize to [0,1]
        for i in 0..1024 {
            img[i] = buf[2 + i] as f32 / 255.0; // R
            img[1024 + i] = buf[1026 + i] as f32 / 255.0; // G
            img[2048 + i] = buf[2050 + i] as f32 / 255.0; // B
        }

        dataset.push((img, label));
    }

    Ok(dataset)
}

fn main() -> ort::Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = &args[1];
    let mut session = Session::builder()?.commit_from_file(model_path)?;

    let cifar_path = &args[2]; // path to test.bin
    let dataset = load_cifar100_test(cifar_path).unwrap();

    let mut correct = 0usize;
    let mut total_latency = 0f64;

    let mut idx = 0;
    for (img, label) in dataset.iter() {
        let input = Array::from_shape_vec((1, 3, 32, 32), img.clone()).unwrap();

        let start = Instant::now();

        let outputs: SessionOutputs =
            session.run(inputs!["input" => TensorRef::from_array_view(&input)?])?;

        let elapsed = start.elapsed().as_secs_f64();
        total_latency += elapsed;

        let logits: Vec<f32> = outputs["output"]
            .try_extract_array::<f32>()?
            .iter()
            .copied()
            .collect();

        let pred = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u8;

        if pred == *label {
            correct += 1;
        }

        idx += 1;
        if idx > 500 {
            break;
        }
    }

    let accuracy = correct as f32 / dataset.len() as f32 * 100.0;
    let mean_latency_ms = (total_latency / dataset.len() as f64) * 1000.0;

    println!("Accuracy: {:.2}%", accuracy);
    println!("Mean latency: {:.3} ms/image", mean_latency_ms);

    Ok(())
}

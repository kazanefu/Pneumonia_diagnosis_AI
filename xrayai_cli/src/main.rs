use clap::Parser;
use image::imageops::FilterType;
use onnxruntime::{LoggingLevel, environment::Environment, session::Session};
use std::env;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Path to input image
    image_path: PathBuf,
}

fn model_path() -> PathBuf {
    let exe_path = env::current_exe()
        .expect("Failed to get exe path");

    let exe_dir = exe_path
        .parent()
        .expect("Failed to get exe directory");

    exe_dir.join("model").join("model.onnx")
}

fn preprocess_image(path: &PathBuf) -> onnxruntime::ndarray::Array4<f32> {
    let img = image::open(path).expect("Failed to open image").to_luma8();

    let img = image::imageops::resize(&img, 224, 224, FilterType::Triangle);

    let mut input = onnxruntime::ndarray::Array4::<f32>::zeros((1, 1, 224, 224));

    for y in 0..224 {
        for x in 0..224 {
            let pixel = img.get_pixel(x, y)[0] as f32 / 255.0;
            input[[0, 0, y as usize, x as usize]] = pixel;
        }
    }

    input
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();

    exp.iter().map(|x| x / sum).collect()
}

fn main() {
    let args = Args::parse();
    let model_path = model_path();

    // =========================
    // ONNX Runtime 初期化
    // =========================
    let environment = Environment::builder()
        .with_name("xray")
        .with_log_level(LoggingLevel::Warning)
        .build()
        .expect("Failed to create environment");

    let mut session: Session = environment
        .new_session_builder()
        .expect("Failed to create session builder")
        .with_model_from_file(&model_path)
        .expect("Failed to load model");

    // 前処理
    let input_tensor = preprocess_image(&args.image_path);

    // 推論
    let outputs = session.run(vec![input_tensor]).expect("Inference failed");

    let output = &outputs[0];
    let logits_view = output.view();

    // logits_view は ndarrayなのでVecに変換
    let logits: Vec<f32> = logits_view.iter().cloned().collect();

    // softmaxで確率へ
    let probs = softmax(&logits);
    let abnormal_prob = probs[1];

    // 結果表示
    println!("Abnormal probability: {:.2}", abnormal_prob);

    if abnormal_prob > 0.5 {
        println!("Possible diseases:");
        println!(" - Pneumonia");
    } else {
        println!("Result: Normal");
    }
}

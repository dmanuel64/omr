use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::collections::HashMap;
use serde_json::Value as JsonValue;
use std::fs;

fn read_json_map_string_string(path: &str) -> Result<HashMap<String, String>> {
    let s = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&s)?)
}

fn read_json_map_string_usize(path: &str) -> Result<HashMap<String, usize>> {
    let s = fs::read_to_string(path)?;
    let v: HashMap<String, JsonValue> = serde_json::from_str(&s)?;
    let mut out = HashMap::new();
    for (k, vv) in v {
        let n = vv
            .as_u64()
            .ok_or_else(|| anyhow!("w2i value for {k} not an integer"))?;
        out.insert(k, n as usize);
    }
    Ok(out)
}

fn main() {
    let mut model = Session::builder()
        .expect("builder to be created")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("optimization level to succeed")
        .with_intra_threads(4)
        .expect("4 intra threads")
        .commit_from_file("C:\\Users\\dylan\\Repositories\\omr\\models\\smt\\smt_encoder.onnx")
        .expect("model to be loaded");
    let i2w = read_json_map_string_string(
        "C:\\Users\\dylan\\Repositories\\omr\\models\\smt\\smt_decoder_step.onnx",
    )
    .expect("vocab to be loaded");
    let w2i = read_json_map_string_usize(
        "C:\\Users\\dylan\\Repositories\\omr\\models\\smt\\smt_decoder_step.json",
    )
    .expect("vocab to be loaded");
    println!("{:?}, i2w: {}, w2i: {}", model, i2w.len(), w2i.len());
}

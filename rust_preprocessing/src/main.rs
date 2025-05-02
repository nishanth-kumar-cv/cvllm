use tokenizers::tokenizer::Tokenizer;
use std::env;
use std::fs::{self, File};
use std::io::Write;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: rust_preprocessing <input_folder> <output_file>");
        return;
    }

    let input_folder = &args[1];
    let output_file = &args[2];
    let tokenizer = Tokenizer::from_file("mistral-tokenizer.json").unwrap();
    let mut output = File::create(output_file).unwrap();

    for entry in fs::read_dir(input_folder).unwrap() {
        let file_path = entry.unwrap().path();
        if let Ok(content) = fs::read_to_string(&file_path) {
            let encoding = tokenizer.encode(content, true).unwrap();
            let ids = encoding.get_ids();
            writeln!(output, "{{\"input_ids\": {:?}}}", ids).unwrap();
        }
    }
}

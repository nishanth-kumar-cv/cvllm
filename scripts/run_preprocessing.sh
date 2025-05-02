#!/bin/bash
cd rust_preprocessing
cargo build --release
./target/release/rust_preprocessing ../data/raw ../data/tokenized.jsonl

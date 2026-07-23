#[allow(dead_code)]
#[path = "../src/jamie_import.rs"]
mod jamie_import;

use std::{collections::BTreeMap, env, path::Path, time::Instant};

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        panic!("usage: cargo run --example inspect_jamie_export -- <export.txt>")
    });
    let started = Instant::now();
    let archive = jamie_import::parse_jamie_export(Path::new(&path))
        .unwrap_or_else(|error| panic!("{error}"));
    println!("importer_version={}", jamie_import::JAMIE_IMPORTER_VERSION);
    println!("source_sha256={}", archive.metadata.source_sha256);
    println!("source_size_bytes={}", archive.metadata.source_size_bytes);
    for (key, value) in jamie_import::archive_statistics(&archive) {
        println!("{key}={value}");
    }
    println!(
        "stable_aliases={}",
        jamie_import::stable_alias_counts(&archive).len()
    );
    let mut warning_codes = BTreeMap::<String, usize>::new();
    for warning in archive.warnings.iter().chain(
        archive
            .meetings
            .iter()
            .flat_map(|meeting| &meeting.warnings),
    ) {
        *warning_codes.entry(warning.code.clone()).or_default() += 1;
    }
    for (code, count) in warning_codes {
        println!("warning.{code}={count}");
    }
    println!("elapsed_ms={}", started.elapsed().as_millis());
}

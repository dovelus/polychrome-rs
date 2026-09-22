use std::{env, fs, path::PathBuf};

const META_FILE: &str = "resources/payload.meta.json";

fn main() {
    // Re-runs when the sidecar appears, changes or disappears.
    println!("cargo:rerun-if-changed={}", META_FILE);

    let json = fs::read_to_string(META_FILE).unwrap_or_else(|err| {
        println!(
            "cargo:warning=ShaderLoader: {} not readable ({}); \
             the binary will panic until it is generated",
            META_FILE, err
        );
        String::new()
    });

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by cargo")).join("meta.rs");
    fs::write(&out, format!("pub const META_JSON: &str = {:?};\n", json))
        .expect("failed to write meta.rs");
}

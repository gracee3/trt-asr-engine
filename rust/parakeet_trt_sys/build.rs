use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = manifest_dir.join("../../cpp");
    
    println!("cargo:rustc-link-lib=parakeet_trt");
    println!("cargo:rustc-link-search=native={}", cpp_dir.join("build").display());

    let bindings = bindgen::Builder::default()
        .header(cpp_dir.join("include/parakeet_trt.h").to_str().unwrap())
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

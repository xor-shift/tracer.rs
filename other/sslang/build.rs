fn main() {
    println!("cargo:rustc-env=OUT_DIR=.");
    //println!("cargo:rerun-if-changed=build.rs");
    //println!("cargo:rerun-if-changed=./src/grammar.lalrpop");
}

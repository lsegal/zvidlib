//! Puts the Swift runtime search path into this crate's own linked targets;
//! see `zvidlib-build` for why the macOS hardware backends need it (#327).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    zvidlib_build::swift_runtime_rpath();
}

//! Puts the Swift runtime search path into this crate's own linked targets,
//! which link the macOS hardware backends; see `zvidlib-build` (#327).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    zvidlib_build::swift_runtime_rpath();
}

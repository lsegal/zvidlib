//! Re-emits the Swift runtime search path for this crate's own linked targets.
//!
//! The macOS hardware decode path goes through `apple-cf`, whose build script
//! compiles a Swift bridge and links it statically. Everything Swift in that
//! bridge references the runtime as `@rpath/libswiftCore.dylib` and friends, so
//! the binary needs an `LC_RPATH` pointing at the OS Swift runtime in
//! `/usr/lib/swift` to launch at all.
//!
//! `apple-cf` does emit that rpath, but `cargo:rustc-link-arg` only applies to
//! the *emitting* package's own binaries, tests, examples, benches, and
//! cdylibs - it is not passed on to dependents. So `apple-cf`'s own targets
//! link fine and every `zvidlib` target links with no `LC_RPATH` at all, and
//! dyld aborts the process before `main` runs (issue #327). Re-emitting the
//! same flags here is what puts the rpath into *our* binaries.
//!
//! This is not arch-specific - an `aarch64` binary is missing the rpath in
//! exactly the same way - but on Apple silicon dyld finds the runtime in the
//! shared cache regardless, which is why CI's `macos-latest` job never saw it.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    // The ABI-stable runtime shipped with the OS. It lives in the dyld shared
    // cache rather than on disk, so this path is emitted unconditionally
    // instead of being probed for.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

    // The toolchain copy, as a fallback for hosts whose OS runtime predates a
    // symbol the bridge uses. Only emitted when it is really there; a
    // dangling rpath is harmless but noise in `otool -l` output.
    if let Some(dir) = xcode_swift_lib_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
}

fn xcode_swift_lib_dir() -> Option<String> {
    let output = Command::new("xcode-select").arg("-p").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let developer_dir = String::from_utf8(output.stdout).ok()?;
    let dir = format!(
        "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/swift/macosx",
        developer_dir.trim()
    );
    Path::new(&dir).is_dir().then_some(dir)
}

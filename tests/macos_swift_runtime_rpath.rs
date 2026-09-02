//! The macOS Swift bridge in `apple-cf` leaves `@rpath/libswiftCore.dylib`
//! references in everything that links it, so a `zvidlib` binary without an
//! `LC_RPATH` for the Swift runtime is aborted by dyld before `main` runs.
//!
//! That this test process started at all already proves the rpath resolves on
//! *this* host - dyld would have killed it otherwise. What the assertion below
//! adds is that the rpath is actually baked into the binary rather than being
//! supplied by a `DYLD_*` variable in the caller's environment, which is the
//! workaround issue #327 exists to replace, and that it survives on a host
//! whose dyld shared cache would have found the runtime anyway (Apple silicon,
//! where the missing rpath went unnoticed for exactly that reason).

#![cfg(target_os = "macos")]

use std::process::Command;

/// The `LC_RPATH` paths baked into `path`, read out of `otool -l`.
///
/// The load-command listing puts each rpath on its own `path <dir> (offset N)`
/// line two lines below its `cmd LC_RPATH`, so the parse tracks whether the
/// most recent `cmd` was an `LC_RPATH` and takes the next `path` after it.
fn baked_rpaths(path: &std::path::Path) -> Vec<String> {
    let output = Command::new("otool")
        .arg("-l")
        .arg(path)
        .output()
        .expect("otool is part of the Xcode command line tools this crate already builds with");
    assert!(
        output.status.success(),
        "otool -l {} failed",
        path.display()
    );
    let listing = String::from_utf8_lossy(&output.stdout);

    let mut rpaths = Vec::new();
    let mut in_rpath_command = false;
    for line in listing.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("cmd ") {
            in_rpath_command = rest.trim() == "LC_RPATH";
        } else if in_rpath_command && let Some(rest) = line.strip_prefix("path ") {
            let dir = rest.split(" (offset").next().unwrap_or(rest);
            rpaths.push(dir.trim().to_string());
            in_rpath_command = false;
        }
    }
    rpaths
}

#[test]
fn test_binary_bakes_in_a_swift_runtime_rpath() {
    let exe = std::env::current_exe().expect("a running test binary has a path");
    let rpaths = baked_rpaths(&exe);

    assert!(
        rpaths.iter().any(|dir| dir == "/usr/lib/swift"),
        "{} has no LC_RPATH for the OS Swift runtime, so it only launches when the \
         caller sets DYLD_FALLBACK_LIBRARY_PATH; found rpaths: {rpaths:?}",
        exe.display(),
    );
}

/// The same reference is in the shipped `cdylib`, not just in test binaries, so
/// the rpath has to reach it too. `cargo:rustc-link-arg` covers both, and this
/// pins that: a change that narrowed the fix to test targets would leave a
/// consumer of `libzvidlib.dylib` with the original launch failure.
#[test]
fn cdylib_bakes_in_a_swift_runtime_rpath() {
    // `target/<triple>/<profile>/deps/<test binary>` - the cdylib is built by
    // the same invocation and sits two directories up.
    let exe = std::env::current_exe().expect("a running test binary has a path");
    let profile_dir = exe
        .parent()
        .and_then(|deps| deps.parent())
        .expect("the test binary lives under <profile>/deps/");
    let cdylib = profile_dir.join("libzvidlib.dylib");
    if !cdylib.exists() {
        // `cargo test --lib` alone does not build the cdylib. Nothing to check.
        return;
    }

    let rpaths = baked_rpaths(&cdylib);
    assert!(
        rpaths.iter().any(|dir| dir == "/usr/lib/swift"),
        "{} has no LC_RPATH for the OS Swift runtime; found rpaths: {rpaths:?}",
        cdylib.display(),
    );
}

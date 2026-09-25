fn main() {
    // tch links libtorch from the oracle venv (LIBTORCH_USE_PYTORCH). Two
    // package-scoped link adjustments keep the final binary self-contained:
    //
    // 1. Keep libtorch_cuda in DT_NEEDED. Its CUDA kernels register via
    //    static initializers, but nothing in the shim references its symbols,
    //    so --as-needed prunes it and the CUDA dispatch never registers.
    //    Appended at the end of the link line, after torch-sys's -L path.
    // 2. Embed an rpath so the binary runs without LD_LIBRARY_PATH. The repo
    //    path contains a space, which breaks a literal -rpath value, so use
    //    $ORIGIN (target/debug -> repo/.venv is three levels up).
    println!("cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed,-ltorch_cuda,--pop-state");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../../.venv/lib/python3.12/site-packages/torch/lib"
    );
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../../../.venv/lib/python3.12/site-packages/torch/lib"
    );
    println!("cargo:rerun-if-changed=build.rs");
}

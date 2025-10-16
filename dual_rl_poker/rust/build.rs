fn main() {
    // Ensure unresolved Python symbols are resolved at runtime when building macOS wheels.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}

//! build.rs - 用 cbindgen 生成 include/aibridge.h
//!
//! 每次 `cargo build` 触发，根据 cbindgen.toml 与 src/lib.rs 生成 C 头文件。
//! 生成失败不阻断构建（开发期可继续），仅在 stderr 输出警告。

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));
    let config = cbindgen::Config::from_root_or_default(&crate_dir);

    let result = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate();

    match result {
        Ok(bindings) => {
            let out_dir = crate_dir.join("include");
            // 确保 include 目录存在
            std::fs::create_dir_all(&out_dir).ok();
            let header_path = out_dir.join("aibridge.h");
            if !bindings.write_to_file(&header_path) {
                eprintln!("cargo:warning=cbindgen 写入头文件失败");
            } else {
                // 让 cargo 在头文件变化时重新构建
                println!("cargo:rerun-if-changed=src/lib.rs");
                println!("cargo:rerun-if-changed=cbindgen.toml");
            }
        }
        Err(e) => {
            // 生成失败仅警告，不阻断构建（CI 仍可继续编译 cdylib）
            eprintln!("cargo:warning=cbindgen 生成头文件失败: {e}");
        }
    }
}

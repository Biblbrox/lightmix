## ViT inference project
### Inference setup (Native method)
Make sure you installed arm64 cross-compiler:
```
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Export pkg-config sysroot variable to your aarch64 sysroot:
```
export PKG_CONFIG_SYSROOT_DIR="/usr/aarch64-linux-gnu"
```

Run cargo build for the specified architecture:
```
cargo build --release --target aarch64-unknown-linux-gnu
```

If you want to build for x64, omit --target keyword:
```
cargo build --release
```

To specify a required glibc version edit file glibc.version and run cargo with the following variable:
```
RUSTFLAGS="-C link-arg=-Wl,--version-script=./glibc.version" cargo build --release --target aarch64-unknown-linux-gnu
```

### Inference setup (Docker-based cross method)
Make sure you have docker installed. If it's so, run the following command to build the project for aarch64 architecture:
```
cross build --release --target aarch64-unknown-linux-gnu
```

You can compile for the default architecture (x64) as well:
```
cross build --release
```

NOTE: compilation via cross can be slower if you use architecture different from your host one.

#### Usage
Inference build will generate spectre_vit binary. It supports the model path argument. In order to run inference with a specified model, use the following command:
```
./spectre_vit model_name.onnx
```
It will look for model_name.onnx and model.onnx.data in the specified path.
At this moment, ort api doesn't support custom names for data files. Thus, you're required to name data file 'model.onnx.data' exactly. 

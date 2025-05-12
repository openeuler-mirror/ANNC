# aicompiler

An AI compiler designed to optimize and compile ML model into high-performance executable code that can be executed on various targets.

### Build shared library
```bash
bazel --output_user_root=./output build -c opt annc/service/cpu:libannc.so
```

### Build static library
```bash
bazel --output_user_root=./output build -c opt annc/service/cpu:annc
```

### Build test
```bash
bazel --output_user_root=./output build -c opt annc/tools/kp-opt:kp-opt
bazel-bin/annc/tools/kp-opt/kp-opt {test_hlo_cluster_file.dat}
```

Use `--distdir=/path/to/proxy` to download files locally.
Use `--copt="-g"` for debug mode.

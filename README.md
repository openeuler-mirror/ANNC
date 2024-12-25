# aicompiler

An AI compiler designed to optimize and compile ML model into high-performance executable code that can be executed on various targets.

```bash
bazel --output_user_root=./output build \
    -c opt --copt="-std=c++17"          \
    aicompiler/tools/kp-opt:kp-opt      \
    --experimental_repo_remote_exec
```

Use `--distdir=/path/to/proxy` to download files locally.
Use `--copt="-g"` for debug mode.

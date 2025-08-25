<!--
Copyright 2025 Huawei. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
-->

To dynamically map a set of input dimensions to a symbol for the kernel selector:

# Environment variable

Set the `KERNEL_MAP_FILE` environment variable to point to a text file. This file must contain lines which look like the following:

```
[gemm](6656,8,8) -> __xla_cpu_runtime_KernelSelectorGEMM
```

This means that when M,N,K (in this order) are 6658,8,8 then the selector should use the GEMM implemented by the symbol `__xla_cpu_runtime_KernelSelectorGEMM`.

# Possible options:

```
[gemv](M,N) -> symbol
[gemm](M,N,K) -> symbol
[batch3d](P,M,N,K) -> symbol
[batch4d](P,Q,M,N,K) -> symbol
[argmax](M,N) -> symbol
```

Where `P`,`Q`,`M`,`N`,`K` are replaced by integer values.

# Special operators:

- Use `*` as a wildcard for *any* size. The following means that the function `f` will be used to run a gemm whenever M=32 and K=8, while N can be any possible value.

```
[gemm](32,*,8) -> f
```


- Use `:` to define ranges. The following means that the function `g` will be used to run a gemv whenever M is between 23 and 42 (inclusive), and N is equal to 8.

```
[gemv](23:42,8) -> g
```

- Both special operators can be combined. The following calls the function `h` whenever M is greater than 256, N is equal to 100, and K is any value:
```
[gemm](256:*,100,*) -> h
```

Whitespaces are ignored by the kernel selector. An example file (`example_kernel_map.txt`) is provided in the current directory.

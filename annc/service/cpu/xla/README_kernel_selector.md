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

# For GEMM

Set the `GEMM_MAP_FILE` environment variable to point to a text file. This file must contains lines which look like the following:

```
(6656,8,8) -> __xla_cpu_runtime_KernelSelectorGEMM
```

An example file (`gemm_map.txt`) is provided in the current directory.

This means that when m,n,k (in this order) are 6658,8,8 then the selector should use the GEMM implemented by the symbol (6656,8,8) -> __xla_cpu_runtime_KernelSelectorGEMM.

Be mindful of following the exact pattern. This is space-sensitive so will not work if spaces are added inside the tuple, for example.

# For GEMV.

Same as above, but set the `GEMV_MAP_FILE` environment variable. Its content should look like:

```
(m,n) -> function_name
```

Where `m`, `n`, are integers for the input sizes.

# For BATCH_MATMUL:

Same as above, but set the `BATCHMATMUL_MAP_FILE` environment variable. Its content should look like:

```
(p,m,n,k) -> function_name
```

Where `p`, `m`, `n`, `k` are integers for the input sizes.
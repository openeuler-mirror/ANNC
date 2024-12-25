# OpenXLA & TensorFlow Installation

Here provides some patches for OpenXLA and TensorFlow to quickly apply aicompiler optimization features.

### OpenXLA

Repository: https://github.com/openxla/xla.git
Commit ID: 40008cb2c85749ae436be61c40d3279cb24705c7
sha256: 90e72fa3558a69cf2562e4600e62c478d22c3986c642d7dcdc7ef0841ded52c5

To update OpenXLA to a new revision,
a. update URL and strip_prefix to the new git commit hash
b. get the sha256 hash of the commit by running:
   curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
   and update the sha256 with the result.

- Patches
  - xla.patch


### TensorFlow

Repository: https://gitee.com/openeuler/sra_tensorflow_adapter
Branch: r2.15

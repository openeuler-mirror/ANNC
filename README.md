# ANNC-Next

## 
ANNC-NextAI/MLMLIRopenEuler

## 
### 2.1 
|  |  |
| --- | --- |
|  | openEuler-24.03-LTS-SP4 |
|  | 920X |
|  | 500GB |
|  | LLVM 21.1.3 |

### 2.2 
- 50GB
- Gitee
- 

### 2.3 
```shell
# Python
pip install pybind11 nanobind

# 
yum install ninja-build cmake
yum-builddep llvm

#  yum
yum clean all
```
> ****:  `sudo` yum
## 2.4 LLVM&MLIR
> ****: ANNCMLIRMLIRANNC

```shell
git clone --depth 1 https://gitee.com/mirrors/LLVM.git --branch llvmorg-21.1.3

mkdir build

cd build

cmake -G Ninja \
  -DCMAKE_INSTALL_PREFIX=$YOUR_LLVM_PATH \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lldb;lld;mlir;llvm" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_EH=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python3) \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DLLVM_ENABLE_ZLIB=ON \
  ../LLVM/llvm

ninja && ninja install
```


> ****:
> 1. `${YOUR_LLVM_PATH}` LLVM `/opt/llvm-21.1.3`
> 2. 1-2
> 3. 255400G `-j` 


LLVM
```bash
export PATH=${YOUR_LLVM_PATH}/bin:$PATH
llvm-config --version  # 21.1.3
mlir-opt --version     # MLIR
```

## ANNC

```shell
# 
git clone https://codehub-dg-y.huawei.com/Computing_Product_Line_Compiler_Group/AICompiler/annc-toolchain.git --branch dev

# clone
cd annc-toolchain

# 
git submodule init
git submodule update

# 
mkdir -p build && cd build

# CMake
cmake -G Ninja .. \
  -DCMAKE_INSTALL_PREFIX=$YOUR_ANNC_INSTALL_PATH \
  -DMLIR_DIR=${YOUR_LLVM_PATH}/lib/cmake/mlir \
  -DLLVM_DIR=${YOUR_LLVM_PATH}/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=${YOUR_LLVM_PATH}/bin/clang \
  -DCMAKE_CXX_COMPILER=${YOUR_LLVM_PATH}/bin/clang++ \
  -DLLVM_ENABLE_LIBCXX=ON \
  -Dnanobind_DIR=/usr/local/lib/python3.*/site-packages/nanobind/cmake \
  -Dpybind11_DIR=/usr/local/lib/python3.*/site-packages/pybind11/share/cmake/pybind11 \
  -DCMAKE_CXX_FLAGS="-fPIC"

# ninja
ninja && ninja install
```

> ****:
> 1. `${YOUR_ANNC_INSTALL_PATH}` ANNC `/opt/ANNC`
> 2. Python `/usr/local/lib/python3.9/` `python3.9` Python
> 3. Python `~/.local/lib/python3.*/site-packages/`
> 4. 

## 
```bash
# 
export PATH=${YOUR_ANNC_INSTALL_PATH}/bin:$PATH

# 
which annc-opt annc-asm

# 
python -m pytest tests/
```

## 
1. ****:  `-D*_DIR` 
2. **Python**: pybind11nanobind
3. ****: 

## 
```shell
# 
cd /path/to/annc-toolchain/build

# tf-adaptortfjson
./bin/tf-adaptor ../tests/graph_demo ./test.json

# mlir
./bin/annc-opt ./test.json --atir-op-fusion -o output.bin -emit-bytecode

# anncmlir
./bin/annc-asm output.bin --atir-prune-func --atir-block-fusion --atir-tiling --atir-unroll --convert-atir-to-affine -o ../tests/annc/asm.mlir

# annc driverllvm
cd ../tests/annc
annc -t driver_dynamic.c -o 123 asm.mlir -v --save-temps --shared

# annc-verify
cd ../../build
./bin/annc-verify output.bin --atir-op-verify="kpGenLibPath=../tests/annc/fused_matmul_add_relu_A0732AD9DB33D09F.so"

# llmcodegenLLMtensorflow opkernel
./bin/annc-asm ./output.bin --atir-LLM-CodeGen

# annc-verify
./bin/annc-verify output.bin --atir-op-verify="llmGenLibPath=../annc/lib/Dialect/Atir/Passes/outputs/so/fused_matmul_add_relu_A0732AD9DB33D09F.so"
```

## 
- [LLVM](https://llvm.org/)
- [MLIR](https://mlir.llvm.org/)
- [openEuler](https://openeuler.org/zh/)

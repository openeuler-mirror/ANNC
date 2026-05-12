# 测试指南

# 环境要求

## 硬件要求

| 硬件配置项                              | 验收标准                    |
| -------------------------------------- | --------------------------- |
| 服务器型号                              | 鲲鹏 950         |
| CPU                                     | 80Cx4           |
| 缓存                                    | L1d: 10 MiB (160 instances) L1i: 10 MiB (160 instances) L2: 200 MiB (160 instances) L3: 280 MiB (4 instances) |
| Flags                                   | fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint svei8mm svef32mm svef64mm svebf16 i8mm bf16 dgh rng ecv |
| BIOS配置                                | Enable SMT 2  Enable SMMU  Enable CPU prefetching  Power policy-> performance  Performance Profile-> Custom(HPC off)|
| 内存                                    | 1024G（16通道x64G） |

## 软件要求
| 配置项 | 验收标准 |
| -----  | ------- |
|CMake   | 3.22.0  |
|Python  | 3.11    |
|TensorFlow| 2.15  |
|TF-Serving| 2.15  |
|Bazel     | 6.5.0 |
|TritonServer| 24.03|
|验收场景| wide\_and\_deep dlrm deepfm dffm dssm |

# 测试环境搭建
## 基线
步骤1：编译TF-Serving

```
git clone -b r2.15.1 https://github.com/tensorflow/serving.git
bazel --output_user_root=./output build -c opt --distdir=./proxy \
  tensorflow_serving/model_servers:tensorflow_model_server
```

步骤2：测试模型基线性能

## 使能ANNC优化

### 步骤1：拉取ANNC源码

1）拉取ANNC源码

```
git clone -b v0.0.4 https://gitcode.com/openeuler/ANNC.git
export ANNC="your_path_to_ANNC"
```
### 步骤2：编译ANNC

2）进入ANNC路径，编译ANNC。构建前修改脚本中的环境变量，包括`BAZEL`, `GO`,`CC`, `CXX`, `LD_LIBRARY_PATH` 和 `CPLUS_INCLUDE_PATH`

```
cd $ANNC
bash build.sh

cp $ANNC/bazel-out/aarch64-opt/bin/annc/service/cpu/xla/libs/libblas_mlir.so /usr/lib64
export XNNPACK_BASE="$ANNC/annc/service/cpu/xla/libs"
export XNNPACK_DIR="$XNNPACK_BASE/XNNPACK"

CPLUS_INCLUDE_PATH+="$ANNC/annc/service/cpu/xla/:"
CPLUS_INCLUDE_PATH+="$ANNC/annc/service/:"
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/:"
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/include/:"
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/src/:"
CPLUS_INCLUDE_PATH+="$XNNPACK_DIR/build/pthreadpool-source/include/:"
export CPLUS_INCLUDE_PATH
```
3）将构建完成的libannc.so和头文件拷贝到/usr/lib64目录下，其中XXX是构建路径的哈希码
```
cp $ANNC/output/XXX/execroot/aicompiler/bazel-out/aarch64-opt/bin/annc/service/cpu/libannc.so /usr/lib64
mkdir -p /usr/include/annc
cp annc/service/cpu/kdnn_rewriter.h /usr/include/annc/
cp annc/service/cpu/annc_flags.h /usr/include/annc/
```

4）安装图优化模型转换工具

```
cd python
python3 setup.py bdist_wheel
python3 -m pip install dist/*.whl --force-reinstall
```

### 步骤3：算子注册

5）进入TF-Serving目录，创建自定义算子文件夹
```
cd /path/to/serving
mkdir tensorflow_serving/custom_ops
```

6） 将ANNC提供的开源算子库拷贝到自定义算子文件夹下
若通过方式一安装，算子库位于/usr/include/annc：
```
cp /usr/include/annc/fused_sparse_embedding.cc tensorflow_serving/custom_ops/
cp /usr/include/annc/fused_dnn_embedding_with_hash_bucket.cc tensorflow_serving/custom_ops/
cp /usr/include/annc/fused_linear_embdedding_with_hash_bucket.cc tensorflow_serving/custom_ops/
cp /usr/include/annc/fused_embdedding_with_hash_bucket.cc tensorflow_serving/custom_ops/
```

若通过方式二安装，算子库位于$ANNC/python/tensorflow/kernels：

```
cp $ANNC/python/tensorflow/kernels/fused_sparse_embedding.cc tensorflow_serving/custom_ops/
cp $ANNC/python/tensorflow/kernels/fused_dnn_embedding_with_hash_bucket.cc tensorflow_serving/custom_ops/
cp $ANNC/python/tensorflow/kernels/fused_linear_embdedding_with_hash_bucket.cc tensorflow_serving/custom_ops/
cp $ANNC/python/tensorflow/kernels/fused_embdedding_with_hash_bucket.cc tensorflow_serving/custom_ops/
```

7）创建算子编译文件，并注册到TF-Serving中

- `vim tensorflow_serving/custom_ops/BUILD`，写入内容如下

```
package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = 'recom_embedding_ops',
    srcs = [
        "fused_sparse_embedding.cc",
        "fused_linear_embdedding_with_hash_bucket.cc",
        "fused_dnn_embedding_with_hash_bucket.cc",
        "fused_embdedding_with_hash_bucket.cc",
    ],
    alwayslink = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@eigen_archive//:eigen3",
        "@farmhash_archive//:farmhash",
    ]
)
```

- `vim tensorflow_serving/model_servers/BUILD`，搜索`SUPPORTED_TENSORFLOW_OPS`，在末尾位置增加一行算子注册

```
SUPPORTED_TENSORFLOW_OPS = if_v2([]) + if_not_v2([
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
]) + [
    "@org_tensorflow_text//tensorflow_text:ops_lib",
    # Support for TensorFlow Decision Forests inference ops.
    # [September 2022] This dependency adds 3 MB to the model server (352 MB -> 355 MB i.e. + 0.8%).
    "@org_tensorflow_decision_forests//tensorflow_decision_forests/tensorflow/ops/inference:kernel_and_op",
    "//tensorflow_serving/custom_ops:recom_embedding_ops"
]
```

### 步骤4：编译TF-Serving

8）编译TF-Serving

```
cd serving
bazel --output_user_root=./output build -c opt --define tflite_with_xnnpack=false --distdir=$DIST_DIR tensorflow_serving/model_servers:tensorflow_model_server
```

构建时遇到upb.c的构建问题，进入文件output/XXX/external/upb/upb/upb.c，其中XXX是构建路径的哈希码，按照如下修改：
```
void upb_status_seterrmsg(upb_status *status, const char *msg) {
  if (!status) return;
  status->ok = false;
  // 修改前
  // strncpy(status->msg, msg, sizeof(status->msg));
  // 修改后
  strncpy(status->msg, msg, sizeof(status->msg) - 1);
  // <<<<<<
  nullz(status);
}
```

### 步骤5：使能算子优化和图优化特性

9）在构建好的server的tensorflow的xla路径下，通过补丁脚本使能以下补丁：

```
export TF_PATH="$HOME/serving/output/XXX/external/org_tensorflow"
export XLA_PATH="$HOME/serving/output/XXX/external/org_tensorflow/third_party/xla"

# 通过方式一安装的ANNC：
cd /usr/include/annc/tfserver/xla
# 修改xla2.sh前两行为：
TF_PATCH_PATH="$ANNC" 
PATH_OF_PATCHES="$ANNC/xla"
export ANNC_PATH=/usr/include/annc
bash xla2.sh

# 通过方式二安装的ANNC：
cd $ANNC/install/tfserver/xla
export ANNC_PATH=$ANNC
bash xla2.sh
```

10）**补丁应用后，参考 [步骤4] 重新构建server**。

### 步骤6：获取测试模型

11）下载基线模型

```
git clone https://gitcode.com/openeuler/sra_benchmark.git
```

从基线模型库中获取目标推荐模型：

* DeepFM：modelzoo/deepfm/result/model\_DeepFM
* DFFM: modelzoo/dffm/result/model\_DFFM
* DLRM: modelzoo/dlrm/result/model\_DLRM
* DSSM: modelzoo/dssm/result/model\_DSSM
* W&D: modelzoo/wide\_and\_deep/result/model\_WIDE\_AND\_DEEP

12）命令行实现常量折叠和图融合

```
# 运行模型转换
annc-opt -I /path/to/DeepFM_cf/1730800001/1 \
    -O folding/1 \
   layout_matmul
cp folding/1/saved_model.pbtxt /path/to/deepfm_cf/1/
cp -r folding/1/variables /path/to/deepfm_cf/1/
annc-opt -I /path/to/deepfm_cf/1730800001/1 \
    -O deepfm_new/1 \
   embed_hash_bucket
cp -r folding/1/variables /path/to/deepfm_new/1/
```

13）设置基础环境变量

```
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=16"
export OMP_NUM_THREADS=1
export PORT=7004  # 端口号
```

# 性能测试

## 测试方法

- 基线和带优化特性各测试 3~5 轮，取吞吐率中位数对比，作为性能测试结果
- 模型参数配置表如下:
  
| model | measurement_interval | batch | concurrency | min_cluster_size |
| ----- | -------------------- | ----- | ----------- | ---------------- |
| wide_and_deep | 10000 |   128/150 | 60:80:2 | 16 |
| dlrm      | 10000 |   200/256/300 | 75:90:2 | 16 |
| deepfm    | 10000 |   150/200/256 | 70:80:2 | 16 |
| dffm      | 10000 |   100/128     | 65:80:2 | 16 |
| dssm      | 10000 |   128/256/512 | 60:80:2 | 16 |
  
- 初始环境变量配置如下

```
# TensorFlow XLA编译优化标志：
# --tf_xla_auto_jit=2: 启用自动JIT编译，对所有可能的操作进行XLA编译
# --tf_xla_cpu_global_jit: 启用CPU全局JIT编译，跨函数边界进行优化
# --tf_xla_min_cluster_size=16: 设置XLA最小聚类大小为16，控制融合算子的规模
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=16"

# 线程数设置为1，确保性能测试的单线程基准一致性，避免多线程干扰
export OMP_NUM_THREADS=1
```

## 测试对照组


### base对照组

仅启用xla，不使能优化特性

```
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=16"
export OMP_NUM_THREADS=1

# 启动TF-Serving
/path/to/tensorflow_model_server  --port=$PORT --model_name=wide_and_deep --model_base_path=/path/to/wide_and_deep/1731296400     --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=-1 --xla_cpu_compilation_enabled=true

# 启动client
docker run --rm --net host nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 60:80:2  -p 8000 --latency-threshold 200 -f perf.csv -m wide_and_deep --service-kind tfserving -i grpc --request-distribution poisson -b 128 -u localhost:$PORT --percentile 99 --input-data=random
```


### 完整ANNC优化

通过模型预处理与环境变量使能完整优化特性

步骤1：完成算子注册和模型转换（参考前述步骤）

步骤2：进行测试

```
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_min_cluster_size=16"
export OMP_NUM_THREADS=1

export ANNC_FLAGS="--graph-opt --gemm-opt"
export ENABLE_BISHENG_GRAPH_OPT=""
export XLA_FLAGS="--xla_cpu_enable_xnnpack=true"

# 启动TF-Serving
/path/to/tensorflow_model_server  --port=$PORT --model_name=wide_and_deep --model_base_path=/path/to/wide_and_deep/1731296400     --tensorflow_intra_op_parallelism=1 --tensorflow_inter_op_parallelism=-1 --xla_cpu_compilation_enabled=true

# 启用client
docker run --rm --net host nvcr.io/nvidia/tritonserver:24.05-py3-sdk perf_analyzer --concurrency-range 60:80:2  -p 8000 --latency-threshold 200 -f perf.csv -m wide_and_deep --service-kind tfserving -i grpc --request-distribution poisson -b 128 -u localhost:$PORT --percentile 99 --input-data=random
```

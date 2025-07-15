# TF2.2使能XLA

## 修复aarch64架构支持，开启XLA

- 下载patch

https://gitee.com/openeuler/ANNC/blob/tf2.2/install/tf2_2_aarch64.patch

- 应用patch

```
cd /path/to/tensorflow
patch -p1 < /path/to/tf2_2_aarch64.patch
```

- 编译TensorFlow（开启XLA配置）

在编译命令中增加配置 `--config=xla`

```
#示例 #编译动态库
bazel build --config=xla --config=v2 --config=opt --config=noaws  //tensorflow:tensorflow_cc

#编译python版本
bazel build --config=xla \
	--config=v2 --config=opt --config=noaws //tensorflow/tools/pip_package:build_pip_package
```

- 配置环境变量

```
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_lazy_compilation=false --tf_xla_min_cluster_size=16"
```

# TF2.2 使能ANNC

* 下载ANNC源码

```bash
git clone -b tf2.2 https://gitee.com/openeuler/ANNC.git
```

* 在tf2.2 源码路径下应用patch

```bash
cd /path/to/tensorflow
patch -p1 < /path/to/ANNC/install/tensorflow.patch
```

* 将ANNC文件移到对应文件夹下

```bash
cd /path/to/tensorflow
mkdir ./tensorflow/compiler/xla/ANNC
cp -r /path/to/ANNC/annc ./tensorflow/compiler/xla/ANNC
cp -r /path/to/ANNC/third_party ./
```

**`注意`**`依赖第三方库openblas`:/path/to/ANNC/third\_party/openblas/repo.bzl

![](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI202506287303016/24841855/1e2ba0017b424c5bb5669bbcb87728b4.png)

* 编译TensorFlow，使能ANNC

 编译命令中增加配置`--config=xla`

```bash
bazel build --config=v2 --config=xla --config=opt --config=noaws  //tensorflow:tensorflow_cc
```

# TF2.2 使能ANNC remapper & l常量折叠优化

* 下载ANNC最新代码

```bash
git clone -b tf2.2 https://gitee.com/openeuler/ANNC.git
```

* 拷贝图优化及模型优化模块至目标路径

```
cp -r /path/to/ANNC/annc/tensorflow/graph_optimizer \
  /path/to/tensorflow/tensorflow/core/grappler/optimizers/

cp -r /path/to/ANNC/annc/tensorflow/model_optimizer \
/path/to/tensorflow/tensorflow/cc/saved_model
```

* 在tf2.2路径下应用补丁

```
 patch -p1 < /path/to/ANNC/annc/tensorflow/tf_annc_optimizer.patch
```

* 默认开启全部pattern图重写，可通过配置环境n变量关闭（0 :关闭，  1：开启）

```
# pattern 1-7
export  ANNC_FUSED_SPS_STITCH=0
export  ANNC_FUSED_SPS_REDUCE=0
export  ANNC_FUSED_EMD_PADDING=0
export  ANNC_FUSED_EMD_PADDING_FAST=0
export  ANNC_FUSED_SPS_SELECT=0
export  ANNC_FUSED_GATHER=0
export  ANNC_FUSED_SPS_RESHAPE=0
export  ANNC_FUSED_EMB_ACTIONID_GATHER=0
```

* 默认开启常量折叠优化，可通过配置环境n变量关闭（0 :关闭，  1：开启）

```
export ANNC_CF_MATMUL_BADD_BN=0
export ANNC_CF_RELU=0
```


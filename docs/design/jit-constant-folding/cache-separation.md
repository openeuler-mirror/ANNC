# JIT GEMM 代码与预打包权重分离：实现契约

| 元数据 | 内容 |
| --- | --- |
| 编号 | ANNC-JIT-CF-CACHE-001 |
| 状态 | implemented，2026-10-25 |
| 责任角色 | TensorFlow addon / AArch64 GEMM 维护者 |
| 源码基线 | 0e554ff |
| 关联决策 | [ADR-017](../../decisions/decisions.md#adr-017) |

## 范围与前提

修复 TensorFlow ANNCFusedOp 同模板/shape、不同权重错误复用 JIT 代码的问题。
预打包能力始终构建，运行时统一读取 ANNC_GEMM：
0/未设置关闭自动 GEMM fusion，1=Basic，2=Epilogue，3=Prepack，99=All。
模式 3 不开启 epilogue，模式 2 不开启预打包；99 固定保留，不随新增模式移动。
同一套产物支持全部模式，不需要编译期开关。KDNN packed-B 适配器按依赖库的
实际接口能力构建注册；缺少该接口不影响 ANNC 原生预打包。
非法模式提示合法值并回退 0，空值为 0，首尾空白允许。
JIT 代码不内嵌权重，无权重快照/哈希。

候选 RHS 必须在节点生命周期内不可变，模型更新创建新实例。fixed 仅表示
形状，不证明内容不可变。首次初始化复用静态 prepack 的 checkpoint 名称匹配、
F32 类型及数据字节数检查；仅通过检查的输入进入 immutable 参数集合。
原始名称由既有 annc-fusion-metadata 从 atir_module_path 提取，按 kernel_name
和 template_fingerprint 选择当前融合，并恢复 constants/fixed/dynamic 输入顺序；
不新增 GraphDef 属性。元数据不可读取或不匹配时保守使用普通 packing。
checkpoint 使用既有 ANNC_DEMO_CHECKPOINT_DIR，不匹配或不可读取的输入
走普通 runtime-packed/direct。checkpoint 只用于准入，打包仍读取当前节点
首次输入 Tensor；不证明运行期不可变，也不检测原地更新。在线更新已准入
权重仍不支持；可变权重使用模式 `0`/`1`/`2`，本次不新增版本协议。

AOT/手工 pass 保留 checkpoint/global 路径。默认内置 GEMM fusion 标记为
JIT，AOT 分支按执行模式裁剪；不扩展 AOT、ONNX EP 或 execution_v2 JIT 特化。

## 缓存身份与布局

代码复用既有进程级 single-flight/LRU。身份为模板、实际参数 shape、
packed-rhs-v1、执行 ABI 与不可变 kernel 参数索引，不含权重数值。
工具链、目标、配置及跨会话线程预算仍须一致，沿用已有缓存约束。

PackedRhsCache 由各节点独立持有，短 vector 按完整 PackSpec 比较：
version、argument、K、N、KC、NC、ISA、panel、elements、alignment。
NEON 使用 packed-b-v1、panel=4；SVE 使用 packed-b-v2，panel 反映 VL。
完整 pc-jc 布局相同才跨 batch 复用；切块不同分别准备。节点所有权隔离
不同权重，不引入全局权重哈希表或跨节点数据去重。

SVE 生成布局时与既有 packer sizing helper 核对 VL、块容量及偏移；节点
准备 packed buffer 前再检查当前线程 VL，不匹配则报错且不发布缓存。
执行线程须保持与目标一致的 VL，不支持运行期间切换；热路径不重复检查。

## 热路径与生命周期

预打包开关在首次读取后固定，Op 构造时保存，仅 last-shape miss 路径使用；切换需要
重启服务。关闭时不初始化权重、不建立绑定、不生成或链接 packing C。
开启的代码身份含 packed-rhs-v1/ABI/参数槽，关闭保留原模板身份，二者不混用。
TF pipeline 自动 AOT 预打包及 packed KDNN 候选遵循同一开关；已生成 AOT
库和显式手工 packed-c/pass 不受变量改写。

1. last-shape 命中同时取得 executable 与节点 binding，无初始化或开关检查。
2. shape 切换查原代码缓存，再按 executable 取节点已有绑定。
3. 绑定未命中时验证布局；新布局才分配、调用既有 PackB、发布数据。
4. 构建输入 descriptor 时替换对应 RHS 的 allocated/aligned 指针，保持
   逻辑 shape、原始 Tensor、kernel 参数数目及微内核/线程池 ABI 不变。

每个节点首次初始化在全局代码缓存查找之前执行，即使其他节点已编译同模板
也独立完成准入。元数据提取子进程和 checkpoint 读取只发生在这一次初始化，
临时字节随初始化释放；未配置 checkpoint 时不启动提取工具。保留通过准入的
Tensor 引用并记录 immutable 参数，不复制或哈希运行期权重，后续任意 shape
均不重读元数据或 checkpoint。
last-shape miss 不等于全局代码缓存 miss；已出现的 shape 可命中代码与
绑定，动态 batch 交替时这一查找路径也可能频繁执行。
初始化仅在首次 last-shape miss 执行一次。
冷路径节点 mutex 保证并发首次准备只做一次。
调用期间强引用代码与 packed 数据；绑定弱引用 executable，过期绑定在
后续 miss 清理。代码淘汰重编可复用原布局数据。packed 数据保留到节点
卸载，内存随遇到的布局种类增长。引用计数、分支、指针写入仍有成本，
不宣称数学意义的零开销。

## 编译器与私有 ABI

JIT 传入 jit-inputs：只含不可变 kernel 参数索引的 JSON 数组，例如 [1]。
不传权重二进制或 checkpoint。JitPrepack 在 Finalize 前读取已选策略及
线程规划，复用 partitionCache2D，生成只读布局和调用现有 packb 的准备 C。
沿用 packed-c / annc --packed-rhs-c 链接，不新增共享库或 driver 协议。

只有函数输入、连续二维 F32、单一 GEMM 消费者（包括 cast 链）可替换。
共享 RHS、转置、非连续等不合格候选回退已有普通 packed 路径。
Finalize 传播 rhs_data_input，leaf 从原 RHS descriptor 基址和已有布局
偏移读取；线程子任务常量偏移使用 fullN，不能使用 task 局部 N。

私有入口 annc_jit_packed_rhs_v1 / annc_jit_prepare_rhs_v1 随 executable
加载一次。校验版本、维度、容量、对齐、ISA/panel 后复制布局；稳态不读
manifest、不 dlsym、不重新打包。编译/分配/ABI 错误沿用错误传播；准备
函数同步调用已有 packb，本次不新增 fallback 或内部硬件故障恢复协议。

## 回归入口

新增测试使用项目既有的 GTest/CTest、FileCheck 和 pytest 入口。

- GemmConfig：验证数字模式、两项优化的独立组合、非法值诊断及首次读取后固定。
- tests/integration/cmake/test_kdnn_build.py：验证 LOCAL / RELEASE 依赖的 packed-B 接口与链接
  符号检测，以及切换依赖后重新检测，缺少接口时保留普通适配器。

- PackedRhsCacheTest：节点权重隔离、完整布局判等、布局切换、并发首次
  准备、引用生命周期和非法契约不发布。沿用 JitCompilationCacheTest
  验证代码缓存淘汰及 single-flight。
- tests/FileCheck/Target/aarch64/gemm-jit-prepack*.mlir：参数式预打包、
  shared RHS 回退、manifest 验证及生成布局；JIT 不生成权重 global。
- tests/integration/jit/test_jit_prepack.py：本机 AArch64 下编译普通/预打包 kernel，
  K=73/N=67、batch 1/8/32/100、线程规划 1/8；同一 kernel 交替使用两个
  权重并回访，完整矩阵与 float64 参考比较，覆盖尾块和线程切片偏移。
  没有 TF 线程池时任务串行执行，不据此声称覆盖并行调度。

运行方式：

    ctest --test-dir build -R "GemmConfig|JitCompilationCache|PackedRhsCache" --output-on-failure
    python3 -m pytest tests/integration/cmake/test_kdnn_build.py -v
    bash tests/FileCheck/run.sh Target/aarch64
    ANNC_BUILD_DIR=build python3 -m pytest tests/integration/jit/test_jit_prepack.py -v

数值测试依赖 numpy、pytest 和已构建的 annc/annc-asm。非 AArch64、缺少
工具时会明确跳过；可沿用 ANNC_CLANG 等 driver 环境配置。
测试自行生成输入与临时产物，不依赖外部模型、服务可执行文件或本机绝对路径。

权重更新、可变线程预算/配置和 execution_v2 JIT 特化不在本次覆盖范围内。

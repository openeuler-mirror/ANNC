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
ANNC_ENABLE_CONSTANT_FOLDING 开启即使用分离缓存，关闭则使用已有普通
runtime-packed/direct。JIT 代码不内嵌权重，无权重快照/哈希或运行时模式路由。

候选 RHS 必须在节点生命周期内不可变，模型更新创建新实例。fixed 仅表示
形状，不证明内容不可变；保留 Tensor 也不检测原地更新。可变权重使用关闭
常量折叠的构建，本次不新增版本协议。

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

## 热路径与生命周期

1. last-shape 命中同时取得 executable 与节点 binding，无初始化检查或模式路由。
2. shape 切换查原代码缓存，再按 executable 取节点已有绑定。
3. 绑定未命中时验证布局；新布局才分配、调用既有 PackB、发布数据。
4. 构建输入 descriptor 时替换对应 RHS 的 allocated/aligned 指针，保持
   逻辑 shape、原始 Tensor、kernel 参数数目及微内核/线程池 ABI 不变。

初始化仅在首次 shape miss 执行一次，保留 Tensor 引用并记录 immutable
参数，不复制或哈希权重。冷路径节点 mutex 保证并发首次准备只做一次。
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

- PackedRhsCacheTest：节点权重隔离、完整布局判等、布局切换、并发首次
  准备、引用生命周期和非法契约不发布。沿用 JitCompilationCacheTest
  验证代码缓存淘汰及 single-flight。
- tests/FileCheck/Target/aarch64/gemm-jit-prepack*.mlir：参数式预打包、
  shared RHS 回退、manifest 验证及生成布局；JIT 不生成权重 global。
- tests/integration/jit/test_jit_prepack.py：本机 AArch64、CF-ON 下编译普通/预打包 kernel，
  K=73/N=67、batch 1/8/32/100、线程规划 1/8；同一 kernel 交替使用两个
  权重并回访，完整矩阵与 float64 参考比较，覆盖尾块和线程切片偏移。
  没有 TF 线程池时任务串行执行，不据此声称覆盖并行调度。

运行方式：

    ctest --test-dir build -R JitCompilationCache --output-on-failure
    bash tests/FileCheck/run.sh Target/aarch64
    ANNC_BUILD_DIR=build python3 -m pytest tests/integration/jit/test_jit_prepack.py -v

数值测试依赖 numpy、pytest 和已构建的 annc/annc-asm。非 AArch64、缺少
工具或 CF-OFF 配置会明确跳过；可沿用 ANNC_CLANG 等 driver 环境配置。
测试自行生成输入与临时产物，不依赖外部模型、服务可执行文件或本机绝对路径。

权重更新、可变线程预算/配置和 execution_v2 JIT 特化不在本次覆盖范围内。

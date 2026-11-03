# GEMM epilogue 二进制 codegen 集成契约

- **状态**：已实现；该契约适用于当前 ANNC 与 GEMM-ASM 二进制包接口。

## 职责与交付

ANNC 保留既有语义分析、选核、分块、KC 角色、调用参数及 suffix。
私有 GEMM-ASM 根据所选 microtile 与有序 epilogue 链决定可融合前缀，
负责寄存器分配、指令生成和 ELF object。ANNC 不再保存 emitter、
寄存器表、私有模板名或汇编文本，不读取私有 checkout。

一份版本化包位于 `third_party/annc-gemm-kernels/`，包含基础 `.a`、
编译期 codegen `.so`、公开头文件、CMake 配置、manifest 与许可证。
CMake 导入 `ANNC::AArch64GemmMicrokernels` / `ANNC::GemmCodegen`，
校验文件摘要、API 1 和 kernel ABI 1。
两个库整体升级。缺包可以构建普通工具；实际 epilogue 请求会在执行时明确失败。
`ANNC_REQUIRE_GEMM_PACKAGE=ON` 是可选的构建期强校验，适合 CI、发布构建等场景，
会在 CMake configure 阶段发现缺少兼容完整包时直接失败，避免运行时才报错。

编译工具链接 codegen，安装时携带整包并使用相对 RPATH。
模型 `.so` 链接生成的 `.o` 和原有运行时归档，不依赖 codegen。
该包面向已验收的 AArch64 Linux 宿主/目标组合，不承诺交叉编译。

## 查询与生成

`ANNCGemmCodegenAdapter` 只转换公开语义描述和错误，不依赖 MLIR。
后端桥接层提交所选 family、实际 microtile M/N/K、VL、累加模式与有序操作；
KC 分块角色只用于 ANNC 在 leaf 阶段标记 epilogue 插入点，不进入 codegen 请求；
K-unroll/余数归一化由库内部完成。

现有范围为 NEON packed/direct/matvec、SVE packed/direct 的 F32，
Add/Mul 支持 scalar/N 广播，Relu 保留可选上限，Sigmoid 保留原路径
常量和运算顺序；M/matrix 等未支持部分保留 suffix。每个 binary step
占一个运行时输入槽，最多 8 槽。单次 prepare 接收完整候选并返回可生成前缀
及该前缀的语义请求；不支持的首步返回空请求，ANNC 将其保留在常规 lowering。
first/middle KC 不执行 epilogue。
向量-矩阵路径与 suffix 算法保持现状，本次不扩大融合范围。

线程 task ABI 也固定为 8 槽。当前超过容量的候选会在后端线程划分
时被拒绝；优化阶段提前限制或切分这类候选仍待实现。容量内的候选
在实际 microtile 已知后由私有库选择可融合前缀，剩余链作为 suffix 执行。

`GemmKernelTiling` 后按数据逐步具体化：`GemmLeafMaterialization` 只处理
GEMM 主体，在最终 microtile 上选择一次 microkernel，把符号、family、KC 等
选择结果写入 leaf 的 `annc.aarch64.final_microkernel_selection`；M/N/K 仍取自
leaf 已有属性，并以 `annc.aarch64.epilogue_insertion_point` 标记 final-K leaf，
在其后保留完整 epilogue carrier。`GemmEpilogueLowering` 位于 leaf
之后，只消费这份既有选择和 carrier 中的 epilogue 链：调用 ASM prepare 可融合
前缀、将纯 leaf 转为 fused leaf，并把剩余链物化为 suffix；它不读取
`GemmPlan` 重新选核。随后 `GemmMicrokernelLowering` 只绑定已选普通符号或
生成入口，`GemmABILowering` 只把 leaf 调用转换为汇编 ABI。
最终选核记录是跨 pass 携带选择结果的结构化字典，family 用整数枚举保存；
KC role 保留在 ANNC 的分块属性中，不下沉到 codegen adapter。

## IR 与 object 生成

`annc.aarch64.generated_microkernel` 是恰含五个字段的字典：

| 字段 | 含义 |
| --- | --- |
| `version` | IR 元数据版本，当前为 1，与私有请求 schema 1 独立 |
| `symbol` | 库返回的当前调用入口 |
| `normalized_request` | 规范化请求字节的小写十六进制表示 |
| `artifact_key` | 生成单元身份；一个 object 可提供多个入口 |
| `build_id` | 同一发布包生成器身份 |

`annc-gemm-epilogue-kernel-compile` 按 artifact 身份去重，拒绝同身份不同请求、
同入口不同 artifact、缺失入口和外部依赖符号。adapter 校验 codegen 包身份和 ABI。
compile 返回的 object 写入 `model_generated_kernels/<artifact_key>.o`。
编译工具要求 `model_generated_kernels` 子目录尚不存在；失败时清理本次生成的对象。
空请求创建空目录，不会链接旧 object。
旧 `{symbol, asm}` 属性显式报错，需重新 lowering。

adapter 校验生成对象的包身份和 ABI，链接器校验对象格式与架构。
`annc` 在独立的本次编译临时目录中收集生成的 `.o` 并链接，
不再调用私有模板汇编流程。

## 验证与限制

公开测试检查语义请求、调用 ABI、object 契约、缺库行为和真实
生成链接；指令与寄存器对照在私有仓库测试。新旧数值比较必须使用
迁移前的同一融合路径；原融合 Sigmoid 与 suffix 算法已有差异，
不以数学参考替换兼容性对照，也不借此修改原算法。
私有回归同时比较旧独立参考与新生成 object 的完整 `.text`，
保持逐字节相等；matvec 先对齐两个旧生成单元与新四入口 object
的组织，不归一化指令。符号名和 ELF 元数据不要求相同。

性能验证应在缓存命中的热路径上比较 kernel 延迟与稳态推理性能，
并将首次编译成本单独记录。公开测试验证请求、ABI、对象和链接契约；
指令、寄存器和迁移前后数值对照属于 GEMM-ASM 私有测试。

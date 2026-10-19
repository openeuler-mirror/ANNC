# 历史夹具

这些输入目前没有被任何自动化测试引用，保留是为了避免在目录重构中丢失可能
仍有参考价值的数据。

- `config-fusion-parser-smoke.json`：`config-fusion` 解析器的早期 smoke 配置
  （`parser_smoke` pattern，`out = Relu(input)`），原位于
  `tests/FileCheck/Dialect/Atir/`。当前没有对应的 `.mlir` 用例；如果后续补
  解析器负例/正例，应把用例放到 `tests/FileCheck/` 或
  `tests/integration/fusion/` 并在此文件被实际引用后移出 `legacy/`。

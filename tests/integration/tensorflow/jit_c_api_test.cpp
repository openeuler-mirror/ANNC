#include <unistd.h>

#include <array>
#include <cerrno>
#include <iostream>
#include <memory>
#include <string>

#include "Kernel/MemRefTypes.h"
#include "jit_cache.h"

namespace {
#define CHECK(condition)                          \
  do {                                            \
    if (!(condition)) {                           \
      std::cerr << "Failed: " #condition << '\n'; \
      return false;                               \
    }                                             \
  } while (false)

bool CheckInvalidDimensions(AnncJitCompileRequest request) {
  const int64_t* dims[] = {nullptr};
  request.argument_count = 1;
  request.argument_dims = dims;
  AnncJitKernelResult result{};
  anncCompileJitKernel(&request, &result);
  annc::tensorflow::JitKernelPtr handle(result.kernel);
  CHECK(result.status == ANNC_JIT_INVALID_ARGUMENT && handle);
  CHECK(!result.kernel_function && result.error && *result.error);
  return true;
}

annc::jit::JitCompileResult Compile(AnncJitCompileRequest request) {
  AnncJitKernelResult result{};
  anncCompileJitKernel(&request, &result);
  annc::tensorflow::JitKernelPtr handle(result.kernel);
  if (result.status != ANNC_JIT_OK) {
    return {nullptr, result.error ? result.error : ""};
  }
  return {std::make_shared<annc::jit::JitExecutable>(std::move(handle), result),
          ""};
}

bool ComputesSum(void* function) {
  CHECK(function);
  float a[] = {1, 2, 3, 4, 5};
  float b[] = {4, -8, 2, 0, 7};
  std::array<float, 5> out{};
  AnncMemRef2DF32 lhs{a, a, 0, {1, 5}, {5, 1}};
  AnncMemRef2DF32 rhs{b, b, 0, {1, 5}, {5, 1}};
  AnncMemRef2DF32 dst{out.data(), out.data(), 0, {1, 5}, {5, 1}};
  using Kernel = void (*)(AnncMemRef2DF32*, AnncMemRef2DF32*, AnncMemRef2DF32*);
  reinterpret_cast<Kernel>(function)(&lhs, &rhs, &dst);
  const std::array<float, 5> expected{{5, -6, 5, 4, 12}};
  CHECK(out == expected);
  return true;
}

bool CheckCache(AnncJitCompileRequest request) {
  annc::jit::JitCompilationCache cache(1);
  auto invalid = request;
  invalid.kernel_name = "missing_kernel";
  AnncJitKernelResult result{};
  anncCompileJitKernel(&invalid, &result);
  annc::tensorflow::JitKernelPtr failed_kernel(result.kernel);
  CHECK(result.status == ANNC_JIT_INTERNAL && failed_kernel &&
        !result.kernel_function);
  CHECK(result.error && *result.error);
  const std::string error = result.error;
  failed_kernel.reset();
  auto failure = cache.GetOrCompile(
      {"first"}, [&] { return annc::jit::JitCompileResult{nullptr, error}; });
  CHECK(!failure.ok() && failure.error == error);
  auto compile = [&] { return Compile(request); };
  auto first = cache.GetOrCompile({"first"}, compile);
  if (!first.ok()) std::cerr << first.error << '\n';
  CHECK(first.ok());
  bool recompiled = false;
  {
    auto hit = cache.GetOrCompile({"first"}, [&] {
      recompiled = true;
      return compile();
    });
    CHECK(hit.ok() && hit.executable == first.executable && !recompiled);
  }
  auto second = cache.GetOrCompile({"second"}, compile);
  CHECK(second.ok() && second.executable != first.executable);
  CHECK(ComputesSum(first.executable->kernelFunction()));
  const std::string path = first.executable->workDir();
  CHECK(access(path.c_str(), F_OK) == 0);
  first.executable.reset();
  CHECK(access(path.c_str(), F_OK) != 0 && errno == ENOENT);
  CHECK(ComputesSum(second.executable->kernelFunction()));
  return true;
}
bool CheckPrepackInterface(const char* source) {
  int64_t lhs_shape[] = {2, 4, 73}, rhs_shape[] = {73, 67};
  int64_t out_shape[] = {2, 4, 67};
  const int64_t* dims[] = {lhs_shape, rhs_shape, out_shape};
  size_t ranks[] = {3, 2, 3};
  int64_t immutable = 3;
  AnncJitCompileRequest request{source,  "matmul", 1, dims,       ranks, 3,
                                nullptr, nullptr,  1, &immutable, 1};
  {
    AnncJitKernelResult result{};
    anncCompileJitKernel(&request, &result);
    annc::tensorflow::JitKernelPtr handle(result.kernel);
    CHECK(handle && result.status == ANNC_JIT_INVALID_ARGUMENT);
    CHECK(!result.kernel_function && !result.packed_rhs_count);
  }
  immutable = 1;
  AnncJitKernelResult result{};
  anncCompileJitKernel(&request, &result);
  annc::tensorflow::JitKernelPtr handle(result.kernel);
  if (result.status != ANNC_JIT_OK) std::cerr << result.error << '\n';
  CHECK(result.status == ANNC_JIT_OK && result.kernel_function);
  CHECK(result.packed_rhs_specs && result.packed_rhs_count == 1 &&
        result.prepare_rhs);
  const auto& spec = result.packed_rhs_specs[0];
  CHECK(spec.argument == 1 && spec.k == 73 && spec.n == 67);
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) return 1;
  int64_t shape[] = {1, 5};
  const int64_t* dims[] = {shape, shape, shape};
  size_t ranks[] = {2, 2, 2};
  AnncJitCompileRequest request{argv[1], "add", 1,       dims,
                                ranks,   3,     nullptr, nullptr};
  return CheckInvalidDimensions(request) && CheckCache(request) &&
                 (argc == 2 || CheckPrepackInterface(argv[2]))
             ? 0
             : 1;
}

// Unit tests for the ONNX execution context's failure bookkeeping and the
// description contract.
//
// The fixture suite exercises report_error and the slot contract, but never a
// failing ORT call -- and that is exactly where the first-error rule lives
// (RecordStatus), as does the pool adapter's degrade-to-inline path. A fake
// OrtApi drives the callbacks directly, so those branches are covered without
// a live ORT host.

#include <algorithm>
#include <cstdint>
#include "Context/KernelContextUtils.h"
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <onnxruntime_c_api.h>

#include "context/annc_ep_execution.h"
#include "Kernel/PackedB.h"

namespace annc_ep {
namespace {

// Packed-B description contract: full-shape matching, broadcast slice
// selection, retained for the execution compatibility ABI.
TEST(PackedBContract, MatchesFullShapeAndSelectsBroadcastSlices) {
  float first = 1.0f, second = 2.0f;
  const float* slices[] = {&first, &second};
  AnncPackedB rank4{4, {2, 1, 4, 8}, 2, slices};
  const int64_t shape4[] = {2, 1, 4, 8};
  const int64_t wrong4[] = {1, 2, 4, 8};
  EXPECT_TRUE(annc::kernels::packedBMatches(&rank4, shape4, 4));
  EXPECT_FALSE(annc::kernels::packedBMatches(&rank4, wrong4, 4));
  EXPECT_EQ(annc::kernels::packedBSlice(&rank4, 0, 0), &first);
  EXPECT_EQ(annc::kernels::packedBSlice(&rank4, 1, 0), &second);
  EXPECT_EQ(annc::kernels::packedBSlice(&rank4, 1, 1), nullptr);

  AnncPackedB rank3{3, {2, 4, 8, 0}, 2, slices};
  const int64_t shape3[] = {2, 4, 8};
  EXPECT_TRUE(annc::kernels::packedBMatches(&rank3, shape3, 3));
  EXPECT_EQ(annc::kernels::packedBSlice(&rank3, 1, 0), &second);
  EXPECT_EQ(annc::kernels::packedBSlice(&rank3, 2, 0), nullptr);

  AnncPackedB wrong_count{3, {2, 4, 8, 0}, 1, slices};
  EXPECT_FALSE(annc::kernels::packedBMatches(&wrong_count, shape3, 3));
  const float* null_slice[] = {&first, nullptr};
  AnncPackedB null_entry{3, {2, 4, 8, 0}, 2, null_slice};
  EXPECT_FALSE(annc::kernels::packedBMatches(&null_entry, shape3, 3));
}

TEST(PackedBContract, RejectsEffectiveWeightSizeMismatch) {
  EXPECT_TRUE(annc::kernels::effectiveWeightSizeMatches(16, 16 * sizeof(float)));
  EXPECT_FALSE(annc::kernels::effectiveWeightSizeMatches(15, 16 * sizeof(float)));
  EXPECT_FALSE(annc::kernels::effectiveWeightSizeMatches(
      std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()));
}

// One tagged fake status per call; only identity and lifetime matter here.
struct FakeStatus {
  int tag = 0;
};

const int64_t kNegativeDim[1] = {-1};

// A zeroed OrtApi with just the entries the execution context uses. Handles
// are opaque tags -- the stubs never dereference anything the host passes
// them, exactly like the real API treats them.
struct FakeTensor {
  std::vector<int64_t> dims;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  const void* data = nullptr;
};

struct FakeApi {
  std::vector<FakeTensor> inputs;
  std::vector<size_t> output_indices;
  int kernel_calls = 0;
  OrtApi api{};
  OrtKernelContext* kernel_context = reinterpret_cast<OrtKernelContext*>(0x1);
  void* tensor_data = nullptr;
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

  // Behaviour switches.
  bool fail_get_output = false;
  bool fail_get_tensor_type = false;
  bool fail_get_data = false;
  bool fail_free = false;
  bool fail_parallel_for = false;

  // Bookkeeping: every status handed out is one `made`, every one the context
  // gives back is one `released`.
  std::deque<FakeStatus> statuses;
  int made = 0;
  int released = 0;
  int get_output_calls = 0;
  int shape_info_allocations = 0;
  int shape_reference_calls = 0;
  std::string last_status_message;
  size_t last_output_rank = 0;
  std::vector<int64_t> last_output_dims;

  int mem_info_created = 0;
  int parallel_for_calls = 0;
  size_t last_parallel_total = 0;
  size_t last_parallel_batches = 0;
  int mem_info_released = 0;
  int allocator_requested = 0;
  int allocator_released = 0;
  int alloc_calls = 0;
  int free_calls = 0;
  size_t last_alloc_bytes = 0;
  alignas(16) unsigned char alloc_buffer[256] = {};
  size_t alloc_offset = 0;

  FakeApi() {
    tensor_data = alloc_buffer;
    api.CreateStatus = &CreateStatus;
    api.ReleaseStatus = &ReleaseStatus;
    api.KernelContext_GetOutput = &GetOutput;
    api.KernelContext_GetInputCount = &GetInputCount;
    api.KernelContext_GetInput = &GetInput;
    api.GetTensorData = &GetTensorData;
    api.GetDimensionsCount = &GetDimensionsCount;
    api.GetDimensions = &GetDimensions;
    api.GetTensorTypeAndShape = &GetTensorTypeAndShape;
    api.GetTensorElementTypeAndShapeDataReference = &GetShapeReference;
    api.GetTensorElementType = &GetTensorElementType;
    api.ReleaseTensorTypeAndShapeInfo = &ReleaseTensorTypeAndShapeInfo;
    api.GetTensorMutableData = &GetTensorMutableData;
    api.CreateCpuMemoryInfo = &CreateCpuMemoryInfo;
    api.KernelContext_GetAllocator = &GetAllocator;
    api.AllocatorAlloc = &AllocatorAlloc;
    api.AllocatorFree = &AllocatorFree;
    api.ReleaseAllocator = &ReleaseAllocator;
    api.ReleaseMemoryInfo = &ReleaseMemoryInfo;
    api.KernelContext_ParallelFor = &ParallelFor;
    api.GetErrorMessage = &GetErrorMessage;
  }

  static FakeApi& Current() { return *current_; }
  static FakeApi* current_;

  static OrtStatus* MakeStatus() {
    FakeApi& self = Current();
    self.statuses.push_back(FakeStatus{++self.made});
    return reinterpret_cast<OrtStatus*>(&self.statuses.back());
  }

  static OrtStatus* ORT_API_CALL CreateStatus(OrtErrorCode,
                                              const char* message) noexcept {
    FakeApi& self = Current();
    self.last_status_message = message != nullptr ? message : "";
    return MakeStatus();
  }
  static void ORT_API_CALL ReleaseStatus(OrtStatus*) noexcept {
    ++Current().released;
  }
  static OrtStatus* ORT_API_CALL GetOutput(OrtKernelContext*, size_t index,
                                           const int64_t* dims, size_t rank,
                                           OrtValue** out) noexcept {
    FakeApi& self = Current();
    ++self.get_output_calls;
    self.last_output_rank = rank;
    self.last_output_dims.clear();
    if (rank) self.last_output_dims.assign(dims, dims + rank);
    if (self.fail_get_output) return MakeStatus();
    self.output_indices.push_back(index);
    *out = reinterpret_cast<OrtValue*>(0x2);
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL GetTensorTypeAndShape(
      const OrtValue* value, OrtTensorTypeAndShapeInfo** out) noexcept {
    FakeApi& self = Current();
    if (self.fail_get_tensor_type) return MakeStatus();
    ++self.shape_info_allocations;
    *out = reinterpret_cast<OrtTensorTypeAndShapeInfo*>(
        const_cast<OrtValue*>(value));
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL GetShapeReference(
      const OrtValue* value, ONNXTensorElementDataType* type,
      const int64_t** dims, size_t* rank) noexcept {
    FakeApi& self = Current();
    ++self.shape_reference_calls;
    if (self.fail_get_tensor_type) return MakeStatus();
    auto* input = Input(value);
    *type = input ? input->type : self.element_type;
    const auto& shape = input ? input->dims : self.last_output_dims;
    *dims = shape.data();
    *rank = shape.size();
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL GetTensorElementType(
      const OrtTensorTypeAndShapeInfo* info,
      ONNXTensorElementDataType* out) noexcept {
    auto* input = Input(reinterpret_cast<const OrtValue*>(info));
    *out = input ? input->type : Current().element_type;
    return nullptr;
  }
  static void ORT_API_CALL ReleaseTensorTypeAndShapeInfo(
      OrtTensorTypeAndShapeInfo*) noexcept {}
  static OrtStatus* ORT_API_CALL GetTensorMutableData(OrtValue*,
                                                      void** out) noexcept {
    FakeApi& self = Current();
    if (self.fail_get_data) return MakeStatus();
    *out = std::find(self.last_output_dims.begin(), self.last_output_dims.end(), 0)
                   != self.last_output_dims.end() ? nullptr : self.tensor_data;
    return nullptr;
  }

  static const FakeTensor* Input(const OrtValue* value) {
    if (value == reinterpret_cast<const OrtValue*>(0x2)) return nullptr;
    return reinterpret_cast<const FakeTensor*>(value);
  }
  static OrtStatus* ORT_API_CALL GetInputCount(const OrtKernelContext*,
                                               size_t* out) noexcept {
    *out = Current().inputs.size();
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL GetInput(const OrtKernelContext*, size_t index,
                                         const OrtValue** out) noexcept {
    *out = reinterpret_cast<const OrtValue*>(&Current().inputs.at(index));
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL GetTensorData(const OrtValue* value,
                                              const void** out) noexcept {
    *out = Input(value)->data;
    return nullptr;
  }
  static const std::vector<int64_t>& Shape(const OrtTensorTypeAndShapeInfo* info) {
    auto* input = Input(reinterpret_cast<const OrtValue*>(info));
    return input ? input->dims : Current().last_output_dims;
  }
  static OrtStatus* ORT_API_CALL GetDimensionsCount(
      const OrtTensorTypeAndShapeInfo* info, size_t* out) noexcept {
    *out = Shape(info).size();
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL GetDimensions(
      const OrtTensorTypeAndShapeInfo* info, int64_t* out, size_t rank) noexcept {
    std::copy_n(Shape(info).data(), rank, out);
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateCpuMemoryInfo(OrtAllocatorType,
                                                     OrtMemType,
                                                     OrtMemoryInfo** out) noexcept {
    ++Current().mem_info_created;
    *out = reinterpret_cast<OrtMemoryInfo*>(0x4);
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL GetAllocator(const OrtKernelContext*,
                                              const OrtMemoryInfo*,
                                              OrtAllocator** out) noexcept {
    ++Current().allocator_requested;
    *out = reinterpret_cast<OrtAllocator*>(0x5);
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL AllocatorAlloc(OrtAllocator* allocator,
                                                size_t size,
                                                void** out) noexcept {
    FakeApi& self = Current();
    ++self.alloc_calls;
    self.last_alloc_bytes = size;
    (void)allocator;
    if (size == 0 || self.alloc_offset + size > sizeof(self.alloc_buffer)) {
      return MakeStatus();
    }
    *out = self.alloc_buffer + self.alloc_offset;
    self.alloc_offset += size;
    return nullptr;
  }
  static OrtStatus* ORT_API_CALL AllocatorFree(OrtAllocator*, void*) noexcept {
    FakeApi& self = Current();
    ++self.free_calls;
    if (self.fail_free) return MakeStatus();
    return nullptr;
  }
  static void ORT_API_CALL ReleaseAllocator(OrtAllocator*) noexcept {
    ++Current().allocator_released;
  }
  static void ORT_API_CALL ReleaseMemoryInfo(OrtMemoryInfo*) noexcept {
    ++Current().mem_info_released;
  }

  static const char* ORT_API_CALL GetErrorMessage(const OrtStatus*) noexcept {
    return "fake failure";
  }
  // Dispatches the way ORT's pool does: one callback per index, in order, on
  // this thread.
  static OrtStatus* ORT_API_CALL ParallelFor(const OrtKernelContext*,
                                             void (*fn)(void*, size_t),
                                             size_t total, size_t num_batch,
                                             void* usr_data) noexcept {
    FakeApi& self = Current();
    ++self.parallel_for_calls;
    self.last_parallel_total = total;
    self.last_parallel_batches = num_batch;
    if (self.fail_parallel_for) return MakeStatus();
    for (size_t i = 0; i < total; ++i) fn(usr_data, i);
    return nullptr;
  }
};

FakeApi* FakeApi::current_ = nullptr;

int TagOf(OrtStatus* status) {
  return reinterpret_cast<FakeStatus*>(status)->tag;
}

// Drive the C services and expose their returned view to the legacy assertions.
AnncStatusCode AllocateOutput(AnncKernelContext* context, uint32_t slot,
                               AnncTensorDesc* desc) {
  AnncTensorView view{};
  auto code = context->services->allocate_output(
      context->services->opaque, slot, desc, &view);
  if (code == ANNC_STATUS_OK) desc->data = view.aligned;
  return code;
}
AnncStatusCode AllocateTemp(AnncKernelContext* context, AnncTensorDesc* desc) {
  AnncTensorView view{};
  auto code = context->services->allocate_temp(
      context->services->opaque, desc, &view);
  if (code == ANNC_STATUS_OK) desc->data = view.aligned;
  return code;
}

class ExecutionContextFailureTest : public ::testing::Test {
 protected:
  void SetUp() override { FakeApi::current_ = &fake_; }
  void TearDown() override { FakeApi::current_ = nullptr; }

  FakeApi fake_;
  std::vector<OutputContract> contracts_{{3, ANNC_ELEMENT_TYPE_F32}};
};

TEST_F(ExecutionContextFailureTest, KeepsTheFirstOrtStatusAndReleasesLaterOnes) {
  OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
  AnncKernelContext* abi = context.abi();
  const int64_t dims[3] = {2, 3, 5};
  AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 3,
                      ANNC_ALLOCATION_NONE, dims, nullptr};

  fake_.fail_get_output = true;
  // The first failing ORT call becomes the run's error.
  EXPECT_EQ(AllocateOutput(abi, 0, &desc),
            ANNC_STATUS_ALLOCATION_FAILED);
  EXPECT_EQ(fake_.made, 1);
  EXPECT_EQ(fake_.released, 0);

  // A second one must not replace (or leak) the first: the newer status is
  // released, the recorded one stays. A first-error branch that recursed
  // instead of storing would never return from the call above.
  EXPECT_EQ(AllocateOutput(abi, 0, &desc),
            ANNC_STATUS_ALLOCATION_FAILED);
  EXPECT_EQ(fake_.made, 2);
  EXPECT_EQ(fake_.released, 1);

  OrtStatus* error = context.TakeError();
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(TagOf(error), 1);
  EXPECT_EQ(context.TakeError(), nullptr);
  fake_.api.ReleaseStatus(error);
}

TEST_F(ExecutionContextFailureTest, LaterFailureDoesNotReplaceTheFirst) {
  OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
  AnncKernelContext* abi = context.abi();
  const int64_t dims[3] = {2, 3, 5};
  AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 3,
                      ANNC_ALLOCATION_NONE, dims, nullptr};

  // GetOutput succeeds, reading the tensor's element type fails.
  fake_.fail_get_tensor_type = true;
  EXPECT_EQ(AllocateOutput(abi, 0, &desc), ANNC_STATUS_INTERNAL);
  fake_.fail_get_tensor_type = false;
  fake_.fail_get_data = true;
  EXPECT_EQ(AllocateOutput(abi, 0, &desc), ANNC_STATUS_INTERNAL);

  OrtStatus* error = context.TakeError();
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(TagOf(error), 1);
  fake_.api.ReleaseStatus(error);
}

TEST_F(ExecutionContextFailureTest, ContractViolationsDoNotTouchOrt) {
  OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
  AnncKernelContext* abi = context.abi();
  const int64_t dims[3] = {2, 3, 5};

  // Wrong rank: rejected by the contract check, before any ORT call.
  AnncTensorDesc wrong_rank{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 2,
                            ANNC_ALLOCATION_NONE, dims, nullptr};
  EXPECT_EQ(AllocateOutput(abi, 0, &wrong_rank),
            ANNC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fake_.get_output_calls, 0);
  EXPECT_EQ(fake_.last_status_message.find("rank mismatch") != std::string::npos,
            true);
}

TEST_F(ExecutionContextFailureTest, ScalarOutputAndInvalidDimensions) {
  // The ranks match their contracts, so the dims check is reached rather than
  // short-circuited by the rank check.
  {
    // Common context permits a null dimensions pointer for a scalar.
    std::vector<OutputContract> rank0{{0, ANNC_ELEMENT_TYPE_F32}};
    OnnxExecutionContext context(&fake_.api, fake_.kernel_context, rank0);
    AnncTensorDesc no_dims{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 0,
                           ANNC_ALLOCATION_NONE, nullptr, nullptr};
    EXPECT_EQ(AllocateOutput(context.abi(), 0, &no_dims), ANNC_STATUS_OK);
  }
  {
    std::vector<OutputContract> rank1{{1, ANNC_ELEMENT_TYPE_F32}};
    OnnxExecutionContext context(&fake_.api, fake_.kernel_context, rank1);
    AnncTensorDesc negative{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 1,
                            ANNC_ALLOCATION_NONE, kNegativeDim, nullptr};
    EXPECT_EQ(AllocateOutput(context.abi(), 0, &negative),
              ANNC_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(fake_.last_status_message.find("invalid output request") !=
                  std::string::npos,
              true);
  }
  EXPECT_EQ(fake_.get_output_calls, 1);
}

TEST_F(ExecutionContextFailureTest, ZeroSizeOutputPassesTheRuntimeDims) {
  OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
  AnncKernelContext* abi = context.abi();
  const int64_t dims[3] = {0, 3, 5};  // batch 0: legal, empty tensor
  AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 3,
                      ANNC_ALLOCATION_NONE, dims, nullptr};

  EXPECT_EQ(AllocateOutput(abi, 0, &desc), ANNC_STATUS_OK);
  EXPECT_EQ(fake_.get_output_calls, 1);
  EXPECT_EQ(fake_.last_output_rank, 3u);
  EXPECT_EQ(fake_.last_output_dims, (std::vector<int64_t>{0, 3, 5}));
  EXPECT_EQ(desc.data, nullptr);  // ORT hands back a null data pointer
  EXPECT_TRUE(context.AllOutputsAllocated());
  EXPECT_EQ(context.TakeError(), nullptr);
}

TEST_F(ExecutionContextFailureTest, OutputTypeCheckBorrowsShapeMetadata) {
  OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
  const int64_t dims[3] = {2, 3, 5};
  AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 3,
                      ANNC_ALLOCATION_NONE, dims, nullptr};
  EXPECT_EQ(AllocateOutput(context.abi(), 0, &desc),
            ANNC_STATUS_OK);
  EXPECT_EQ(fake_.shape_reference_calls, 1);
  EXPECT_EQ(fake_.shape_info_allocations, 0);
}

TEST_F(ExecutionContextFailureTest, ZeroByteTemporaryNeedsNoAllocator) {
  OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
  AnncKernelContext* abi = context.abi();
  const int64_t dims[2] = {0, 5};
  AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 2,
                      ANNC_ALLOCATION_ZERO, dims, nullptr};

  EXPECT_EQ(AllocateTemp(abi, &desc), ANNC_STATUS_OK);
  EXPECT_EQ(desc.data, nullptr);
  // Nothing was requested from ORT: an empty scratch buffer is not allocated.
  EXPECT_EQ(fake_.allocator_requested, 0);
  EXPECT_EQ(fake_.alloc_calls, 0);
}

TEST_F(ExecutionContextFailureTest, TemporaryIsAllocatedAndFreedThroughOrt) {
  {
    OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
    AnncKernelContext* abi = context.abi();
    const int64_t dims[2] = {4, 5};
    AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 2,
                        ANNC_ALLOCATION_ZERO, dims, nullptr};

    EXPECT_EQ(AllocateTemp(abi, &desc), ANNC_STATUS_OK);
    EXPECT_EQ(fake_.alloc_calls, 1);
    EXPECT_EQ(fake_.last_alloc_bytes, 20u * sizeof(float));
    EXPECT_NE(desc.data, nullptr);
    EXPECT_EQ(fake_.mem_info_created, 1);
    EXPECT_EQ(fake_.allocator_requested, 1);
    EXPECT_EQ(fake_.free_calls, 0);
  }  // the destructor frees the temporary and releases the handles

  EXPECT_EQ(fake_.free_calls, 1);
  EXPECT_EQ(fake_.allocator_released, 1);
  EXPECT_EQ(fake_.mem_info_released, 1);
}

TEST_F(ExecutionContextFailureTest, FreeFailureStatusIsReleasedNotDropped) {
  {
    OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
    AnncKernelContext* abi = context.abi();
    const int64_t dims[2] = {4, 5};
    AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 2,
                        ANNC_ALLOCATION_NONE, dims, nullptr};
    ASSERT_EQ(AllocateTemp(abi, &desc), ANNC_STATUS_OK);
    fake_.fail_free = true;
  }
  EXPECT_EQ(fake_.free_calls, 1);
  EXPECT_EQ(fake_.released, 1);  // the failing free returned a status; released
  EXPECT_EQ(fake_.allocator_released, 1);
  EXPECT_EQ(fake_.mem_info_released, 1);
}

TEST_F(ExecutionContextFailureTest, ParallelForFailureRunsTheRangeInline) {
  OrtThreadPool pool(&fake_.api, fake_.kernel_context);
  fake_.fail_parallel_for = true;
  std::vector<int64_t> begins;
  std::vector<int64_t> ends;

  // Cost well past Eigen's startup+per-thread cycles, so the range really is
  // handed to the pool -- otherwise the adapter would run it inline and never
  // exercise the failure path.
  annc::threadpool::ParallelForOptions options;
  options.cost_per_unit = 1;
  pool.parallel_for(300000, options, [&](int64_t begin, int64_t end) {
    begins.push_back(begin);
    ends.push_back(end);
  });

  // parallel_for must not fail: when the pool call fails, the whole range ran
  // once on this thread and the failing status was released, not dropped.
  EXPECT_EQ(fake_.parallel_for_calls, 1);
  ASSERT_EQ(begins.size(), 1u);
  EXPECT_EQ(begins[0], 0);
  EXPECT_EQ(ends[0], 300000);
  EXPECT_EQ(fake_.released, 1);
}

// The shim sizes the batches the way ORT sizes the blocks of its own kernels:
// (total * cost - 100000) / 100000 + 0.9 decides whether to parallelise at all
// and 1 means the range runs on the calling thread with no dispatch.
TEST_F(ExecutionContextFailureTest, CostBelowTheStartupThresholdRunsInline) {
  OrtThreadPool pool(&fake_.api, fake_.kernel_context);
  annc::threadpool::ParallelForOptions options;
  options.cost_per_unit = 8;  // e.g. 1600 rows x 8 columns of a masked mul
  std::vector<int64_t> begins;
  std::vector<int64_t> ends;

  pool.parallel_for(1600, options, [&](int64_t begin, int64_t end) {
    begins.push_back(begin);
    ends.push_back(end);
  });

  EXPECT_EQ(fake_.parallel_for_calls, 0);  // 12800 cycles < kStartupCycles
  EXPECT_EQ(begins, (std::vector<int64_t>{0}));
  EXPECT_EQ(ends, (std::vector<int64_t>{1600}));
}

TEST_F(ExecutionContextFailureTest, CostAboveTheThresholdSetsTheShardCount) {
  OrtThreadPool pool(&fake_.api, fake_.kernel_context);
  annc::threadpool::ParallelForOptions options;
  options.cost_per_unit = 128;
  std::vector<int64_t> shard_sizes;

  // The cost exceeds the startup threshold. Request two shards explicitly
  // so this test checks the limit independently of the machine's CPU count.
  options.max_parallelism = 2;
  pool.parallel_for(4096, options, [&](int64_t begin, int64_t end) {
    shard_sizes.push_back(end - begin);
  });

  EXPECT_EQ(fake_.parallel_for_calls, 1);
  EXPECT_EQ(fake_.last_parallel_total, 2u);
  EXPECT_EQ(fake_.last_parallel_batches, 2u);
  ASSERT_EQ(shard_sizes.size(), 2u);
  int64_t covered = 0;
  for (int64_t size : shard_sizes) {
    EXPECT_GT(size, 0);
    covered += size;
  }
  EXPECT_EQ(covered, 4096);  // every unit lands in exactly one shard
}

TEST_F(ExecutionContextFailureTest, ParallelForShardsAndFlagsTheRegion) {
  OrtThreadPool pool(&fake_.api, fake_.kernel_context);
  annc::threadpool::ParallelForOptions options;
  options.grain_size = 2;
  std::vector<int64_t> shard_sizes;
  bool in_region_during_callbacks = false;

  EXPECT_FALSE(pool.in_parallel_region());
  pool.parallel_for(4, options, [&](int64_t begin, int64_t end) {
    in_region_during_callbacks = in_region_during_callbacks ||
                                 pool.in_parallel_region();
    shard_sizes.push_back(end - begin);
  });

  // One ORT batch per shard: the callback receives a batch index and turns it
  // into the range it covers, so grain_size 2 over 4 units arrives as the two
  // half ranges -- not four one-unit callbacks (which is what ORT's per-unit
  // dispatch would do, and what cost ~1.4us of dispatch per unit on the target
  // models).  The nesting gate is up while the callbacks run and down after.
  EXPECT_EQ(shard_sizes, (std::vector<int64_t>{2, 2}));
  EXPECT_TRUE(in_region_during_callbacks);
  EXPECT_FALSE(pool.in_parallel_region());
  EXPECT_EQ(fake_.last_parallel_total, 2u);
  EXPECT_EQ(fake_.last_parallel_batches, 2u);
}

// With neither signal the range is split the way the in-tree pool's grain
// default (1) and ORT's own num_batches<=0 default split it: one unit per
// block, bounded by the pool's degree.
TEST_F(ExecutionContextFailureTest, NoCostSignalUsesPoolDegreeAndCoversRange) {
  OrtThreadPool pool(&fake_.api, fake_.kernel_context);
  std::vector<int64_t> shard_sizes;
  pool.parallel_for(64, annc::threadpool::ParallelForOptions{},
                    [&](int64_t begin, int64_t end) {
                      shard_sizes.push_back(end - begin);
                    });
  EXPECT_EQ(fake_.parallel_for_calls, 1);
  const auto expected_shards = static_cast<size_t>(std::min(64, pool.num_threads()));
  EXPECT_EQ(shard_sizes.size(), expected_shards);
  EXPECT_EQ(fake_.last_parallel_batches, expected_shards);
  int64_t covered = 0;
  for (int64_t size : shard_sizes) {
    EXPECT_GT(size, 0);
    covered += size;
  }
  EXPECT_EQ(covered, 64);
}

TEST_F(ExecutionContextFailureTest, TemporaryWithoutDimsIsRejected) {
  OnnxExecutionContext context(&fake_.api, fake_.kernel_context, contracts_);
  AnncKernelContext* abi = context.abi();
  AnncTensorDesc desc{sizeof(AnncTensorDesc), ANNC_ELEMENT_TYPE_F32, 1,
                      ANNC_ALLOCATION_NONE, nullptr, nullptr};

  EXPECT_EQ(AllocateTemp(abi, &desc),
            ANNC_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(fake_.last_status_message.find("invalid temporary request") !=
                std::string::npos,
            true);
  EXPECT_EQ(fake_.alloc_calls, 0);
}

TEST_F(ExecutionContextFailureTest, ServiceViewsOwnShapeAndKeepRequestImmutable) {
  OnnxExecutionContext owner(&fake_.api, fake_.kernel_context, contracts_);
  auto* abi = owner.abi();
  int64_t dims[] = {2, 3, 5};
  AnncTensorDesc request{sizeof(request), ANNC_ELEMENT_TYPE_F32, 3, 0, dims, nullptr};
  AnncTensorView output{};
  ASSERT_EQ(abi->services->allocate_output(abi->services->opaque, 0, &request,
                                           &output), ANNC_STATUS_OK);
  EXPECT_EQ(request.data, nullptr);
  dims[0] = 99;
  EXPECT_EQ(output.sizes[0], 2);
  EXPECT_EQ(output.strides[0], 15);
  EXPECT_EQ(abi->services->native_thread_pool(abi->services->opaque, "unknown"),
            nullptr);
}

TEST_F(ExecutionContextFailureTest, ScalarTempAndExplicitShardLimit) {
  OnnxExecutionContext owner(&fake_.api, fake_.kernel_context, contracts_);
  annc::kernels::KernelContext context(owner.abi());
  AnncTensorView temp{};
  ASSERT_EQ((context.allocateTemp<float, 0>(nullptr, &temp)), ANNC_STATUS_OK);
  EXPECT_EQ(fake_.last_alloc_bytes, sizeof(float));
  int64_t covered = 0;
  auto pool = context.getThreadPool();
  EXPECT_EQ(annc::kernels::Shard(2, pool, 300000, 128,
                  [&](int64_t begin, int64_t end) { covered += end - begin; }),
            ANNC_STATUS_OK);
  EXPECT_EQ(covered, 300000);
  EXPECT_LE(fake_.last_parallel_batches, 2u);
}

TEST_F(ExecutionContextFailureTest, PacksSlotsScalarAttributesAndOutputMapping) {
  float first[] = {1, 2, 3, 4}, second[] = {5, 6, 7, 8}, eps = 0.125f;
  fake_.inputs = {{{2, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, first},
                  {{2, 2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, second},
                  {{}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &eps}};
  ContextCall request;
  request.arg_indices = {1, 0, 2};  // Equal rank and dtype must not swap slots.
  request.arg_ranks = {2, 2, 0};
  request.arg_types.assign(3, ANNC_ELEMENT_TYPE_F32);
  request.outputs = {{2, ANNC_ELEMENT_TYPE_F32}};
  request.output_slots = {3};
  request.i64_attrs = {17, -4};
  // Runtime fusion resources use their own array and survive multiple calls.
  const int64_t instance_value = 91;
  request.resources = {{sizeof(AnncResourceView), 1, "test.instance",
                        &instance_value, sizeof(instance_value)}};
  request.symbol = reinterpret_cast<void*>(+[](AnncKernelContext* raw) {
    ++FakeApi::Current().kernel_calls;
    EXPECT_EQ(raw->num_attrs, 2u);
    EXPECT_EQ(raw->num_resources, 1u);
    return annc::kernels::invokeContextKernel(raw, [](auto& context) {
      const auto* a = context.getInput(0);
      const auto* b = context.getInput(1);
      const auto* eps = context.getInput(2);
      EXPECT_EQ(a->rank, 2u);
      EXPECT_EQ(a->strides[0], 2);
      EXPECT_EQ(eps->rank, 0u);
      EXPECT_EQ(*annc::kernels::contextData<float>(*a), 5);
      EXPECT_EQ(*annc::kernels::contextData<float>(*b), 1);
      EXPECT_EQ(*annc::kernels::contextData<float>(*eps), 0.125f);
      EXPECT_EQ(*context.getI64Attr(0), 17);
      EXPECT_EQ(*context.getI64Attr(1), -4);
      const auto* resource = static_cast<const int64_t*>(
          context.getResource("test.instance", 1, sizeof(int64_t)));
      EXPECT_NE(resource, nullptr);
      if (!resource) return ANNC_STATUS_INVALID_ARGUMENT;
      EXPECT_EQ(*resource, 91);
      auto output = context.template allocateOutput<float>(0, {2, 2});
      if (!output) return context.status();
      auto* out = annc::kernels::contextData<float>(*output);
      for (int i = 0; i < 4; ++i)
        out[i] = annc::kernels::contextData<float>(*a)[i] +
                 annc::kernels::contextData<float>(*b)[i];
      return ANNC_STATUS_OK;
    });
  });
  // Both public shape APIs must construct the same call context.
  for (bool borrowed : {true, false}) {
    fake_.api.GetTensorElementTypeAndShapeDataReference =
        borrowed ? &FakeApi::GetShapeReference : nullptr;
    EXPECT_EQ(ComputeWithContext(&fake_.api, fake_.kernel_context, request), nullptr);
    const auto* out = reinterpret_cast<const float*>(fake_.alloc_buffer);
    EXPECT_EQ((std::vector<float>(out, out + 4)),
              (std::vector<float>{6, 8, 10, 12}));
  }
  EXPECT_EQ(fake_.kernel_calls, 2);
  EXPECT_EQ(fake_.output_indices, (std::vector<size_t>{3, 3}));
}

TEST_F(ExecutionContextFailureTest, ContextArraysSupportMoreThanFourteenInputs) {
  float value = 1;
  fake_.inputs.assign(16, FakeTensor{{}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value});
  ContextCall request;
  request.arg_ranks.assign(16, 0);
  request.arg_types.assign(16, ANNC_ELEMENT_TYPE_F32);
  for (int i = 0; i < 16; ++i) request.arg_indices.push_back(i);
  request.symbol = reinterpret_cast<void*>(+[](AnncKernelContext* raw) -> AnncStatusCode {
    ++FakeApi::Current().kernel_calls;
    EXPECT_EQ(raw->num_inputs, 16u);
    EXPECT_EQ(raw->inputs[15].rank, 0u);
    return ANNC_STATUS_OK;
  });
  EXPECT_EQ(ComputeWithContext(&fake_.api, fake_.kernel_context, request), nullptr);
  EXPECT_EQ(fake_.kernel_calls, 1);
}

TEST_F(ExecutionContextFailureTest, InvalidInputContractDoesNotInvokeKernel) {
  float value = 1;
  fake_.inputs = {{{1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value}};
  ContextCall request;
  request.arg_indices = {0};
  request.arg_ranks = {1};
  request.arg_types = {ANNC_ELEMENT_TYPE_F32};
  request.symbol = reinterpret_cast<void*>(+[](AnncKernelContext*) -> AnncStatusCode {
    ++FakeApi::Current().kernel_calls;
    return ANNC_STATUS_OK;
  });
  for (int kind = 0; kind < 4; ++kind) {
    auto bad = request;
    if (kind == 0) bad.arg_indices[0] = 1;
    if (kind == 1) bad.arg_ranks[0] = 2;
    if (kind == 2) bad.arg_types[0] = ANNC_ELEMENT_TYPE_I64;
    if (kind == 3) fake_.inputs[0].data = nullptr;
    auto* error = ComputeWithContext(&fake_.api, fake_.kernel_context, bad);
    ASSERT_NE(error, nullptr);
    fake_.api.ReleaseStatus(error);
  }
  EXPECT_EQ(fake_.kernel_calls, 0);
}

TEST_F(ExecutionContextFailureTest, MissingOutputAndReportedErrorReachCaller) {
  ContextCall request;
  request.outputs = contracts_;
  request.symbol = reinterpret_cast<void*>(+[](AnncKernelContext*) -> AnncStatusCode {
    return ANNC_STATUS_OK;
  });
  auto* missing = ComputeWithContext(&fake_.api, fake_.kernel_context, request);
  ASSERT_NE(missing, nullptr);
  EXPECT_NE(fake_.last_status_message.find("did not allocate"), std::string::npos);
  fake_.api.ReleaseStatus(missing);
  request.symbol = reinterpret_cast<void*>(+[](AnncKernelContext* raw) -> AnncStatusCode {
    raw->services->report_error(raw->services->opaque, ANNC_STATUS_INTERNAL,
                                "kernel's first error");
    return ANNC_STATUS_OK;  // A recorded error wins over an accidental OK return.
  });
  auto* reported = ComputeWithContext(&fake_.api, fake_.kernel_context, request);
  ASSERT_NE(reported, nullptr);
  EXPECT_EQ(fake_.last_status_message,
            "ANNC EP execution context: kernel's first error");
  fake_.api.ReleaseStatus(reported);
}

TEST_F(ExecutionContextFailureTest, AllocatedButUnboundOutputIsRejected) {
  ContextCall request;
  request.outputs.assign(2, {1, ANNC_ELEMENT_TYPE_F32});
  request.symbol =
      reinterpret_cast<void*>(+[](AnncKernelContext* raw) -> AnncStatusCode {
        const auto unbound_slot =
            *static_cast<const int64_t*>(raw->attrs[0].data);
        const int64_t dims[] = {2};
        AnncTensorDesc desc{sizeof(desc), ANNC_ELEMENT_TYPE_F32, 1, 0, dims,
                            nullptr};
        for (uint32_t slot = 0; slot < raw->num_outputs; ++slot) {
          AnncTensorView local{};
          auto code = raw->services->allocate_output(raw->services->opaque,
                                                     slot, &desc, &local);
          if (code != ANNC_STATUS_OK) return code;
          if (slot != unbound_slot) raw->outputs[slot] = local;
        }
        return ANNC_STATUS_OK;
      });
  for (int64_t slot : {0, 1}) {
    SCOPED_TRACE(slot);
    request.i64_attrs = {slot};
    auto* error = ComputeWithContext(&fake_.api, fake_.kernel_context, request);
    ASSERT_NE(error, nullptr);
    EXPECT_NE(fake_.last_status_message.find("did not bind every output"),
              std::string::npos);
    fake_.api.ReleaseStatus(error);
  }
  EXPECT_EQ(fake_.get_output_calls, 4);
}

TEST_F(ExecutionContextFailureTest, ModifiedOutputViewIsRejected) {
  ContextCall request;
  request.outputs = {{1, ANNC_ELEMENT_TYPE_F32}};
  request.symbol =
      reinterpret_cast<void*>(+[](AnncKernelContext* raw) -> AnncStatusCode {
        static const int64_t dims[] = {2}, strides[] = {1};
        AnncTensorDesc desc{sizeof(desc), ANNC_ELEMENT_TYPE_F32, 1, 0, dims,
                            nullptr};
        auto& view = raw->outputs[0];
        auto code = raw->services->allocate_output(raw->services->opaque, 0,
                                                   &desc, &view);
        if (code != ANNC_STATUS_OK) return code;
        switch (*static_cast<const int64_t*>(raw->attrs[0].data)) {
          case 0:
            view.allocated = nullptr;
            break;
          case 1:
            view.aligned = nullptr;
            break;
          case 2:
            view.offset = 1;
            break;
          case 3:
            view.element_type = ANNC_ELEMENT_TYPE_I32;
            break;
          case 4:
            view.rank = 0;
            break;
          // Even equal metadata must retain the host-owned pointers.
          case 5:
            view.sizes = dims;
            break;
          case 6:
            view.strides = strides;
            break;
          case 7:
            view.layout = ANNC_LAYOUT_STRIDED + 1;
            break;
          case 8:
            view.flags &= ~ANNC_TENSOR_BOUND;
            break;
          case 9:
            view.flags |= ANNC_TENSOR_READ_ONLY;
            break;
        }
        return ANNC_STATUS_OK;
      });
  const char* fields[] = {"allocated", "aligned",  "offset",  "element_type",
                          "rank",      "sizes",    "strides", "layout",
                          "bound",     "read_only"};
  for (int64_t field = 0; field < 10; ++field) {
    SCOPED_TRACE(fields[field]);
    request.i64_attrs = {field};
    auto* error = ComputeWithContext(&fake_.api, fake_.kernel_context, request);
    ASSERT_NE(error, nullptr);
    EXPECT_NE(fake_.last_status_message.find("did not bind every output"),
              std::string::npos);
    fake_.api.ReleaseStatus(error);
  }
}

TEST_F(ExecutionContextFailureTest,
       ScalarAndEmptyOutputsCanBindInReverseOrder) {
  ContextCall request;
  request.outputs = {{0, ANNC_ELEMENT_TYPE_F32}, {1, ANNC_ELEMENT_TYPE_F32}};
  request.output_slots = {3, 5};
  request.symbol =
      reinterpret_cast<void*>(+[](AnncKernelContext* raw) -> AnncStatusCode {
        const int64_t dims[] = {0};
        AnncTensorDesc desc{sizeof(desc), ANNC_ELEMENT_TYPE_F32, 1, 0, dims,
                            nullptr};
        auto& fake = FakeApi::Current();
        fake.tensor_data =
            nullptr;  // A bound empty tensor may have no storage.
        auto code = raw->services->allocate_output(raw->services->opaque, 1,
                                                   &desc, &raw->outputs[1]);
        if (code != ANNC_STATUS_OK) return code;
        fake.tensor_data = fake.alloc_buffer;
        desc.rank = 0;
        desc.dims = nullptr;
        AnncTensorView scalar{};
        code = raw->services->allocate_output(raw->services->opaque, 0, &desc,
                                              &scalar);
        if (code != ANNC_STATUS_OK) return code;
        raw->outputs[0] = scalar;  // Explicit publication is also valid.
        return ANNC_STATUS_OK;
      });
  EXPECT_EQ(ComputeWithContext(&fake_.api, fake_.kernel_context, request),
            nullptr);
  EXPECT_EQ(fake_.output_indices, (std::vector<size_t>{5, 3}));
}

// The argument resolver is pure, so its rules are pinned directly -- inputs
// shaped like the real fused node's, no ORT host involved.
TEST(ArgumentBindingTest, BindsRealSiteOperandsAndIgnoresShapeConstants) {
  // The 16-node FTDMM fused node, as ANNC_EP_DEBUG reports it: x(rank 3, f32),
  // three rank-1 i32 shape constants, one rank-1 i64 constant, w(rank 2, f32).
  const std::vector<InputShapeKey> inputs = {
      {3, ANNC_ELEMENT_TYPE_F32}, {1, ANNC_ELEMENT_TYPE_I32},
      {1, ANNC_ELEMENT_TYPE_I32}, {1, ANNC_ELEMENT_TYPE_I32},
      {1, ANNC_ELEMENT_TYPE_I64}, {2, ANNC_ELEMENT_TYPE_F32}};

  const ArgumentBinding binding =
      BindArgumentsToInputs(inputs, {3, 2},
                            {ANNC_ELEMENT_TYPE_F32, ANNC_ELEMENT_TYPE_F32});

  EXPECT_EQ(binding.indices, (std::vector<int64_t>{0, 5}));
  EXPECT_EQ(binding.ambiguous_arg, -1);
}

TEST(ArgumentBindingTest, ElementTypeDisambiguatesASharedRank) {
  // A rank-3 i64 tensor (a shape-chain remnant) beside the rank-3 f32 operand
  // is not a candidate, and does not make the binding ambiguous.
  const std::vector<InputShapeKey> inputs = {{3, ANNC_ELEMENT_TYPE_I64},
                                             {3, ANNC_ELEMENT_TYPE_F32},
                                             {2, ANNC_ELEMENT_TYPE_F32}};

  const ArgumentBinding binding =
      BindArgumentsToInputs(inputs, {3, 2},
                            {ANNC_ELEMENT_TYPE_F32, ANNC_ELEMENT_TYPE_F32});

  EXPECT_EQ(binding.indices, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(binding.ambiguous_arg, -1);
}

TEST(ArgumentBindingTest, TwoInputsFittingOneArgumentAreAmbiguous) {
  const std::vector<InputShapeKey> inputs = {{3, ANNC_ELEMENT_TYPE_F32},
                                             {3, ANNC_ELEMENT_TYPE_F32}};

  const ArgumentBinding binding =
      BindArgumentsToInputs(inputs, {3}, {ANNC_ELEMENT_TYPE_F32});

  // Which tensor the kernel should read is undecidable; the caller refuses
  // rather than let first-match pick one.
  EXPECT_EQ(binding.ambiguous_arg, 0);
  EXPECT_EQ(binding.ambiguous_inputs[0], 0);
  EXPECT_EQ(binding.ambiguous_inputs[1], 1);
}

TEST(ArgumentBindingTest, AnUnfilledArgumentStaysMinusOne) {
  const std::vector<InputShapeKey> inputs = {{3, ANNC_ELEMENT_TYPE_F32}};

  const ArgumentBinding binding =
      BindArgumentsToInputs(inputs, {3, 2},
                            {ANNC_ELEMENT_TYPE_F32, ANNC_ELEMENT_TYPE_F32});

  EXPECT_EQ(binding.indices, (std::vector<int64_t>{0, -1}));
  EXPECT_EQ(binding.ambiguous_arg, -1);
}

TEST(ArgumentBindingTest, AnUnmappableElementTypeFitsNothing) {
  // Element type 0 is the resolver's "no ANNC counterpart"; it must never
  // satisfy a declared argument.
  const std::vector<InputShapeKey> inputs = {{3, 0}};

  const ArgumentBinding binding =
      BindArgumentsToInputs(inputs, {3}, {ANNC_ELEMENT_TYPE_F32});

  EXPECT_EQ(binding.indices, (std::vector<int64_t>{-1}));
}

}  // namespace
}  // namespace annc_ep

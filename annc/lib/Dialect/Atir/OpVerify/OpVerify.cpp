#include <dlfcn.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <queue>
#include <vector>

#include "Dialect/Atir/OpVerify/OpVerify.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/cc/ops/array_ops.h"

using namespace llvm;
using namespace mlir;
using namespace tensorflow;
using namespace std;

// Anonymous Namespace
namespace {
#define KP_GEN_LIB_PATH "./kpGenOp.bin"
#define DEFAULT_INPUT_DIM 4
//  Op 
#define ALL_OP_TYPES(OP, SrcOp) \

#define INTERPRET_SINGLE_OP(OpType, SrcOp)            \
  if (llvm::isa<atir::OpType>(SrcOp)) {               \
    auto kp_op = llvm::dyn_cast<atir::OpType>(SrcOp); \
    kp_op.Interpret();                                \
    break;                                            \
  }

// 
#define INTERPRET_OP(SrcOp)                         \
  do {                                              \
    ALL_OP_TYPES(INTERPRET_SINGLE_OP, SrcOp)        \
    llvm::errs() << "Error: operation not support!" \
                 << "\n";                           \
    return;                                         \
  } while (0);

class IoTensorDef {
 public:
  using TensorData = std::pair<std::string, mlir::Value>;
  using TensorStorage = llvm::SmallVector<TensorData, 4>;

  // 
  void addInput(const std::string &name, mlir::Value value) {
    inputs.emplace_back(name, value);
  }

  const TensorStorage &getInputs() const { return inputs; }

  const TensorStorage &getOutputs() const { return outputs; }

  // 
  void addOutput(const std::string &name, mlir::Value value) {
    outputs.emplace_back(name, value);
  }

  // 
  bool isEmpty() const { return inputs.empty() && outputs.empty(); }

  // 
  size_t getInputCount() const { return inputs.size(); }

  // 
  size_t getOutputCount() const { return outputs.size(); }

 private:
  TensorStorage inputs;   // 
  TensorStorage outputs;  // 
};

/**
 * @brief
 * mlirgetOperation();
 * @param root MLIR
 * @param inputs 
 * @param funcName 
 */
void createIoDef(mlir::ModuleOp *root, IoTensorDef *inputs,
                 const std::string &funcName) {
  if (!root || !inputs) {
    llvm::errs() << "Error: Invalid parameters in createIoDef\n";
    return;
  }

  // 
  auto targetFuncOp = root->lookupSymbol<func::FuncOp>(funcName);
  if (!targetFuncOp) {
    llvm::errs() << "Error: Could not find target function:" << funcName
                 << "\n";
    return;
  }

  llvm::dbgs() << "Creating input/output definitions for function: " << funcName
               << "\n";

  // 
  MutableArrayRef<BlockArgument> graphInputs;
  if (targetFuncOp.getNumArguments() > 0) {
    llvm::dbgs() << funcName << " inputs:\n";
    graphInputs = targetFuncOp.getArguments();
    for (auto arg : targetFuncOp.getArguments()) {
      llvm::dbgs() << "  " << arg << " : " << arg.getType() << "\n";
    }
  }

  // IoTensorDef
  int inputIndex = 0;
  for (mlir::Value &arg : graphInputs) {
    // todo 
    std::string inputName = "input_" + std::to_string(inputIndex++);

    mlir::Type type = arg.getType();
    llvm::dbgs() << "Processing argument type: " << type << "\n";

    auto tensorType = dyn_cast<atir::TensorType>(type);
    if (!tensorType) {
      // TensorType MemRef
      llvm::dbgs()
          << "Info: tensorType is null! continue process other op...\n";
      continue;
    }

    mlir::ArrayRef<int64_t> shape = tensorType.getShape();
    mlir::Type elementType = tensorType.getElementType();

    // 
    int64_t numElements = 1;
    std::vector<int64_t> staticShape;
    for (int64_t dim : shape) {
      if (dim == mlir::ShapedType::kDynamic) {
        llvm::dbgs()
            << "Info: dynamic shape argument, using DEFAULT_INPUT_DIM. \n";
        dim = DEFAULT_INPUT_DIM;
      }
      numElements *= dim;
      staticShape.push_back(dim);
    }

    if (numElements == 0) continue;

    //  RankedTensorType
    mlir::RankedTensorType standardTensorType =
        mlir::RankedTensorType::get(staticShape, elementType);

    //  Attribute
    mlir::Attribute randomAttr;
    if (elementType.isF32()) {
      std::vector<float> randomData(numElements);
      // 
      for (int64_t i = 0; i < numElements; ++i) {
        randomData[i] =
            static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f;
      }
      //  DenseElementsAttr
      auto randomArData =
          llvm::ArrayRef<float>(randomData.data(), randomData.size());
      llvm::dbgs() << "F32 randomData to ArrayRef done"
                   << "\n";
      randomAttr =
          mlir::DenseElementsAttr::get(standardTensorType, randomArData);
    } else if (elementType.isInteger(32)) {
      std::vector<int32_t> randomData(numElements);
      for (int64_t i = 0; i < numElements; ++i) {
        randomData[i] = rand() % 100;  // 0  99 
      }
      auto randomArData =
          llvm::ArrayRef<int32_t>(randomData.data(), randomData.size());
      randomAttr =
          mlir::DenseElementsAttr::get(standardTensorType, randomArData);
    }
    // todo ...  (f64, i64, etc.) 

    if (!randomAttr) {
      llvm::errs() << "Error: randomAttr null! "
                   << "\n";
      continue;
    }

    tensorType.setCacheData(dyn_cast<mlir::DenseElementsAttr>(randomAttr));

    // IoTensorDef
    atir::TensorType resultTensor = atir::TensorType::get(
        staticShape, tensorType.getElementType(), tensorType.getName(),
        tensorType.getEncoding(), tensorType.getStride(),
        tensorType.getLayout(), tensorType.getMemType(),
        tensorType.getAddress(), tensorType.getDeviceParallel(),
        tensorType.getOnchipParallel(), tensorType.getCacheData());
    arg.setType(resultTensor);

    // IoTensorDef *inputs
    inputs->addInput(inputName, arg);

    // hexdebug
    llvm::dbgs() << "Input '" << inputName << " hex: ";
    auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(randomAttr);
    if (denseAttr) {
      denseAttr.dump();
    }
  }

  // 
  llvm::dbgs() << "Info: created " << inputs->getInputCount()
               << " input tensors in createIoDef.\n";
}

std::string getfuncNameFromPath(const std::string &pathValue) {
  namespace fs = std::filesystem;

  fs::path p(pathValue);
  if (pathValue.empty() || !fs::exists(p)) {
    return "";
  }
  std::string fileNameWithExt = p.filename().string();
  std::string baseName = p.stem().string();

  return baseName;
}

/* memref<1536x1344xf32>, memref<1344x1152xf32>, memref<1536x1152xf32> */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} memref_2d_f32;

/* memref<1152xf32> */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} memref_1d_f32;

/* ================================
 * MLIR generated C interface
 * ================================ */

typedef void (*fused_matmul_func_t)(memref_2d_f32 *ret, memref_2d_f32 *A,
                                    memref_2d_f32 *B, memref_2d_f32 *C,
                                    memref_1d_f32 *bias);
static void *aligned_malloc(size_t size) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 64, size) != 0) {
    return nullptr;
  }
  return ptr;
}
// todo ret type: mlir::success()
// wrap to: void **args
// todo 
// 
bool runKpGenOP(const std::string &libPath, IoTensorDef *inputs,
                IoTensorDef *outputs) {
  llvm::dbgs() << "\n";
  if (inputs->getInputCount()) {
    auto inputStore = inputs->getInputs();
    for (const auto &inputData : inputStore) {
      auto inputValue = inputData.second;
      inputValue.dump();
    }
  }

  // 
  printf("...%s\n", libPath.c_str());
  void *handle = dlopen(libPath.c_str(), RTLD_LAZY);
  if (!handle) {
    llvm::errs() << ": " << dlerror() << "\n";
    return false;
  }

  // 
  printf("...\n");
  auto funcName = getfuncNameFromPath(libPath);
  funcName = "_mlir_ciface_" + funcName;
  auto fused_matmul = (fused_matmul_func_t)dlsym(handle, funcName.c_str());
  if (!fused_matmul) {
    llvm::errs() << ": " << dlerror() << "\n";
    dlclose(handle);
    return false;
  }

  // inputshape
  // IoTensorDefinput
  std::vector<std::vector<int64_t>> shapes;
  std::vector<mlir::DenseElementsAttr> values;
  for (const auto &tensorData : inputs->getInputs()) {
    auto opResult = tensorData.second;

    // OpResult
    mlir::Type type = opResult.getType();

    if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
      // shapevector
      mlir::ArrayRef<int64_t> shape = tensorType.getShape();
      std::vector<int64_t> shapeVec(shape.begin(), shape.end());
      shapes.emplace_back(shapeVec);
      auto attr = tensorType.getCacheData();
      values.emplace_back(attr);
    }
  }
  long M = shapes[0][0];
  long K = shapes[0][1];
  long N = shapes[1][1];
  llvm::outs() << "M: " << M << ", N: " << N << ", K: " << K << "\n";

  printf("...\n");
  printf("[test] allocating buffers...\n");

  auto *A_buf =
      static_cast<float *>(aligned_malloc((size_t)M * K * sizeof(float)));
  auto *B_buf =
      static_cast<float *>(aligned_malloc((size_t)K * N * sizeof(float)));
  auto *C_buf =
      static_cast<float *>(aligned_malloc((size_t)M * N * sizeof(float)));
  auto *bias_buf =
      static_cast<float *>(aligned_malloc((size_t)N * sizeof(float)));

  if (!A_buf || !B_buf || !C_buf || !bias_buf) {
    fprintf(stderr, "[test] allocation failed\n");
    return false;
  }

  for (int i = 0; i < M * K; ++i) {
    auto value = values[0].getValues<float>();
    A_buf[i] = value[i];
    llvm::dbgs() << A_buf[i] << ", ";
  }
  llvm::dbgs() << "\n";

  for (int i = 0; i < K * N; ++i) {
    auto value = values[1].getValues<float>();
    B_buf[i] = value[i];
    llvm::dbgs() << B_buf[i] << ", ";
  }
  llvm::dbgs() << "\n";

  for (int i = 0; i < M * N; ++i) C_buf[i] = 0.0f;

  for (int i = 0; i < N; ++i) {
    auto value = values[3].getValues<float>();
    bias_buf[i] = value[i];
    llvm::dbgs() << bias_buf[i] << ", ";
  }
  llvm::dbgs() << "\n";

  memref_2d_f32 A = {.allocated = A_buf,
                     .aligned = A_buf,
                     .offset = 0,
                     .sizes = {M, K},
                     .strides = {K, 1}};

  memref_2d_f32 B = {.allocated = B_buf,
                     .aligned = B_buf,
                     .offset = 0,
                     .sizes = {K, N},
                     .strides = {N, 1}};

  memref_2d_f32 C = {.allocated = C_buf,
                     .aligned = C_buf,
                     .offset = 0,
                     .sizes = {M, N},
                     .strides = {N, 1}};

  memref_1d_f32 bias = {.allocated = bias_buf,
                        .aligned = bias_buf,
                        .offset = 0,
                        .sizes = {N},
                        .strides = {1}};

  printf("[test] calling fused_matmul...\n");

  memref_2d_f32 ret = C;  //  alloc
  fused_matmul(&ret, &A, &B, &C, &bias);

  printf("[test] done fused_matmul...\n");

  // /
  // RankedTensorType
  auto ctx = inputs->getInputs()[2].second.getContext();
  mlir::Type f32Type = mlir::Float32Type::get(ctx);
  auto rankedTensorType = RankedTensorType::get({M, N}, f32Type);
  // DenseElementsAttr
  std::vector<float> result(C_buf, C_buf + (M * N));
  auto denseAttr =
      DenseElementsAttr::get(rankedTensorType, llvm::ArrayRef(result));
  denseAttr.dump();

  // atir::TensorTypecacheData
  auto tensorType = atir::TensorType::get({M, N}, f32Type);
  tensorType.setCacheData(denseAttr);
  // Value
  mlir::Value v = inputs->getInputs()[2].second;
  v.setType(tensorType);
  outputs->addOutput("1", v);

  free(A_buf);
  free(B_buf);
  free(C_buf);
  free(bias_buf);
  return true;
}

bool runLLMGenOP(const std::string &libPath, IoTensorDef *inputs, IoTensorDef *outputs) {
  llvm::dbgs() << "=====  =====\n";
  TF_Status* status = TF_NewStatus();
  TF_Library* lib = TF_LoadLibrary(libPath.c_str(), status);
  if (TF_GetCode(status) != TF_OK) {
    llvm::dbgs() << "===== " << TF_Message(status) << " =====\n";
    TF_DeleteStatus(status);
    return false;
  }
  llvm::dbgs() << "=====  =====\n";

  llvm::dbgs() << "===== Input Shape =====\n";
  std::vector<std::vector<int64_t>> shapes;
  std::vector<mlir::DenseElementsAttr> values;
  for (const auto &tensorData : inputs->getInputs()) {
    auto opResult = tensorData.second;
    mlir::Type type = opResult.getType();
    if (auto tensorType = dyn_cast<atir::TensorType>(type)) {
      // shapevector
      mlir::ArrayRef<int64_t> shape = tensorType.getShape();
      std::vector<int64_t> shapeVec(shape.begin(), shape.end());
      shapes.emplace_back(shapeVec);
      auto attr = tensorType.getCacheData();
      values.emplace_back(attr);
    }
  }
  long M = shapes[0][0];
  long N = shapes[1][1];
  llvm::dbgs() << "===== Input Shape =====\n";

  llvm::dbgs() << "=====  =====\n";
  TF_Buffer opListBuffer = TF_GetOpList(lib);
  if (opListBuffer.data == nullptr || opListBuffer.length == 0) {
    llvm::dbgs() << "=====  =====\n";
    TF_DeleteLibraryHandle(lib);
    TF_DeleteStatus(status);
    return false;
  }
  tensorflow::OpList op_list;
  if (!op_list.ParseFromArray(opListBuffer.data, opListBuffer.length)) {
    llvm::dbgs() << "=====  =====\n";
    TF_DeleteLibraryHandle(lib);
    TF_DeleteStatus(status);
    return false;
  }
  if (op_list.op_size() != 1) {
    llvm::dbgs() << "=====  =====\n";
    TF_DeleteLibraryHandle(lib);
    TF_DeleteStatus(status);
    return false;
  }
  std::string op_name = op_list.op(0).name();
  llvm::dbgs() << ": " << op_name << "\n";
  llvm::dbgs() << "=====  =====\n";

  llvm::dbgs() << "=====  =====\n";
  TF_Graph* graph = TF_NewGraph();
  TF_Status* graph_status = TF_NewStatus();
  llvm::dbgs() << "=====  =====\n";

  llvm::dbgs() << "=====  =====\n";
  TF_OperationDescription* op_desc = TF_NewOperation(graph, op_name.c_str(), op_name.c_str());
  std::vector<TF_Output> input_nodes;
  for (int i = 0; i < 4; ++i) {
    // 
    std::string placeholder_name = "input_" + std::to_string(i);
    TF_OperationDescription* placeholder_desc = TF_NewOperation(graph, "Placeholder", placeholder_name.c_str());
    TF_SetAttrType(placeholder_desc, "dtype", TF_FLOAT);    
    // 
    std::vector<int64_t> shape = shapes[i];
    TF_SetAttrShape(placeholder_desc, "shape", shape.data(), shape.size());
    // 
    TF_Operation* placeholder = TF_FinishOperation(placeholder_desc, graph_status);
    if (TF_GetCode(graph_status) != TF_OK) {
      llvm::dbgs() << "===== " << TF_Message(graph_status) << " =====\n";
      TF_DeleteStatus(graph_status);
      TF_DeleteGraph(graph);
      TF_DeleteLibraryHandle(lib);
      TF_DeleteStatus(status);
      return false;
    }
    // 
    TF_Output output = {placeholder, 0};
    input_nodes.push_back(output);
  }
  llvm::dbgs() << "=====  =====\n";
  
  // 
  for (const auto& input : input_nodes) {
    TF_AddInput(op_desc, input);
  }
  llvm::dbgs() << "=====  =====\n";
  TF_Operation* op = TF_FinishOperation(op_desc, graph_status);
  if (TF_GetCode(graph_status) != TF_OK) {
    llvm::dbgs() << "===== " << TF_Message(graph_status) << " =====\n";
    TF_DeleteStatus(graph_status);
    TF_DeleteGraph(graph);
    TF_DeleteLibraryHandle(lib);
    TF_DeleteStatus(status);
    return false;
  }
  llvm::dbgs() << "=====  =====\n";

  llvm::dbgs() << "=====  =====\n";
  std::vector<TF_Tensor*> input_tensors;
  for (int i = 0; i < 4; ++i) {
    // DenseElementsAttr
    auto dense_attr = values[i];
    auto float_data = dense_attr.getValues<float>();
    
    // 
    size_t num_elements = 1;
    for (auto dim : shapes[i]) {
      num_elements *= dim;
    }
    // TF_Tensor
    size_t data_size = num_elements * sizeof(float);
    TF_Tensor* tensor = TF_AllocateTensor(TF_FLOAT, shapes[i].data(), shapes[i].size(), data_size);
    if (!tensor) {
      llvm::dbgs() << "===== Tensor =====\n";
      TF_DeleteStatus(graph_status);
      for (auto t : input_tensors) {
        TF_DeleteTensor(t);
      }
      TF_DeleteGraph(graph);
      TF_DeleteLibraryHandle(lib);
      TF_DeleteStatus(status);
      return false;
    }
    // 
    float* tensor_data = static_cast<float*>(TF_TensorData(tensor));
    for (size_t j = 0; j < num_elements; ++j) {
      tensor_data[j] = float_data[j];
    }
    input_tensors.push_back(tensor);
  }
  llvm::dbgs() << "=====  =====\n";

  // 
  llvm::dbgs() << "=====  =====\n";
  TF_SessionOptions* session_opts = TF_NewSessionOptions();
  tensorflow::ConfigProto config;
  config.mutable_graph_options()->mutable_optimizer_options()->set_opt_level(tensorflow::OptimizerOptions::L0);
  config.mutable_graph_options()->mutable_rewrite_options()->set_disable_meta_optimizer(true);

  // 
  std::string config_str;
  if (!config.SerializeToString(&config_str)) {
      llvm::dbgs() << "\n";
      TF_DeleteSessionOptions(session_opts);
      TF_DeleteStatus(graph_status);
      for (auto t : input_tensors) {
        TF_DeleteTensor(t);
      }
      TF_DeleteGraph(graph);
      TF_DeleteLibraryHandle(lib);
      TF_DeleteStatus(status);
      return false;
  }

  // 
  TF_SetConfig(session_opts, config_str.data(), config_str.size(), graph_status);
  TF_Session* session = TF_NewSession(graph, session_opts, graph_status);
  if (TF_GetCode(graph_status) != TF_OK) {
    llvm::dbgs() << "===== " << TF_Message(graph_status) << " =====\n";
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteStatus(graph_status);
    for (auto t : input_tensors) {
      TF_DeleteTensor(t);
    }
    TF_DeleteGraph(graph);
    TF_DeleteLibraryHandle(lib);
    TF_DeleteStatus(status);
    return false;
  }
  llvm::dbgs() << "=====  =====\n";

  // 
  TF_Output output = {op, 0};
  TF_Tensor* output_tensor = nullptr;

  // 
  llvm::dbgs() << "=====  =====\n";
  TF_SessionRun(session, nullptr,
                input_nodes.data(), input_tensors.data(), input_nodes.size(),
                &output, &output_tensor, 1,
                nullptr, 0, nullptr, graph_status);

  if (TF_GetCode(graph_status) != TF_OK) {
    llvm::dbgs() << "===== " << TF_Message(graph_status) << " =====\n";
  } else {
    llvm::dbgs() << "=====  =====\n";

    // 
    if (output_tensor) {
      float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
      int64_t num_dims = TF_NumDims(output_tensor);
      // 
      std::vector<int64_t> output_shape;
      for (int i = 0; i < num_dims; ++i) {
        output_shape.push_back(TF_Dim(output_tensor, i));
      }
      llvm::dbgs() << ": [";
      for (size_t i = 0; i < output_shape.size(); ++i) {
        llvm::dbgs() << output_shape[i];
        if (i < output_shape.size() - 1) llvm::dbgs() << ", ";
      }
      llvm::dbgs() << "]\n";
      // 
      size_t output_elements = 1;
      for (auto dim : output_shape) {
        output_elements *= dim;
      }
      // 
      llvm::dbgs() << ": ";
      for (size_t i = 0; i < output_elements; ++i) {
        llvm::dbgs() << output_data[i] << " ";
        if ((i+1) % output_shape.back() == 0) {
          llvm::dbgs() << "\n";
        }
      }
      // Tensor
      auto ctx = inputs->getInputs()[0].second.getContext();
      mlir::Type f32Type = mlir::Float32Type::get(ctx);
      auto rankedTensorType = RankedTensorType::get({M, N}, f32Type);
      // DenseElementsAttr
      std::vector<float> result_vector(output_data, output_data + output_elements);
      auto denseAttr = DenseElementsAttr::get(rankedTensorType, llvm::ArrayRef(result_vector));
      // atir::TensorTypecacheData
      auto tensorType = atir::TensorType::get({M, N}, f32Type);
      tensorType.setCacheData(denseAttr);
      // Value
      mlir::Value v = inputs->getInputs()[2].second; // value
      v.setType(tensorType);
      // 
      outputs->addOutput("output", v);
      TF_DeleteTensor(output_tensor);
    }
  }

  // 
  llvm::dbgs() << "=====  =====\n";
  TF_CloseSession(session, graph_status);
  TF_DeleteSession(session, graph_status);
  TF_DeleteSessionOptions(session_opts);
  TF_DeleteStatus(graph_status);
  for (auto tensor : input_tensors) {
    TF_DeleteTensor(tensor);
  }
  TF_DeleteGraph(graph);
  TF_DeleteLibraryHandle(lib);
  TF_DeleteStatus(status);
  return true;
}

/**
 * @brief
 * mlirgetOperation();->Interpret()
 * @param root MLIR
 * @param funcName 
 * @param inputs 
 * @param outputs 
 *
 * inputsoutputs
 * OP hex debug
 */
void runOriGraph(mlir::ModuleOp *root, const std::string &funcName,
                 IoTensorDef *inputs, IoTensorDef *outputs) {
  // 
  auto targetFuncOp = root->lookupSymbol<func::FuncOp>(funcName);
  if (!targetFuncOp) {
    llvm::errs() << "Error: Could not find target function:" << funcName
                 << "\n";
    return;
  }

  // todo ,inputs
  MutableArrayRef<BlockArgument> graphInputs;

  if (targetFuncOp.getNumArguments() > 0) {
    llvm::dbgs() << targetFuncOp.getName() << " inputs:\n";
    graphInputs = targetFuncOp.getArguments();
    for (auto arg : targetFuncOp.getArguments()) {
      llvm::dbgs() << "  " << arg << " : " << arg.getType() << "\n";
    }
  }
  llvm::dbgs() << "inputs size: " << graphInputs.size() << "\n";
  // createIoDef
  if (inputs->getInputCount() == 0) {
    llvm::errs()
        << "Error: inputs should be constructed by createIoDef first!\n";
    return;
  } else {
    llvm::dbgs() << "Info: inputs already contains " << inputs->getInputCount()
                 << " tensors, skipping reconstruction.\n";
  }

  // BFS DAG
  // todo moduleregionannc-opt
  std::vector<mlir::Operation *> allOps;
  targetFuncOp.walk([&](mlir::Operation *op) {
    //  funcOp 
    if (op == targetFuncOp.getOperation()) {
      return;
    }

    //  NoneOp ReturnOp Case
    bool skip = llvm::TypeSwitch<mlir::Operation *, bool>(op)
                    .Case<mlir::func::ReturnOp>([](auto) { return true; })
                    .Case<atir::NoneOp>([](auto) { return true; })
                    .Default([](mlir::Operation *) { return false; });
    if (skip) {
      return;
    }

    allOps.emplace_back(op);
    llvm::dbgs() << "Op\n";
    op->dump();
  });

  std::unordered_map<mlir::Operation *, std::vector<mlir::Operation *>> graph;
  std::unordered_map<mlir::Operation *, int> indegree;

  for (auto *op : allOps) {
    indegree[op] = 0;
  }

  // 
  for (auto *op : allOps) {
    for (auto operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        graph[defOp].push_back(op);
        indegree[op]++;
      }
    }
  }

  // 0
  std::queue<mlir::Operation *> q;
  for (auto *op : allOps) {
    if (indegree[op] == 0) {
      q.push(op);
    }
  }

  llvm::dbgs() << ""
               << "\n";
  // 
  while (!q.empty()) {
    auto *curr = q.front();
    q.pop();

    // 
    // ALL_OP_TYPES
    INTERPRET_OP(curr)

    //  indegree
    auto it = graph.find(curr);
    if (it != graph.end()) {
      for (auto *user : it->second) {
        indegree[user]--;
        if (indegree[user] == 0) {
          q.push(user);
        }
      }
    }
  }

  // shape
  func::FuncOp oriFunc = targetFuncOp;
  auto context = oriFunc->getContext();
  //  block args 
  SmallVector<Type, 4> finalInputTypes;
  finalInputTypes.reserve(oriFunc.getNumArguments());
  for (auto arg : oriFunc.getArguments()) {
    finalInputTypes.push_back(arg.getType());
  }
  //  - 
  mlir::func::ReturnOp returnOp;
  auto resultTypes = llvm::to_vector(oriFunc->getResultTypes());
  if (resultTypes.empty()) {
    llvm::dbgs() << "Info: resultTypes from origin Func is empty, try walking "
                    "ReturnOp...\n";
    oriFunc->walk([&](mlir::func::ReturnOp ret) {
      returnOp = ret;
      for (auto v : ret.getOperands()) {
        llvm::dbgs() << "returnOp operands:\n";
        v.dump();
        resultTypes.push_back(v.getType());
      }
    });
  }

  //  function type
  FunctionType newFuncType =
      FunctionType::get(context, finalInputTypes, resultTypes);
  oriFunc.setType(newFuncType);
  llvm::dbgs() << ":\n";
  oriFunc->dump();

  SymbolTable symbolTable(*root);

  auto uses = symbolTable.getSymbolUses(oriFunc, *root);
  if (uses) {
    for (SymbolTable::SymbolUse use : *uses) {
      auto callOp = dyn_cast<func::CallOp>(use.getUser());
      if (!callOp) continue;

      OpBuilder builder(callOp);
      // 
      // callOp.getOperands().erase(callOp.getOperands().back());

      //  result types
      callOp.getResults()[0].setType(newFuncType.getResult(0));
    }
  }
  // Collect results from block terminators and store in outputs
  if (outputs->getOutputCount() == 0) {
    llvm::dbgs()
        << "Info: collecting results from block terminators to outputs.\n";

    // 
    int outputIndex = 0;

    for (auto result : returnOp.getOperands()) {
      result.dump();
      std::string outputName = "output_" + std::to_string(outputIndex++);
      outputs->addOutput(outputName, result);
    }

    // 
    llvm::dbgs() << "Info: collected " << outputs->getOutputCount()
                 << " output tensors.\n";
  } else {
    llvm::dbgs()
        << "Warning: outputs already contains data, skipping collection.\n";
  }
  llvm::dbgs() << "\n";
  if (outputs->getOutputCount()) {
    auto inputStore = outputs->getOutputs();
    for (const auto &inputData : inputStore) {
      auto inputValue = inputData.second;
      inputValue.dump();
    }
  }
}

/*
class GraphInferEngine
{
// todo 
public:
  GraphInferEngine();
  run();
}
*/

/**
 * @brief 
 * @return std::string 
 */
std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}
/**
 * @brief 
 * @param filename 
 * @param tensorCount 
 * @param passed 
 * @param mismatchedTensors 
 */
void saveVerificationResult(
    const std::string &filename, size_t tensorCount, bool passed,
    const std::vector<std::pair<std::string, std::string>> &mismatchedTensors) {
  std::ofstream outFile(filename);

  if (!outFile.is_open()) {
    llvm::errs() << "Error: Could not open file " << filename
                 << " for writing\n";
    return;
  }

  outFile << "Tensor Verification Result\n";
  outFile << "=========================\n";
  outFile << "Timestamp: " << getCurrentTimestamp() << "\n";
  outFile << "Total Tensors: " << tensorCount << "\n";
  outFile << "Status: " << (passed ? "PASSED" : "FAILED") << "\n";
  outFile << "Mismatched Tensors: " << mismatchedTensors.size() << "\n";

  if (!mismatchedTensors.empty()) {
    outFile << "\nMismatched Details:\n";
    outFile << "------------------\n";
    for (const auto &[name, reason] : mismatchedTensors) {
      outFile << "- " << name << ": " << reason << "\n";
    }
  }

  outFile.close();
  llvm::dbgs() << "Verification result saved to " << filename << "\n";
}

/**
 * @brief DenseElementsAttr
 * @param lhs 
 * @param rhs 
 * @param tolerance 
 * @return double -1
 */
double calculateMaxDiff(DenseElementsAttr lhs, DenseElementsAttr rhs,
                        double tolerance) {
  double maxDiff = 0.0;

  if (lhs.getElementType().isF32()) {
    auto lhsVals = lhs.getValues<float>();
    auto rhsVals = rhs.getValues<float>();

    for (size_t i = 0; i < lhsVals.size() && i < rhsVals.size(); ++i) {
      double diff = std::abs(lhsVals[i] - rhsVals[i]);
      maxDiff = std::max(maxDiff, diff);
    }
  } else if (lhs.getElementType().isInteger(32)) {
    auto lhsVals = lhs.getValues<int32_t>();
    auto rhsVals = rhs.getValues<int32_t>();

    for (size_t i = 0; i < lhsVals.size() && i < rhsVals.size(); ++i) {
      double diff = std::abs(static_cast<double>(lhsVals[i]) -
                             static_cast<double>(rhsVals[i]));
      maxDiff = std::max(maxDiff, diff);
    }
  }

  return maxDiff;
}

/**
 * @brief IoTensorDef
 * @param lhs 
 * @param rhs 
 * @param tolerance 1e-6
 * @return bool truefalse
 *
 * 
 * 
 * 
 */
bool verifyTensor(IoTensorDef *lhs, IoTensorDef *rhs, double tolerance = 1e-6) {
  if (!lhs || !rhs) {
    llvm::errs() << "Error: verifyTensor called with null pointers\n";
    return false;
  }

  llvm::dbgs() << "Starting tensor verification...\n";

  // 
  size_t lhsCount = lhs->getOutputCount();
  size_t rhsCount = rhs->getOutputCount();

  if (lhsCount != rhsCount) {
    llvm::errs() << "Error: tensor count mismatch - lhs: " << lhsCount
                 << ", rhs: " << rhsCount << "\n";
    return false;
  }

  if (lhsCount == 0 || rhsCount == 0) {
    llvm::dbgs() << "Info: no output tensors to verify\n";
    return true;
  }

  llvm::dbgs() << "Verifying " << lhsCount << " output tensors...\n";

  // 
  bool allMatch = true;
  std::vector<std::pair<std::string, std::string>> mismatchedTensors;

  for (size_t i = 0; i < lhsCount; ++i) {
    auto &lhsTensor = lhs->getOutputs()[i];
    auto &rhsTensor = rhs->getOutputs()[i];

    const std::string &name = lhsTensor.first;

    llvm::dbgs() << "Verifying tensor: " << name << "\n";

    try {
      // 
      auto lhsOpResult = lhsTensor.second;
      auto rhsOpResult = rhsTensor.second;
      auto lhsTensorType = dyn_cast<atir::TensorType>(lhsOpResult.getType());
      auto rhsTensorType = dyn_cast<atir::TensorType>(rhsOpResult.getType());
      if (!lhsTensorType || !rhsTensorType) {
        llvm::errs() << "Error: Cannot cast to TensorType for tensor " << name
                     << "\n";
        allMatch = false;
        mismatchedTensors.emplace_back(name, "Cannot cast to TensorType");
        continue;
      }

      auto lhsAttr = lhsTensorType.getCacheData();
      auto rhsAttr = rhsTensorType.getCacheData();
      if (!lhsAttr || !rhsAttr) {
        llvm::errs() << "Warning: No cache data for tensor " << name << "\n";
        continue;
      }

      if (lhsAttr == rhsAttr) {
        llvm::dbgs() << "Info: tensor " << name << " matches\n";
        continue;
      } else {
        llvm::dbgs() << "Warning: tensor " << name << " does not match\n";

        // datadebug
        llvm::dbgs() << "  LHS data for " << name << ": ";
        lhsAttr.dump();
        llvm::dbgs() << "  RHS data for " << name << ": ";
        rhsAttr.dump();

        // DenseElementsAttr
        if (auto lhsDense = dyn_cast<DenseElementsAttr>(lhsAttr)) {
          if (auto rhsDense = dyn_cast<DenseElementsAttr>(rhsAttr)) {
            double maxDiff = calculateMaxDiff(lhsDense, rhsDense, tolerance);
            llvm::dbgs() << "  Max difference: " << maxDiff << "\n";

            if (maxDiff > tolerance) {  // 
              llvm::errs()
                  << "Error: Significant difference detected in tensor " << name
                  << " (threshold: " << tolerance << ", actual: " << maxDiff
                  << ")\n";
              allMatch = false;
              mismatchedTensors.emplace_back(
                  name, "Value difference > " + std::to_string(tolerance));
            } else {
              llvm::dbgs() << "Info: Differences within tolerance for tensor "
                           << name << " (max diff: " << maxDiff
                           << " <= " << tolerance << ")\n";
            }
          }
        }
      }
    } catch (const std::exception &e) {
      llvm::errs() << "Error during verification of tensor " << name << ": "
                   << e.what() << "\n";
      allMatch = false;
      mismatchedTensors.emplace_back(name,
                                     std::string("Exception: ") + e.what());
    }
  }

  // 
  saveVerificationResult("tensor_verification_result.txt", lhsCount, allMatch,
                         mismatchedTensors);
  if (allMatch) {
    llvm::outs() << " Tensor verification PASSED - All tensors match\n";
  } else {
    llvm::outs() << " Tensor verification FAILED " << mismatchedTensors.size()
                 << " tensors do not match\n";
    for (const auto &[name, reason] : mismatchedTensors) {
      llvm::outs() << "  - " << name << ": " << reason << "\n";
    }
  }

  return allMatch;
}
}  // namespace

namespace atir {
class AtirOpVerifyPass : public AtirOpVerifyBase<AtirOpVerifyPass> {
 public:
  AtirOpVerifyPass() = default;

  void runOnOperation() override {
    auto m = getOperation();
    IoTensorDef inputDef;
    // SmallVector
    IoTensorDef oriOutput;
    IoTensorDef kpGenOutput;

    std::string pathValue;
    std::string kpPathValue = kpGenLibPath.getValue();
    std::string llmPathValue = llmGenLibPath.getValue();

    namespace fs = std::filesystem;

    fs::path kp(kpPathValue);
    fs::path lp(llmPathValue);

    if (fs::exists(kp) && !kpPathValue.empty()) {
      pathValue = kpPathValue;
    } else if (fs::exists(lp) && !llmPathValue.empty()) {
      pathValue = llmPathValue;
    } else {
      pathValue = "";
    }
    std::string funcName = getfuncNameFromPath(pathValue);
    if (funcName.empty()) {
      emitError(getOperation().getLoc(),
                "Missing or Wrong option: kpGenLibPath must be specified. "
                "And the file should exists"
                "--atir-op-verify=\"kpGenLibPath=/path/to/lib.so\"");
      signalPassFailure();
      return;
    }
    llvm::outs() << "Start to verify func: " << funcName << "\n";
    llvm::outs() << "Original graph: \n";
    m->dump();

    // 
    llvm::outs() << "Constructing input tensors for original graph...\n";
    createIoDef(&m, &inputDef, funcName);

    // 
    runOriGraph(&m, funcName, &inputDef, &oriOutput);

    // 
    if (fs::exists(kp) && !kpPathValue.empty()) {
      runKpGenOP(pathValue, &inputDef, &kpGenOutput);
    } else if (fs::exists(lp) && !llmPathValue.empty()) {
      llvm::outs() << "======================================================\n";
      runLLMGenOP(pathValue, &inputDef, &kpGenOutput);
      llvm::outs() << "======================================================\n";
    } else {
      return;
    }

    // 
    bool verificationPassed = verifyTensor(&oriOutput, &kpGenOutput, 1e-4);

    // 
    bool strictVerification = verifyTensor(&oriOutput, &kpGenOutput, 1e-6);

    if (verificationPassed) {
      llvm::outs() << " Verification PASSED - Original graph and generated "
                      "operator produce identical results (tolerance: 1e-4)\n";
      if (strictVerification) {
        llvm::outs() << "   Additionally passed strict verification "
                        "(tolerance: 1e-6)\n";
      } else {
        llvm::outs() << "   Note: Failed strict verification but passed "
                        "relaxed criteria\n";
      }
    } else {
      llvm::outs() << " Verification FAILED - Results differ between original "
                      "graph and generated operator\n";
      llvm::outs() << "  Consider adjusting tolerance threshold or checking "
                      "implementation\n";
    }

    llvm::outs() << "OpVerifyPass End"
                 << "\n";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAtirOpVerifyPass() {
  return std::make_unique<AtirOpVerifyPass>();
}
}  // namespace atir
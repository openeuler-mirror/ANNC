#include "Dialect/Atir/OpVerify/runLLMGenOp.h"
#include "Dialect/Atir/OpVerify/IoDef.h"
#include "Dialect/Atir/AtirOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"

// TensorFlow C API (for loading library)
#include <tensorflow/c/c_api.h>
// TensorFlow C++ API (for building graph)
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/lib/core/status.h>
#include <vector>
#include <memory>
#include <chrono>
#include <numeric>

using namespace llvm;
using namespace mlir;

namespace atir {

namespace {

// RAII wrapper for TF_Status
struct TFStatusGuard {
    TF_Status* status = nullptr;
    TFStatusGuard() : status(TF_NewStatus()) {}
    ~TFStatusGuard() { if (status) TF_DeleteStatus(status); }
    TFStatusGuard(const TFStatusGuard&) = delete;
    TFStatusGuard& operator=(const TFStatusGuard&) = delete;
    TF_Status* get() const { return status; }
};

// RAII wrapper for TF_Library
struct TFLibraryHandle {
    TF_Library* lib = nullptr;
    explicit TFLibraryHandle(const char* path) {
        TFStatusGuard status;
        lib = TF_LoadLibrary(path, status.get());
        if (TF_GetCode(status.get()) != TF_OK) {
            llvm::errs() << "Failed to load library: " << TF_Message(status.get()) << "\n";
        }
    }
    ~TFLibraryHandle() { if (lib) TF_DeleteLibraryHandle(lib); }
    TFLibraryHandle(const TFLibraryHandle&) = delete;
    TFLibraryHandle& operator=(const TFLibraryHandle&) = delete;
    explicit operator bool() const { return lib != nullptr; }
    TF_Library* get() const { return lib; }
};

// Helper: Extract input shapes and values from IoTensorDef
struct InputData {
    std::vector<std::vector<int64_t>> shapes;
    std::vector<mlir::DenseElementsAttr> values;
};

InputData extractInputs(IoTensorDef *inputs) {
    InputData result;
    for (const auto &tensorData : inputs->getInputs()) {
        auto opResult = tensorData.second;
        mlir::Type type = opResult.getType();
        if (auto tensorType = llvm::dyn_cast<atir::TensorType>(type)) {
            mlir::ArrayRef<int64_t> shape = tensorType.getShape();
            result.shapes.emplace_back(shape.begin(), shape.end());
            result.values.emplace_back(tensorType.getCacheData());
        }
    }
    return result;
}

// Helper: Get operator name from library
// Try to get op list from library first, fallback to parsing from library path
llvm::Expected<std::string> getOpNameFromLib(TF_Library* lib, const std::string& libPath) {
    // Try to get op list from library
    TF_Status* status = TF_NewStatus();
    TF_Buffer opListBuffer = TF_GetOpList(lib);
    
    if (opListBuffer.data != nullptr && opListBuffer.length > 0) {
        tensorflow::OpList op_list;
        if (op_list.ParseFromArray(opListBuffer.data, opListBuffer.length)) {
            if (op_list.op_size() >= 1) {
                std::string op_name = op_list.op(0).name();
                TF_DeleteStatus(status);
                return op_name;
            }
        }
    }
    
    TF_DeleteStatus(status);
    
    // Fallback: extract op name from library filename
    // e.g., "fused_matmul_add_relu.so" -> "FusedMatmulAddRelu"
    // or "lib_fused_matmul_add_relu_v2.so" -> "FusedMatmulAddReluV2"
    size_t lastSlash = libPath.find_last_of('/');
    std::string filename = (lastSlash != std::string::npos) ? 
                           libPath.substr(lastSlash + 1) : libPath;
    
    // Remove .so extension
    size_t extPos = filename.find(".so");
    if (extPos != std::string::npos) {
        filename = filename.substr(0, extPos);
    }
    
    // Convert snake_case to CamelCase
    std::string opName;
    bool nextUpper = true;
    for (char c : filename) {
        if (c == '_') {
            nextUpper = true;
        } else {
            if (nextUpper) {
                opName += std::toupper(c);
                nextUpper = false;
            } else {
                opName += c;
            }
        }
    }
    
    return opName;
}

// Helper: Convert MLIR Type to TensorFlow DataType
llvm::Expected<tensorflow::DataType> mlirTypeToTFDataType(mlir::Type mlirType) {
    // Define type mapping using a static lookup table
    static const std::pair<std::function<bool(mlir::Type)>, tensorflow::DataType> typeMap[] = {
        {[](mlir::Type t) { return t.isF32(); }, tensorflow::DT_FLOAT},
        {[](mlir::Type t) { return t.isF64(); }, tensorflow::DT_DOUBLE},
        {[](mlir::Type t) { return t.isF16(); }, tensorflow::DT_HALF},
        {[](mlir::Type t) { return t.isBF16(); }, tensorflow::DT_BFLOAT16},
        {[](mlir::Type t) { return t.isInteger(8); }, tensorflow::DT_INT8},
        {[](mlir::Type t) { return t.isInteger(16); }, tensorflow::DT_INT16},
        {[](mlir::Type t) { return t.isInteger(32); }, tensorflow::DT_INT32},
        {[](mlir::Type t) { return t.isInteger(64); }, tensorflow::DT_INT64},
        {[](mlir::Type t) { return t.isUnsignedInteger(8); }, tensorflow::DT_UINT8},
        {[](mlir::Type t) { return t.isUnsignedInteger(16); }, tensorflow::DT_UINT16},
        {[](mlir::Type t) { return t.isUnsignedInteger(32); }, tensorflow::DT_UINT32},
        {[](mlir::Type t) { return t.isUnsignedInteger(64); }, tensorflow::DT_UINT64},
    };
    
    // Search for matching type
    for (const auto& [predicate, tfType] : typeMap) {
        if (predicate(mlirType)) {
            return tfType;
        }
    }
    
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                  "Unsupported MLIR element type");
}

// Helper: Copy data from MLIR DenseElementsAttr to TensorFlow Tensor
template<typename T>
void copyDataToTensor(const mlir::DenseElementsAttr& attr, 
                      tensorflow::Tensor& tensor, 
                      size_t num_elements) {
    auto data = attr.getValues<T>();
    auto tensor_map = tensor.flat<T>();
    for (size_t i = 0; i < num_elements; ++i) {
        tensor_map(i) = data[i];
    }
}

// Helper: Create TensorFlow Tensor from MLIR DenseElementsAttr
llvm::Expected<tensorflow::Tensor> createTFFromMLIR(const mlir::DenseElementsAttr& attr, 
                                                     const std::vector<int64_t>& shape) {
    // Get element type
    auto elemType = attr.getElementType();
    auto tfDataTypeResult = mlirTypeToTFDataType(elemType);
    if (!tfDataTypeResult) {
        return tfDataTypeResult.takeError();
    }
    tensorflow::DataType tf_dtype = *tfDataTypeResult;
    
    // Calculate total elements
    size_t num_elements = 1;
    for (auto dim : shape) {
        num_elements *= dim;
    }
    
    // Create TF Tensor
    tensorflow::Tensor tensor(tf_dtype, tensorflow::TensorShape(shape));
    
    // Copy data based on type using template helper
    switch (tf_dtype) {
        case tensorflow::DT_FLOAT:
            copyDataToTensor<float>(attr, tensor, num_elements);
            break;
        case tensorflow::DT_DOUBLE:
            copyDataToTensor<double>(attr, tensor, num_elements);
            break;
        case tensorflow::DT_INT32:
            copyDataToTensor<int32_t>(attr, tensor, num_elements);
            break;
        case tensorflow::DT_INT64:
            copyDataToTensor<int64_t>(attr, tensor, num_elements);
            break;
        case tensorflow::DT_INT8:
            copyDataToTensor<int8_t>(attr, tensor, num_elements);
            break;
        case tensorflow::DT_UINT8:
            copyDataToTensor<uint8_t>(attr, tensor, num_elements);
            break;
        default:
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                          "Data copy not implemented for this type");
    }
    
    return tensor;
}

// Helper: Store TF output to MLIR IoTensorDef (template for type safety)
template<typename T>
void storeOutputToMLIRTyped(IoTensorDef *outputs, 
                            const tensorflow::Tensor& tf_tensor,
                            MLIRContext* ctx) {
    // Extract shape
    auto tf_shape = tf_tensor.shape();
    std::vector<int64_t> output_shape;
    for (int i = 0; i < tf_shape.dims(); ++i) {
        output_shape.push_back(tf_shape.dim_size(i));
    }
    
    // Extract data with correct type
    auto tf_data = tf_tensor.flat<T>();
    size_t num_elements = tf_data.size();
    std::vector<T> result_vector(tf_data.data(), 
                                 tf_data.data() + num_elements);
    
    // Create MLIR TensorType with cache data
    mlir::Type elemType;
    if constexpr (std::is_same_v<T, float>) {
        elemType = mlir::Float32Type::get(ctx);
    } else if constexpr (std::is_same_v<T, double>) {
        elemType = mlir::Float64Type::get(ctx);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        elemType = mlir::IntegerType::get(ctx, 32);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        elemType = mlir::IntegerType::get(ctx, 64);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        elemType = mlir::IntegerType::get(ctx, 8);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        elemType = mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
    } else {
        llvm::errs() << "Error: Unsupported output type in storeOutputToMLIR\n";
        return;
    }
    
    auto rankedTensorType = RankedTensorType::get(output_shape, elemType);
    auto denseAttr = DenseElementsAttr::get(rankedTensorType, 
                                            llvm::ArrayRef(result_vector));
    
    auto tensorType = atir::TensorType::get(output_shape, elemType);
    tensorType.setCacheData(denseAttr);
    
    // Create a dummy buffer operation to produce a Value with the correct type
    mlir::OperationState state(mlir::UnknownLoc::get(ctx), "atir.buffer");
    state.addTypes(tensorType);
    mlir::Operation *bufferOp = mlir::Operation::create(state);
    mlir::Value bufferValue = bufferOp->getResult(0);
    
    // Add to outputs
    outputs->addOutput("output", bufferValue);
}

// Helper: Store TF output to MLIR IoTensorDef (dispatches based on dtype)
void storeOutputToMLIR(IoTensorDef *outputs, 
                       const tensorflow::Tensor& tf_tensor,
                       MLIRContext* ctx) {
    switch (tf_tensor.dtype()) {
        case tensorflow::DT_FLOAT:
            storeOutputToMLIRTyped<float>(outputs, tf_tensor, ctx);
            break;
        case tensorflow::DT_DOUBLE:
            storeOutputToMLIRTyped<double>(outputs, tf_tensor, ctx);
            break;
        case tensorflow::DT_INT32:
            storeOutputToMLIRTyped<int32_t>(outputs, tf_tensor, ctx);
            break;
        case tensorflow::DT_INT64:
            storeOutputToMLIRTyped<int64_t>(outputs, tf_tensor, ctx);
            break;
        case tensorflow::DT_INT8:
            storeOutputToMLIRTyped<int8_t>(outputs, tf_tensor, ctx);
            break;
        case tensorflow::DT_UINT8:
            storeOutputToMLIRTyped<uint8_t>(outputs, tf_tensor, ctx);
            break;
        default:
            llvm::errs() << "Error: Unsupported TensorFlow dtype: " << tf_tensor.dtype() << "\n";
            break;
    }
}

} // anonymous namespace

bool runLLMGenOp(const std::string &libPath, IoTensorDef *inputs, 
                 IoTensorDef *outputs,
                 PerformanceMetrics &metrics,
                 const WarmupConfig &config) {
    // Step 1: Load library (RAII - automatic cleanup)
    TFLibraryHandle lib(libPath.c_str());
    if (!lib) {
        return false;
    }

    // Step 2: Extract input data
    InputData inputData = extractInputs(inputs);

    if (inputData.shapes.empty()) {
        llvm::errs() << "Error: No input tensors found\n";
        return false;
    }
        
    // Step 3: Get operator name
    auto opNameResult = getOpNameFromLib(lib.get(), libPath);
    if (!opNameResult) {
        llvm::errs() << "Error: " << toString(opNameResult.takeError()) << "\n";
        return false;
    }
    std::string op_name = *opNameResult;
        
    // Step 4: Build TensorFlow graph using C++ API
    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    
    // Create placeholders for inputs
    std::vector<tensorflow::Output> placeholders;
    for (size_t i = 0; i < inputData.shapes.size(); ++i) {
        std::string name = "input_" + std::to_string(i);
        tensorflow::PartialTensorShape tf_shape(inputData.shapes[i]);
        
        // Get element type from MLIR
        auto tensorType = llvm::dyn_cast<atir::TensorType>(inputs->getInputs()[i].second.getType());
        if (!tensorType) {
            llvm::errs() << "Error: Invalid tensor type for input " << i << "\n";
            return false;
        }
        
        auto tfDataTypeResult = mlirTypeToTFDataType(tensorType.getElementType());
        if (!tfDataTypeResult) {
            llvm::errs() << "Error: " << toString(tfDataTypeResult.takeError()) << "\n";
            return false;
        }
        
        auto ph = tensorflow::ops::Placeholder(root.WithOpName(name), *tfDataTypeResult,
                              tensorflow::ops::Placeholder::Shape(tf_shape));
        placeholders.push_back(ph);
    }
    
    // Create custom operation
    tensorflow::NodeDef node_def;
    node_def.set_name(op_name);
    node_def.set_op(op_name);
    
    for (size_t i = 0; i < placeholders.size(); ++i) {
        node_def.add_input(placeholders[i].node()->name() + ":" + 
                          std::to_string(placeholders[i].index()));
    }
    
    tensorflow::Status s;
    tensorflow::Node* node = root.graph()->AddNode(node_def, &s);
    if (!s.ok()) {
        llvm::errs() << "Error creating custom op: " << s.ToString() << "\n";
        return false;
    }
    auto custom_op = tensorflow::Output(node, 0);
        
    // Step 5: Prepare input tensors
    tensorflow::ClientSession::FeedType feeds;
    for (size_t i = 0; i < inputData.shapes.size(); ++i) {
        auto tfTensorResult = createTFFromMLIR(inputData.values[i], inputData.shapes[i]);
        if (!tfTensorResult) {
            llvm::errs() << "Error: " << toString(tfTensorResult.takeError()) << "\n";
            return false;
        }
        feeds.insert({placeholders[i], tensorflow::Input::Initializer(*tfTensorResult)});
    }
        
    // Step 6: Create session
    tensorflow::SessionOptions session_opts;
    session_opts.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(tensorflow::OptimizerOptions::L0);
    session_opts.config.mutable_graph_options()
        ->mutable_rewrite_options()
        ->set_disable_meta_optimizer(true);
    
    tensorflow::ClientSession session(root, session_opts);
    
    // Step 7: Performance profiling with adaptive warmup
    // Configuration: max warmup iterations, min warmup runs for stability check, stability threshold
    std::vector<double> warmupTimes;
    warmupTimes.reserve(config.maxWarmupRuns);
    
    std::vector<tensorflow::Tensor> outputs_tf; // Reuse for all runs
    
    // Adaptive warmup loop
    int warmupIter = 0;
    bool stable = false;
    
    llvm::outs() << "[Profiling] Starting adaptive warmup...\n";
    
    while (warmupIter < config.maxWarmupRuns) {
        outputs_tf.clear();
        auto startTime = std::chrono::high_resolution_clock::now();
        
        tensorflow::Status run_status = session.Run(feeds, {custom_op}, &outputs_tf);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double timeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        if (!run_status.ok()) {
            llvm::errs() << "Error during warmup run " << (warmupIter + 1) << ": " << run_status.ToString() << "\n";
            return false;
        }
        
        warmupTimes.push_back(timeMs);
        warmupIter++;
        
        // Check if we have enough runs for stability check
        if (warmupIter >= config.minWarmupRuns) {
            // Get last 5 runs
            double minTime = warmupTimes[warmupIter - config.minWarmupRuns];
            double maxTime = warmupTimes[warmupIter - config.minWarmupRuns];
            for (int j = 1; j < config.minWarmupRuns; ++j) {
                double t = warmupTimes[warmupIter - config.minWarmupRuns + j];
                if (t < minTime) minTime = t;
                if (t > maxTime) maxTime = t;
            }
            
            // Scheme B: Check if max-min difference is within threshold
            double avgTime = 0.0;
            for (int j = 0; j < config.minWarmupRuns; ++j) {
                avgTime += warmupTimes[warmupIter - config.minWarmupRuns + j];
            }
            avgTime /= config.minWarmupRuns;
            
            double diff = maxTime - minTime;
            double ratio = (avgTime > 0) ? (diff / avgTime) : 0.0;
            
            if (ratio <= config.stabilityThreshold) {
                stable = true;
                llvm::outs() << "[Profiling] Warmup stable after " << warmupIter << " iterations\n";
                llvm::outs() << "[Profiling] Last " << config.minWarmupRuns << " runs: "
                            << "min=" << minTime << "ms, max=" << maxTime << "ms, "
                            << "avg=" << avgTime << "ms, diff_ratio=" << (ratio * 100) << "%\n";
                break;
            }
        }
    }
    
    if (!stable) {
        llvm::outs() << "[Profiling] Warmup reached max iterations (" << config.maxWarmupRuns << "), proceeding with statistics\n";
        if (warmupIter >= config.minWarmupRuns) {
            double minTime = warmupTimes[warmupIter - config.minWarmupRuns];
            double maxTime = warmupTimes[warmupIter - config.minWarmupRuns];
            for (int j = 1; j < config.minWarmupRuns; ++j) {
                double t = warmupTimes[warmupIter - config.minWarmupRuns + j];
                if (t < minTime) minTime = t;
                if (t > maxTime) maxTime = t;
            }
            llvm::outs() << "[Profiling] Last " << config.minWarmupRuns << " runs: "
                        << "min=" << minTime << "ms, max=" << maxTime << "ms\n";
        }
    }
    
    // Run m iterations for statistics (starting from n+1)
    int statRuns = config.statRuns; // Use numRuns as m from the user's diagram
    std::vector<double> statTimes;
    statTimes.reserve(statRuns);
    
    llvm::outs() << "[Profiling] Running " << statRuns << " iterations for statistics...\n";
    
    for (int i = 0; i < statRuns; ++i) {
        outputs_tf.clear();
        auto startTime = std::chrono::high_resolution_clock::now();
        
        tensorflow::Status run_status = session.Run(feeds, {custom_op}, &outputs_tf);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double timeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        if (!run_status.ok()) {
            llvm::errs() << "Error during statistics run " << (i + 1) << ": " << run_status.ToString() << "\n";
            return false;
        }
        
        statTimes.push_back(timeMs);
    }
    
    // Calculate performance metrics (only average time)
    double avgTime = std::accumulate(statTimes.begin(), statTimes.end(), 0.0) / statTimes.size();
    
    metrics = PerformanceMetrics(libPath, avgTime);
    
    llvm::outs() << "[Perf] Avg time: " << metrics.execTimeMs << " ms\n";
    
    // Step 8: Final run for result extraction
    outputs_tf.clear();
    tensorflow::Status run_status = session.Run(feeds, {custom_op}, &outputs_tf);
    if (!run_status.ok()) {
        llvm::errs() << "Error during final inference: " << run_status.ToString() << "\n";
        return false;
    }
        
    // Step 9: Process output
    if (outputs_tf.empty()) {
        llvm::errs() << "Warning: No output tensors\n";
        return false;
    }
    
    auto& result_tensor = outputs_tf[0];
    auto ctx = inputs->getInputs().begin()->second.getContext();
    storeOutputToMLIR(outputs, result_tensor, ctx);
    
    // Cleanup handled by RAII (TFLibraryHandle destructor)
    return true;
}

} // namespace atir
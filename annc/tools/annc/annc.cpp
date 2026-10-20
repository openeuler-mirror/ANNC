#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <unistd.h>

namespace fs = std::filesystem;

using namespace std;

static const char* KERNEL_LIB_PATH = ANNC_KERNEL_LIB_PATH;

string getKernelLibPath() {
    const char* envPath = getenv("ANNC_KERNEL_LIB_PATH");
    if (envPath && envPath[0] != '\0') {
        return envPath;
    }
    return KERNEL_LIB_PATH;
}

#ifndef ANNC_AARCH64_GEMM_KERNELS_AVAILABLE
#define ANNC_AARCH64_GEMM_KERNELS_AVAILABLE 0
#endif

static string getClangPath() {
    const char* env_clang = getenv("ANNC_CLANG");
    if (env_clang && env_clang[0] != '\0') {
        return string(env_clang);
    }
    return "clang";
}

#ifdef ANNC_KERNEL_MLAS
static fs::path absolutePath(const fs::path& path) {
    if (!path.is_relative()) return path;
    std::error_code ec;
    fs::path absolute = fs::absolute(path, ec);
    return ec ? path : absolute;
}

// Resolve the four MLAS archives from the ORT build tree at kernel-link time.
// ANNC_EP_ORT_BUILD_TREE is the ORT --build_dir Release/ tree used to build
// the EP.  Returns false when the variable is unset or any archive is missing:
// linking a kernel that calls MlasGemmBatch without them would produce a .so
// with unresolved symbols that only fails at dlopen time.
static bool getMlasArchiveArgs(string& args) {
    args.clear();
    const char* ortBuildTree = getenv("ANNC_EP_ORT_BUILD_TREE");
    if (!ortBuildTree || ortBuildTree[0] == '\0') {
        cerr << "ANNC_KERNEL_MLAS: ANNC_EP_ORT_BUILD_TREE is not set; it must "
                "point to the ORT build tree (<build_dir>/Linux/Release)."
             << endl;
        return false;
    }
    fs::path tree = absolutePath(ortBuildTree);
    const std::vector<std::pair<string, fs::path>> archives = {
        {"libonnxruntime_mlas.a", tree / "libonnxruntime_mlas.a"},
        {"libkleidiai.a", tree / "_deps/kleidiai-build/libkleidiai.a"},
        {"libonnxruntime_common.a", tree / "libonnxruntime_common.a"},
        {"libcpuinfo.a", tree / "_deps/pytorch_cpuinfo-build/libcpuinfo.a"},
    };
    for (const auto& entry : archives) {
        const fs::path& archive = entry.second;
        if (!fs::exists(archive)) {
            cerr << "ANNC_KERNEL_MLAS: required MLAS archive " << archive
                 << " is missing under ANNC_EP_ORT_BUILD_TREE=" << tree
                 << "; refusing to link a kernel with unresolved "
                    "MlasGemmBatch symbols." << endl;
            return false;
        }
        args += " \"" + archive.string() + "\"";
    }
    return true;
}
#endif

// 
class CommandExecutor {
public:
    static bool executeCommand(const string& command, string& output) {
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            throw runtime_error("popen() failed!");
        }
        
        char buffer[128];
        output.clear();
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        
        int exitCode = pclose(pipe);
        return exitCode == 0;
    }
    
    static bool executeCommand(const string& command) {
        string output;
        return executeCommand(command, output);
    }
};

// 
class CompilationConfig {
public:
    string inputFile;
    string outputFile;
    string testFile;
    string programDir;
    string packedRhsC;
    string kernelLibPath;
    bool verbose = false;
    bool sharedLibrary = false;
    vector<string> extraArgs;
    vector<string> linkObjects;
    string mLirSymbolName = "annc";  // step4.o_mlirannc
    int M = 0, K = 0, N = 0;  // .mlir

    CompilationConfig() = default;
    
    void parseArgs(int argc, char **argv) {
        programDir = fs::path(argv[0]).parent_path().string();
        for (int i = 1; i < argc; i++) {
            string arg = argv[i];
            if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            } else if (arg == "-o") {
                if (i + 1 < argc) {
                    outputFile = argv[++i];
                }
            } else if (arg == "-t" || arg == "--test") {
                if (i + 1 < argc) {
                    testFile = argv[++i];
                }
            } else if (arg == "--kernel-lib-path") {
                if (i + 1 < argc) {
                    kernelLibPath = argv[++i];
                }
            } else if (arg == "--packed-rhs-c") {
                if (i + 1 < argc) {
                    packedRhsC = argv[++i];
                }
            } else if (arg == "--shared" || arg == "-shared") {
                sharedLibrary = true;
            } else if (arg.find("--") == 0) {
                extraArgs.push_back(arg);
            } else if (inputFile.empty()) {
                inputFile = arg;
            }
        }
        
        if (inputFile.empty()) {
            throw runtime_error("Input file not specified");
        }

        if (kernelLibPath.empty()) {
            kernelLibPath = getKernelLibPath();
        }
        if (!fs::is_directory(kernelLibPath)) {
            throw runtime_error("Kernel library directory does not exist: " +
                                kernelLibPath);
        }

        if (outputFile.empty()) {
            if (sharedLibrary) {
                outputFile = fs::path(inputFile).stem().string() + ".so";
            } else {
                outputFile = fs::path(inputFile).stem().string() + ".out";
            }
        }
        
        if (testFile.empty()) {
            testFile = "test.c";
        }
    }
};

// 
class CompilationPipeline {
private:
    CompilationConfig config;
    string workingDir;
    fs::path tempDir;
    
    void createTempDir() {
        auto now = chrono::system_clock::now();
        auto micros = chrono::duration_cast<chrono::microseconds>(
            now.time_since_epoch()).count();
        string baseName = "annc_temp_" + to_string(micros) + "_" +
                          to_string(static_cast<long long>(getpid()));
        for (int attempt = 0; attempt < 100; ++attempt) {
            tempDir = fs::current_path() /
                      (baseName + "_" + to_string(attempt));
            if (fs::create_directory(tempDir)) {
                workingDir = fs::current_path();
                return;
            }
        }
        throw runtime_error("Failed to create unique temporary directory");
    }
    
    void cleanupTempDir() {
        if (!tempDir.empty()) {
            fs::remove_all(tempDir);
        }
    }
    
    void log(const string& message) {
        if (config.verbose) {
            cerr << "[ANNC] " << message << endl;
        }
    }
    
    void logCommand(const string& command) {
        if (config.verbose) {
            cerr << "[ANNC] : " << command << endl;
        }
    }

    bool validateGemmKernelAvailability(const fs::path& objectFile) {
#if ANNC_AARCH64_GEMM_KERNELS_AVAILABLE
        return true;
#else
        string output;
        string command = "nm -u \"" + objectFile.string() + "\"";
        if (!CommandExecutor::executeCommand(command, output)) {
            log("Error: Failed to inspect unresolved object symbols");
            return false;
        }
        if (output.find("annc_aarch64_neon_") == string::npos &&
            output.find("annc_aarch64_sve_") == string::npos) {
            return true;
        }
        cerr << "ANNC error: AArch64 GEMM output requires the optional "
                "microkernel archive. Configure "
                "ANNC_AARCH64_GEMM_KERNEL_LIBRARY with a valid library."
             << endl;
        return false;
#endif
    }

    void appendPrepackedSource(string& command) {
        if (!config.packedRhsC.empty() && fs::exists(config.packedRhsC))
            command += " \"" + config.packedRhsC + "\"";
    }

public:
    CompilationPipeline(const CompilationConfig& cfg) : config(cfg) {
        createTempDir();
    }
    
    ~CompilationPipeline() {
        cleanupTempDir();
    }
    
    bool run() {
        log("Starting compilation pipeline");
        log("Input file: " + config.inputFile);
        log("Output file: " + config.outputFile);
        log("Working directory: " + tempDir.string());
        log("Mode: " + string(config.sharedLibrary ? "Shared Library" : "Executable"));
        
        try {
            //  0: 从输入 MLIR 提取 M/K/N；失败时保留默认值（0）
            (void)extractDimensionsFromMLIR();

            //  1: MLIR -> LLVM-dialect MLIR
            if (!step1_MLIRToLLVMDialect()) {
                log("Step 1 failed");
                return false;
            }
            
            //  2: MLIR -> LLVM IR
            if (!step2_MLIRToLLVMIR()) {
                log("Step 2 failed");
                return false;
            }
            
            //  3: LLVM IR ->  LLVM IR
            if (!step3_OptimizeLLVMIR()) {
                log("Step 3 failed");
                return false;
            }
            
            //  4: LLVM IR -> object
            if (!step4_LLVMIRToObject()) {
                log("Step 4 failed");
                return false;
            }

            if (!step4b_GenerateAndAssembleAotKernels()) {
                log("Step 4b failed");
                return false;
            }
            
            //  5: 
            if (config.sharedLibrary) {
                //  5:  (.so)
                if (!step5_CreateSharedLibrary()) {
                    log("Step 5 (shared library) failed");
                    return false;
                }
                
                //  6: 
                if (!step6_CompileDynamicTest()) {
                    log("Step 6 (dynamic test) failed");
                    return false;
                }
            } else {
                //  5:  C 
                if (!step5_LinkWithCTest()) {
                    log("Step 5 (executable) failed");
                    return false;
                }
            }
            
            log("Compilation completed successfully");
            return true;
            
        } catch (const exception& e) {
            cerr << "Error: " << e.what() << endl;
            return false;
        }
    }
    
private:
    // .mlir M, N, K
    bool extractDimensionsFromMLIR() {
        log("Extracting matrix dimensions from MLIR file");

        ifstream inputFile(config.inputFile);
        if (!inputFile.is_open()) {
            log("Error: Cannot open MLIR file: " + config.inputFile);
            return false;
        }

        string line;
        bool foundDimensions = false;

        while (getline(inputFile, line)) {
            //  memref<...xf32> 
            size_t funcPos = line.find("func.func");
            if (funcPos != string::npos) {
                //  memref 
                size_t memrefPos = line.find("memref<");
                while (memrefPos != string::npos) {
                    size_t start = memrefPos + 6; // "memref<" 
                    size_t end = line.find('>', start);
                    if (end == string::npos) break;

                    string memrefType = line.substr(start, end - start);
                    log("Parsing memref type: " + memrefType);

                    //  '<' 
                    if (memrefType.front() == '<') {
                        memrefType = memrefType.substr(1);
                    }

                    //  "1536x1344xf32"  "4x8xf32"
                    size_t x1 = memrefType.find('x');
                    size_t x2 = memrefType.rfind('x');

                    auto isNumber = [](const string &value) {
                        if (value.empty()) return false;
                        for (char c : value) {
                            if (c < '0' || c > '9') return false;
                        }
                        return true;
                    };

                    if (x1 != string::npos && x2 != string::npos && x1 < x2) {
                        string first = memrefType.substr(0, x1);
                        string second =
                            memrefType.substr(x1 + 1, x2 - x1 - 1);

                        // 只接受静态二维形状；动态维（?）或多维形状跳过。
                        if (isNumber(first) && isNumber(second)) {
                            try {
                                int dim1 = stoi(first);
                                int dim2 = stoi(second);

                                // 位置启发式：假定 %arg0/%arg1 为静态二维
                                // A(M x K)/B(K x N)；动态或非 matmul 输入可能
                                // 不满足该前提，此时不要依赖 -DM/-DK/-DN。
                                // %arg0: memref<M x K x f32> (A)
                                // %arg1: memref<K x N x f32> (B)
                                // %arg2: memref<M x N x f32> (C)
                                // %arg3: memref<N x f32> (bias)
                                if (config.M == 0 && config.K == 0) {
                                    config.M = dim1;
                                    config.K = dim2;
                                } else if (config.N == 0) {
                                    config.N = dim2;
                                }

                                foundDimensions = true;
                                log("Found dimensions: M=" +
                                    to_string(config.M) + ", K=" +
                                    to_string(config.K) + ", N=" +
                                    to_string(config.N));
                            } catch (const exception &e) {
                                log("Error parsing dimensions from memref type: " +
                                    memrefType);
                            }
                        }
                    }

                    // 无论是否解析成功都推进，避免 catch 分支死循环。
                    memrefPos = line.find("memref<", end);
                }
            }
        }

        inputFile.close();

        if (!foundDimensions) {
            log("Warning: Could not extract dimensions from MLIR file, using defaults");
            return false;
        }

        return true;
    }

    bool step1_MLIRToLLVMDialect() {
        log("Step 1: MLIR -> LLVM-dialect MLIR");
        
        string inputFile = fs::path(config.inputFile).filename().string();
        string outputFile = "step1_llvm.mlir";
        
        string command = "mlir-opt \"" + config.inputFile + "\"";
        command += " --canonicalize";
        command += " --cse";
        command += " --lower-affine";
        command += " --convert-scf-to-cf";
        command += " --canonicalize";
        command += " --cse";
        command += " --convert-cf-to-llvm";
        command += " --convert-arith-to-llvm";
        command += " --convert-index-to-llvm";
        command += " --convert-func-to-llvm";
        command += " --expand-strided-metadata";
        command += " --finalize-memref-to-llvm";
        command += " --canonicalize";
        command += " --cse";
        command += " --reconcile-unrealized-casts";
        command += " --affine-super-vectorize";
        command += " --set-llvm-module-datalayout";
        command += " --reconcile-unrealized-casts";
        command += " -o \"" + (tempDir / outputFile).string() + "\"";
        
        if (config.verbose) {
            command += " --mlir-print-op-generic";
        }
        
        log("Executing: " + command);
        return CommandExecutor::executeCommand(command);
    }
    
    bool step2_MLIRToLLVMIR() {
        log("Step 2: MLIR -> LLVM IR");
        
        string inputFile = "step1_llvm.mlir";
        string outputFile = "step2.ll";
        
        string command = "mlir-translate --mlir-to-llvmir \"" + (tempDir / inputFile).string() + "\"";
        command += " -o \"" + (tempDir / outputFile).string() + "\"";
        
        logCommand(command);
        return CommandExecutor::executeCommand(command);
    }
    
    bool step3_OptimizeLLVMIR() {
        log("Step 3: LLVM IR ->  LLVM IR");
        
        string inputFile = "step2.ll";
        string outputFile = "step3_opt.ll";
        
        string command = "opt -O3 -march=native \"" + (tempDir / inputFile).string() + "\"";
        command += " -S";
        command += " -o \"" + (tempDir / outputFile).string() + "\"";
        
        logCommand(command);
        return CommandExecutor::executeCommand(command);
    }
    
    bool step4_LLVMIRToObject() {
        log("Step 4: LLVM IR -> object");
        
        string inputFile = "step3_opt.ll";
        string outputFile = "step4.o";
        
        string command = "llc -filetype=obj \"" + (tempDir / inputFile).string() + "\"";
        command += " -o \"" + (tempDir / outputFile).string() + "\"";
        
        logCommand(command);
        bool success = CommandExecutor::executeCommand(command);
        
        // 
        if (success) {
            if (!generateDimensionsHeader()) {
                log("Failed to generate dimensions header");
                return false;
            }
        }

        // _mlir
        if (success && config.sharedLibrary) {
            extractMLIRSymbolName();
        }
        
        return success;
    }
    
    // 
    bool generateDimensionsHeader() {
        log("Generating dimensions header file");

        string headerContent = "// Auto-generated dimensions from MLIR file\n";
        headerContent += "#ifndef ANNC_DIMENSIONS_H\n";
        headerContent += "#define ANNC_DIMENSIONS_H\n\n";
        headerContent += "// Matrix dimensions extracted from MLIR file\n";
        headerContent += "#define M " + to_string(config.M) + "\n";
        headerContent += "#define K " + to_string(config.K) + "\n";
        headerContent += "#define N " + to_string(config.N) + "\n\n";
        headerContent += "#endif // ANNC_DIMENSIONS_H\n";

        string headerFile = (tempDir / "dimensions.h").string();
        ofstream headerOut(headerFile);
        if (!headerOut.is_open()) {
            log("Error: Cannot create header file: " + headerFile);
            return false;
        }

        headerOut << headerContent;
        headerOut.close();

        log("Generated header file: " + headerFile);
        log("Dimensions: M=" + to_string(config.M) + ", K=" + to_string(config.K) + ", N=" + to_string(config.N));

        return true;
    }

    bool step4b_GenerateAndAssembleAotKernels() {
        log("Step 4b: export and assemble AOT generated kernels");

        string exportTool =
            (fs::path(config.programDir) / "annc-model-kernel-export").string();
        fs::path aotDir = tempDir / "aot-kernels";
        fs::create_directories(aotDir);
        string exportCommand = "\"" + exportTool + "\" \"" +
                               (tempDir / "step1_llvm.mlir").string() +
                               "\" --output-dir \"" + aotDir.string() + "\"";
        logCommand(exportCommand);
        if (!CommandExecutor::executeCommand(exportCommand)) {
            log("AOT kernel export failed");
            return false;
        }

        fs::path kernelsDir = aotDir / "model_generated_kernels";
        if (!fs::exists(kernelsDir)) return true;

        const char* asmRootEnv = getenv("ANNC_GEMM_ASM_SOURCE_DIR");
        fs::path asmRoot = asmRootEnv && *asmRootEnv
                               ? fs::path(asmRootEnv)
                               : fs::path(ANNC_GEMM_ASM_SOURCE_DIR);
        if (asmRoot.empty() || !fs::exists(asmRoot / "Sve") ||
            !fs::exists(asmRoot / "Neon")) {
            cerr << "ANNC_GEMM_ASM_SOURCE_DIR must point to a GEMM-ASM checkout "
                    "containing Sve/ and Neon/ (set the environment variable, or "
                    "configure ANNC with "
                    "-DANNC_GEMM_ASM_SOURCE_DIR=<checkout>)"
                 << endl;
            return false;
        }

        string includeArgs = " -I \"" + (asmRoot / "Sve").string() +
                             "\" -I \"" + (asmRoot / "Neon").string() + "\"";
        vector<fs::path> sources;
        for (const auto& entry : fs::directory_iterator(kernelsDir)) {
            if (entry.path().extension() == ".S")
                sources.push_back(entry.path());
        }
        sort(sources.begin(), sources.end());
        for (const fs::path& source : sources) {
            fs::path object = source;
            object.replace_extension(".o");
            const bool sve = source.filename().string().find("_sve_") !=
                             string::npos;
            string command = getClangPath() +
                             (sve ? " -c -march=armv8-a+sve"
                                  : " -c -march=armv8-a") +
                             includeArgs + " \"" + source.string() +
                             "\" -o \"" + object.string() + "\"";
            logCommand(command);
            if (!CommandExecutor::executeCommand(command)) {
                log("AOT kernel assembly failed: " + source.string());
                return false;
            }
            config.linkObjects.push_back(object.string());
        }
        return true;
    }

    // step4.o_mlir
    void extractMLIRSymbolName() {
        log("Extracting defined MLIR C interface symbols from step4.o");
        
        string command = "nm --defined-only --extern-only \"" + (tempDir / "step4.o").string() + "\"";
        string output;
        
        if (!CommandExecutor::executeCommand(command, output)) {
            log("Failed to extract symbol names");
            return;
        }
        
        // Only consider symbols that are both defined in this object and
        // externally visible. This avoids treating imported custom call
        // declarations as the final shared-library entry symbol.
        stringstream ss(output);
        string line;
        while (getline(ss, line)) {
            string address;
            string symbolType;
            string symbolName;
            stringstream lineStream(line);
            lineStream >> address >> symbolType >> symbolName;

            if (symbolName.find("_mlir_ciface_") == 0) {
                string libName = symbolName.substr(13); // "_mlir_ciface_" = 13
                for (char &c : libName) {
                    if (!isalnum(c) && c != '_') {
                        c = '_';
                    }
                }
                config.mLirSymbolName = libName;
                log("Found exported MLIR C interface symbol: " + symbolName +
                    ", library name: " + libName + ".so");
                break;
            }
        }
        
        if (config.mLirSymbolName.empty()) {
            log("Warning: No defined _mlir_ciface_ symbol found, using default name");
        }
    }
    
    bool step5_LinkWithCTest() {
        log("Step 5:  C ");
        
        string inputFile = "step4.o";
        string outputFile = config.outputFile;
        if (!validateGemmKernelAvailability(tempDir / inputFile)) return false;
        
        // 
        if (!fs::exists(config.testFile)) {
            log("Error: " + config.testFile + " not found, skipping linking step");
            return false;
        }
        
        string command = getClangPath() + " -O3 \"" + (tempDir / inputFile).string() + "\"";
        appendPrepackedSource(command);
        command += " \"" + config.testFile + "\"";
        for (const string& objectFile : config.linkObjects)
            command += " \"" + objectFile + "\"";
        command += " -L" + config.kernelLibPath + " -lANNCBuiltinKernels";
#if ANNC_AARCH64_GEMM_KERNELS_AVAILABLE
        command += " -L" + config.kernelLibPath + " -lannc_gemm_microkernels";
#endif
        command += " -Wl,--whole-archive -L" + config.kernelLibPath +
                   " -lANNCThreadPool -Wl,--no-whole-archive";
        command += " -L" + config.kernelLibPath + " -lANNCGemmSupport";
        command += " -L" + config.kernelLibPath +
                   " -lLLVMSupport -lLLVMDemangle";
#ifdef ANNC_KERNEL_MLAS
        // MLAS GEMM kernels: resolve the four archives at link time from
        // ANNC_EP_ORT_BUILD_TREE.  A missing archive is a hard error: the
        // kernel calls MlasGemmBatch, so a .so linked without these archives
        // can only fail later at dlopen time.
        string mlasArgs;
        if (!getMlasArchiveArgs(mlasArgs)) {
            log("Error: MLAS archive resolution failed; aborting kernel link");
            return false;
        }
        command += " -Wl,--start-group " + mlasArgs +
                   " -Wl,--end-group -lpthread -ldl -lrt";
#endif
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
        command += " -L" KDNN_LIB_DIR " -lkdnn";
        command += " -Wl,-rpath," KDNN_LIB_DIR;
        command += " " ANNC_KDNN_OPENMP_LINK_FLAGS;
#endif
        // 仅在成功提取静态形状时覆盖 driver 的默认维度。
        if (config.M > 0 && config.K > 0 && config.N > 0) {
            command += " -DM=" + to_string(config.M);
            command += " -DK=" + to_string(config.K);
            command += " -DN=" + to_string(config.N);
        }
        command += " -o \"" + (fs::path(config.outputFile)).string() + "\"";
        command += " -lstdc++ -lm";
        
        logCommand(command);
        return CommandExecutor::executeCommand(command);
    }
    
    bool step5_CreateSharedLibrary() {
        log("Step 5: ");
        
        string inputFile = "step4.o";
        if (!validateGemmKernelAvailability(tempDir / inputFile)) return false;
        string sharedLibName = config.outputFile.empty()
                                   ? config.mLirSymbolName + ".so"
                                   : config.outputFile;
        
        string command = getClangPath() + " -shared -fPIC -O3 \"" + (tempDir / inputFile).string() + "\"";
        appendPrepackedSource(command);
        for (const string& objectFile : config.linkObjects)
            command += " \"" + objectFile + "\"";
        command += " -L" + config.kernelLibPath + " -lANNCBuiltinKernels";
#if ANNC_AARCH64_GEMM_KERNELS_AVAILABLE
        command += " -L" + config.kernelLibPath + " -lannc_gemm_microkernels";
#endif
        command += " -Wl,--whole-archive -L" + config.kernelLibPath +
                   " -lANNCThreadPool -Wl,--no-whole-archive";
        command += " -L" + config.kernelLibPath + " -lANNCGemmSupport";
        command += " -L" + config.kernelLibPath +
                   " -lLLVMSupport -lLLVMDemangle";
#ifdef ANNC_KERNEL_MLAS
        // MLAS GEMM kernels: resolve the four archives at link time from
        // ANNC_EP_ORT_BUILD_TREE.  A missing archive is a hard error: the
        // kernel calls MlasGemmBatch, so a .so linked without these archives
        // can only fail later at dlopen time.
        string mlasArgs;
        if (!getMlasArchiveArgs(mlasArgs)) {
            log("Error: MLAS archive resolution failed; aborting kernel link");
            return false;
        }
        command += " -Wl,--start-group " + mlasArgs +
                   " -Wl,--end-group -lpthread -ldl -lrt";
#endif
#ifdef ANNC_ENABLE_KDNN_ADAPTOR
        command += " -L" KDNN_LIB_DIR " -lkdnn";
        command += " -Wl,-rpath," KDNN_LIB_DIR;
        command += " " ANNC_KDNN_OPENMP_LINK_FLAGS;
#endif
        command += " -lstdc++";
        command += " -o \"" + (fs::path(sharedLibName)).string() + "\"";
        
        logCommand(command);
        return CommandExecutor::executeCommand(command);
    }
    
    bool step6_CompileDynamicTest() {
        log("Step 6: ");
        
        // 
        if (!fs::exists(config.testFile)) {
            log("Warning: " + config.testFile + " not found, skipping dynamic test compilation");
            return true; // 
        }
        
        // 
        string libName = config.mLirSymbolName + ".so";
        log("Setting ANNC_LIBRARY_NAME environment variable to: " + libName);
        setenv("ANNC_LIBRARY_NAME", libName.c_str(), 1);
        
        // 
        string command = getClangPath() + " -O3 \"" + config.testFile + "\"";
        // 
        // 仅在成功提取静态形状时覆盖 driver 的默认维度。
        if (config.M > 0 && config.K > 0 && config.N > 0) {
            command += " -DM=" + to_string(config.M);
            command += " -DK=" + to_string(config.K);
            command += " -DN=" + to_string(config.N);
        }
        command += " -o \"" + (fs::path(config.outputFile)).string() + "\"";
        command += " -ldl -lm";
        
        logCommand(command);
        bool result = CommandExecutor::executeCommand(command);
        
        // 
        unsetenv("ANNC_LIBRARY_NAME");
        log("Cleared ANNC_LIBRARY_NAME environment variable");
        
        return result;
    }
};

void printUsage() {
    cout << "ANNC Driver - MLIR to Executable/Shared Library Compiler" << endl;
    cout << "Usage: annc [options] <input.mlir>" << endl;
    cout << "Options:" << endl;
    cout << "  -o <file>              Output file (default: <input stem>.out or <input stem>.so)" << endl;
    cout << "  -t <file>              Test C file (default: test.c)" << endl;
    cout << "  --kernel-lib-path <dir> Kernel library directory (default: env ANNC_KERNEL_LIB_PATH or install prefix)" << endl;
    cout << "  --packed-rhs-c <file>  Link generated GEMM packed RHS C source" << endl;
    cout << "  -v, --verbose          Verbose output" << endl;
    cout << "  --shared, -shared      Generate shared library (.so) instead of executable" << endl;
    cout << "  --help                 Show this help message" << endl;
    cout << endl;
    cout << "Examples:" << endl;
    cout << "  annc demo.mlir                    # Compile to executable (demo.out)" << endl;
    cout << "  annc -shared demo.mlir            # Compile to shared library (demo.so)" << endl;
    cout << "  annc -o myapp demo.mlir           # Compile to myapp" << endl;
    cout << "  annc -shared -o mylib demo.mlir   # Compile to mylib.so" << endl;
}

int main(int argc, char **argv) {  
    if (argc == 1) {
        printUsage();
        return 1;
    }
    
    if (argc == 2 && (string(argv[1]) == "--help" || string(argv[1]) == "-h")) {
        printUsage();
        return 0;
    }
    
    try {
        CompilationConfig config;
        config.parseArgs(argc, argv);
        
        CompilationPipeline pipeline(config);
        bool success = pipeline.run();
        
        return success ? 0 : 1;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        printUsage();
        return 1;
    }
}

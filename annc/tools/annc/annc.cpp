#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <fstream>

namespace fs = std::filesystem;

using namespace std;

static const char* KERNEL_LIB_PATH = ANNC_KERNEL_LIB_PATH;
static const char* DEFAULT_C_COMPILER = ANNC_C_COMPILER;

string getKernelLibPath() {
    return KERNEL_LIB_PATH;
}

string getCCompiler() {
    const char* env = getenv("ANNC_CLANG");
    if (env && env[0] != '\0') {
        return string(env);
    }
    return string(DEFAULT_C_COMPILER);
}

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
    bool verbose = false;
    bool sharedLibrary = false;
    vector<string> extraArgs;
    string mLirSymbolName = "annc";  // step4.o_mlirannc
    int M = 0, K = 0, N = 0;  // .mlir

    CompilationConfig() = default;
    
    void parseArgs(int argc, char **argv) {
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
        tempDir = fs::current_path() / ("annc_temp_" + to_string(time(nullptr)));
        fs::create_directories(tempDir);
        workingDir = fs::current_path();
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
            //  0:  MLIR 
            if (!extractDimensionsFromMLIR()) {
                log("Failed to extract dimensions from MLIR file");
                return false;
            }

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

                    if (x1 != string::npos && x2 != string::npos && x1 < x2) {
                        try {
                            int dim1 = stoi(memrefType.substr(0, x1));
                            int dim2 = stoi(memrefType.substr(x1 + 1, x2 - x1 - 1));

                            // 
                            // %arg0: memref<M x K x f32> ( A)
                            // %arg1: memref<K x N x f32> ( B)
                            // %arg2: memref<M x N x f32> ( C)
                            // %arg3: memref<N x f32> (bias)

                            if (config.M == 0 && config.K == 0) {
                                //  memref  M x K
                                config.M = dim1;
                                config.K = dim2;
                            } else if (config.N == 0) {
                                //  memref  K x N N = dim2
                                config.N = dim2;
                            }

                            foundDimensions = true;
                            log("Found dimensions: M=" + to_string(config.M) + ", K=" + to_string(config.K) + ", N=" + to_string(config.N));
                        } catch (const exception& e) {
                            log("Error parsing dimensions from memref type: " + memrefType);
                            continue;
                        }
                    }

                    //  memref
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
        
        // 
        if (!fs::exists(config.testFile)) {
            log("Error: " + config.testFile + " not found, skipping linking step");
            return false;
        }
        
        string command = getCCompiler() + " -O3 \"" + (tempDir / inputFile).string() + "\"";
        command += " \"" + config.testFile + "\"";
        command += " -L" + getKernelLibPath() + " -lANNCBuiltinKernels";
        command += " -Wl,--whole-archive -L" + getKernelLibPath() +
                   " -lANNCThreadPool -Wl,--no-whole-archive";
        //
        command += " -DM=" + to_string(config.M);
        command += " -DK=" + to_string(config.K);
        command += " -DN=" + to_string(config.N);
        command += " -o \"" + (fs::path(config.outputFile)).string() + "\"";
        command += " -lm";
        
        logCommand(command);
        return CommandExecutor::executeCommand(command);
    }
    
    bool step5_CreateSharedLibrary() {
        log("Step 5: ");
        
        string inputFile = "step4.o";
        string sharedLibName = config.outputFile.empty()
                                   ? config.mLirSymbolName + ".so"
                                   : config.outputFile;
        
        string command = getCCompiler() + " -shared -fPIC -O3 \"" + (tempDir / inputFile).string() + "\"";
        command += " -L" + getKernelLibPath() + " -lANNCBuiltinKernels";
        command += " -Wl,--whole-archive -L" + getKernelLibPath() +
                   " -lANNCThreadPool -Wl,--no-whole-archive";
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
        string command = getCCompiler() + " -O3 \"" + config.testFile + "\"";
        // 
        command += " -DM=" + to_string(config.M);
        command += " -DK=" + to_string(config.K);
        command += " -DN=" + to_string(config.N);
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
    cout << "  -o <file>              Output file (default: a.out or input.so)" << endl;
    cout << "  -t <file>              Test C file (default: test.c)" << endl;
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

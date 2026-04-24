#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Pimp/PimpOps.h"
#include "Dialect/Pimp/Passes/Passes.h"
#include "mlir/Pass/PassManager.h"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <algorithm>
#include <unordered_map>

using namespace llvm;
using namespace mlir;

namespace pimp {
    class PimpLLMCodeGenPass : public PimpLLMCodeGenBase<PimpLLMCodeGenPass> {

    private:
        // the deepest common path of all files
        std::string path = std::string(__FILE__).substr(0, std::string(__FILE__).find("/LLMCodeGen.cpp"));

        std::string readFile(const std::string& filename) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Cannot Read from File: " << filename << std::endl;
                return "";
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }

        bool writeFile(const std::string& filename, const std::string& content) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Cannot Write into File: " << filename << std::endl;
                return false;
            }

            file << content;
            return true;
        }

        std::string getPromptTemplate() {
            return readFile(path + "/templates/prompt_template.txt");
        }

        bool dumpToFile(mlir::Operation* op, const std::string& filename) {
            FILE* originalStderr = stderr;
            FILE* file = freopen(filename.c_str(), "w", stderr);
            if (!file) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return false;
            }
            op->dump();
            fflush(stderr);
            fclose(file);
            stderr = originalStderr;
            return true;
        }

        std::string getModule() {
            if (dumpToFile(getOperation(), path + "/templates/modules/module.txt")) {
                return readFile(path + "/templates/modules/module.txt");
            } else {
                return "";
            }
        }

        std::string getModuleName() {
            std::string module = getModule();
            size_t start_pos = module.find('{', 0) + 1;
            while (true) {
                //  '@' 
                size_t at_pos = module.find('@', start_pos);
                if (at_pos == std::string::npos) {
                    return "";  // '@'
                }

                //  '@'  '('
                size_t paren_pos = module.find('(', at_pos);
                if (paren_pos == std::string::npos) {
                    return "";  // '('
                }

                // 
                std::string func_name = module.substr(at_pos + 1, paren_pos - at_pos - 1);

                // 
                size_t first_non_space = func_name.find_first_not_of(" \t\r\n");
                if (first_non_space != std::string::npos) {
                    size_t last_non_space = func_name.find_last_not_of(" \t\r\n");
                    func_name = func_name.substr(first_non_space, last_non_space - first_non_space + 1);
                }

                // "main"
                if (func_name != "main") {
                    return func_name;  // main
                }

                // main
                start_pos = at_pos + 1;
            }
        }

        // std::string patternName = getPatternName();

        std::string getTFSTemplate() {
            return readFile(path + "/templates/tfs_template.txt");
        }

        std::string getGenCode(int iter, const std::string& patternName) {
            return readFile(path + "/outputs/generated_files/" + std::to_string(iter) + patternName + ".cc");
        }

        std::string getCompileError(int iter, const std::string& patternName) {
            return readFile(path + "/outputs/compile_outputs/" + patternName + std::to_string(iter) + ".txt");
        }

        std::string getTestError(int iter, const std::string& patternName) {
            return readFile(path + "/outputs/test_outputs/" + patternName + std::to_string(iter) + ".txt");
        }

        std::string snakeToCamel(const std::string& snake_str) {
            if (snake_str.empty()) {
                return "";
            }

            std::string result;
            result.reserve(snake_str.size());  // 

            for (size_t i = 0; i < snake_str.size(); ++i) {
                if (snake_str[i] == '_') {
                    continue;  // 
                }

                // 
                if (i == 0 || snake_str[i - 1] == '_') {
                    result.push_back(static_cast<char>(std::toupper(snake_str[i])));
                } else {
                    result.push_back(snake_str[i]);
                }
            }

            return result;
        }

        void buildTFSPrompt(int iter, bool tested) {
            std::string promptTemplate = getPromptTemplate();

            std::string module = getModule();

            std::string patternName = getModuleName();

            std::string tfsTemplate = getTFSTemplate();

            std::string gencode = (iter > 0) ? getGenCode(iter, patternName) : "";

            std::string compileError = (iter > 0) ? getCompileError(iter, patternName) : "";

            std::string testError = (tested) ? getTestError(iter, patternName) : "";

            size_t mlirNamePos = promptTemplate.find("{MLIR_NAME}");
            if (mlirNamePos != std::string::npos) {
                promptTemplate.replace(mlirNamePos, 13, patternName);
            }

            size_t mlirPos = promptTemplate.find("{MLIR_MODULE}");
            if (mlirPos != std::string::npos) {
                promptTemplate.replace(mlirPos, 13, module);
            }
            
            size_t tfsPos = promptTemplate.find("{TFS_TEMPLATE}");
            if (tfsPos != std::string::npos) {
                promptTemplate.replace(tfsPos, 14, tfsTemplate);
            }

            size_t gencodePos = promptTemplate.find("{GEN_CODE}");
            if (gencodePos != std::string::npos) {
                promptTemplate.replace(gencodePos, 10, gencode);
            }

            size_t compileErrorPos = promptTemplate.find("{COMPILE_ERRORS}");
            if (compileErrorPos != std::string::npos) {
                promptTemplate.replace(compileErrorPos, 16, compileError);
            }

            size_t testErrorPos = promptTemplate.find("{TEST_ERRORS}");
            if (testErrorPos != std::string::npos) {
                promptTemplate.replace(testErrorPos, 13, testError);
            }

            std::string opName = snakeToCamel(patternName);

            size_t opNamePos = 0;
            while ((opNamePos = promptTemplate.find("{OP_NAME}", opNamePos)) != std::string::npos) {
                promptTemplate.replace(opNamePos, 9, opName);
                opNamePos += opName.length();
            }

            std::string outputPath = path + "/prompts/tfs_prompt.txt";
            if (!writeFile(outputPath, promptTemplate)) {
                std::cout << "Failed to Write into TFS Prompt" << std::endl;
            }
        }

        int numIter = std::stoi(readFile(path + "/templates/num_iter.txt"));

    public:
        PimpLLMCodeGenPass() = default;

        void runOnOperation() override {
            // const char* key = std::getenv("API_KEY");
            // if (key == nullptr || key[0] == '\0') {
            //     std::cout << "Please set API_KEY" << std::endl;
            //     return
            // }

            std::cout << "If Generation Fails Refer to llm/cert_setup.txt" << std::endl;

            std::string patternName = getModuleName();

            std::cout << "=== PimpLLMCodeGenPass Starting ===" << std::endl;

            // empty generated files from run
            system(("bash " + path + "/scripts/clean.sh").c_str());

            // initialize prompt
            buildTFSPrompt(0, false);

            // indicates whether the code generation succeeds or not
            bool valid = false;
            for (int iter = 1; iter <= numIter; iter++) {
                std::cout << "=== Iteration " << iter << " Started ===" << std::endl;

                // call llmcodegen.py to generate code
                std::cout << "=== Start Generation ===" << std::endl;
                int status = system(("python3 " + path + "/llm/src/llmcodegen.py \"" + path + "\" \"" + std::to_string(iter) + patternName + "\"").c_str());
                // if llmcodegen.py exits with error, something is wrong with llm, should halt further attempts
                if (status != 0) {
                    std::cerr << "=== Generation Failed ===" << std::endl;
                    std::cout << "=== Iteration " << iter << " Ended ===" << std::endl;
                    break;
                }
                std::cout << "=== Generation Complete ===" << std::endl;

                std::cout << "=== Start Compilation ===" << std::endl;
                system(("bash " + path + "/scripts/compile.sh " + patternName + " " + std::to_string(iter)).c_str());
                std::cout << "=== Compilation Complete ===" << std::endl;

                std::cout << "=== Printing Compilation Result ===" << std::endl;
                std::string compileError = getCompileError(iter, patternName);
                std::cout << compileError << std::endl;
                // if (compileError.find("Build completed successfully") == std::string::npos) {
                if (!compileError.empty()) {
                    buildTFSPrompt(iter, false);
                    std::cout << "=== Compilation Failed ===" << std::endl;
                    std::cout << "=== Iteration " << iter << " Finished ===" << std::endl;
                    continue;
                }
                
                std::cout << "=== Start Testing ===" << std::endl;
                system(("bash " + path + "/scripts/test.sh " + patternName + " " + std::to_string(iter)).c_str());
                std::cout << "=== Testing Complete ===" << std::endl;
                
                std::cout << "=== Printing Test Result ===" << std::endl;
                std::string testError = getTestError(iter, patternName);
                std::cout << testError << std::endl;
                if (testError.find("TEST COMPLETED") == std::string::npos) {
                    buildTFSPrompt(iter, true);
                    std::cout << "=== Test Failed ===" << std::endl;
                    std::cout << "=== Iteration " << iter << " Ended ===" << std::endl;
                    continue;
                }
                
                valid = true;
                std::cout << "=== Iteration " << iter << " Finished ===" << std::endl;
                break;
            }

            if (valid) {
                std::cout << "=== GENERATION SUCEEDED ===" << std::endl;
            } else {
                std::cout << "=== GENERATION FAILED ===" << std::endl;
            }

            std::cout << "=== PimpLLMCodeGenPass FINISHED ===" << std::endl;
        }
    };

    std::unique_ptr<OperationPass<ModuleOp>> createPimpLLMCodeGenPass() {
        std::cout << "this is createPimpLLMCodeGenPass" << std::endl;
        return std::make_unique<PimpLLMCodeGenPass>();
    }
}  // namespace pimp
#include <iostream>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>

#include "Adaptor/tensorflow/TFSavedModelParser.h"

using json = nlohmann::json;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> <output_json>\n";
    std::cout << "  model_path: Path to TensorFlow saved model directory\n";
    std::cout << "  output_json: Output JSON file path\n";
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string output_json_path = argv[2];
    
    TFSavedModelParser parser(model_path);
    
    if (!parser.isValid()) {
        std::cerr << "Error: Invalid saved model at path: " << model_path << std::endl;
        return 1;
    }

    std::vector<std::string> output_defs = parser.getModelOutputs();
    if (output_defs.empty()) {
        output_defs = {"output"};
    }

    std::vector<NodeInfo> nodes = parser.parse(output_defs);
    json output_json = parser.toJson(nodes);

    try {
        std::ofstream output_file(output_json_path);
        if (output_file.is_open()) {
            output_file << output_json.dump(4) << std::endl;
            output_file.close();
            std::cout << "JSON output saved to: " << output_json_path << std::endl;
            std::cout << "Parsed " << nodes.size() << " nodes." << std::endl;
        } else {
            std::cerr << "Error: Cannot create output file: " << output_json_path << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing JSON file: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
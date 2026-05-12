#ifndef TFSAVEDMODEL_PARSER_H
#define TFSAVEDMODEL_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <nlohmann/json.hpp>

#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/cc/saved_model/loader.h"

using json = nlohmann::json;

struct NodeInfo {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::string raw_data;
    
    bool isInputNode = false;
    bool isOutputNode = false;
    
    struct OutputInfo {
        std::string name;
        std::string dtype;
        std::vector<int64_t> shape;
        int id = 0;
        
        std::string getName() const {
            if (id == 0) return name;
            return name + ":" + std::to_string(id);
        }
    };
    std::vector<OutputInfo> outputs;

    // TF-specific attributes (for reverse conversion)
    // key: attribute name, value: string representation
    std::map<std::string, std::string> tf_attrs;

    void addOutput(int id, const std::string& name, 
                   const std::vector<int64_t>& s, const std::string& d) {
        OutputInfo out;
        out.shape = s;
        out.dtype = d;
        out.id = id;
        out.name = name;
        outputs.push_back(out);
    }
};

class TFSavedModelParser {
public:
    explicit TFSavedModelParser(const std::string& model_path);
    ~TFSavedModelParser() = default;
    
    // 
    std::vector<NodeInfo> parse(const std::vector<std::string>& output_defs);
    
    //  JSON 
    json toJson(const std::vector<NodeInfo>& nodes);
    
    // 
    std::vector<std::string> getModelOutputs();
    std::vector<std::string> getModelInputs();
    
    // 
    bool isValid() const { return gdef_ != nullptr; }
    
    //  variables  const  tensor 
    void print_tensor_data();
    
    //  variable  const  tensor 
    bool get_tensor_value(const tensorflow::NodeDef* node, std::vector<uint8_t>& tensor_values,
                         std::vector<int64_t>& shape, std::string& dtype);
    
private:
    std::string model_path_;
    const tensorflow::GraphDef* gdef_;
    std::unique_ptr<tensorflow::SavedModel> saved_model_;
    std::vector<NodeInfo>;
    
    void load_model();
    std::string get_input_name(const std::string& name);
    std::vector<int64_t> get_shape_from_proto(const tensorflow::TensorShapeProto& shape_proto);
    std::string get_data_type(const tensorflow::NodeDef* node);
    void prune_graph(const std::vector<std::string>& output_defs,
                    std::unordered_map<std::string, const tensorflow::NodeDef*>& node_map,
                    std::vector<std::string>& pruned_nodes);
    void process_identity_nodes(std::vector<std::string>& pruned_nodes,
                               const std::unordered_map<std::string, const tensorflow::NodeDef*>& node_map,
                               std::unordered_map<std::string, std::string>& identity_redirect);

    void process_node(const tensorflow::NodeDef* node,
                     std::unordered_map<std::string, std::string>& identity_redirect,
                     NodeInfo& info);

    bool extract_tensor_from_attr(const tensorflow::NodeDef* node,
                                 const std::string& attr_name,
                                 std::vector<uint8_t>& tensor_values,
                                 std::vector<int64_t>& shape,
                                 std::string& dtype);

    std::vector<NodeInfo> topologicalSort(const std::vector<NodeInfo>& nodes);
};

#endif // TFSAVEDMODEL_PARSER_H
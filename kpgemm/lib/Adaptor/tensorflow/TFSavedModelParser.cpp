#include "Adaptor/tensorflow/TFSavedModelParser.h"
#include "tensorflow/core/grappler/op_types.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/env.h"
#include <algorithm>
#include <queue>

using namespace tensorflow;
using namespace tensorflow::grappler;
using namespace std;

TFSavedModelParser::TFSavedModelParser(const std::string& model_path)
    : model_path_(model_path), gdef_(nullptr) {
    load_model();
}

void TFSavedModelParser::load_model() {
    saved_model_ = std::make_unique<SavedModel>();
    const string pb_path = model_path_ + "/saved_model.pb";
    const string pbtxt_path = model_path_ + "/saved_model.pbtxt";

    if (Env::Default()->FileExists(pb_path).ok()) {
        TF_CHECK_OK(ReadBinaryProto(Env::Default(), pb_path, saved_model_.get()));
    } else if (Env::Default()->FileExists(pbtxt_path).ok()) {
        TF_CHECK_OK(ReadTextProto(Env::Default(), pbtxt_path, saved_model_.get()));
    } else {
        std::cout << "Saved model not exists: " << model_path_;
        return;
    }

    if (saved_model_->meta_graphs_size() == 0) {
        std::cout << "No meta graphs found in saved model\n";
        return;
    }

    MetaGraphDef* meta_graph = saved_model_->mutable_meta_graphs(0);
    gdef_ = &(meta_graph->graph_def());
}

std::string TFSavedModelParser::get_input_name(const std::string& name) {
    size_t colon_pos = name.find(':');
    std::string input_name = name;
    if (colon_pos != std::string::npos) {
        input_name = name.substr(0, colon_pos);
    }
    return input_name;
}

std::vector<int64_t> TFSavedModelParser::get_shape_from_proto(const TensorShapeProto& shape_proto) {
    std::vector<int64_t> shape;
    for (int i = 0; i < shape_proto.dim_size(); ++i) {
        int64_t dim = shape_proto.dim(i).size();
        shape.push_back(dim);
    }
    return shape;
}

std::string TFSavedModelParser::get_data_type(const NodeDef* node) {
    DataType tf_dtype = DT_INVALID;
    if (node->attr().find("dtype") != node->attr().end())
        tf_dtype = node->attr().at("dtype").type();
    else if (node->attr().find("T") != node->attr().end())
        tf_dtype = node->attr().at("T").type();
    else return "unknown";
    
    switch (tf_dtype) {
        case DT_FLOAT:     return "float32";
        case DT_DOUBLE:    return "float64";
        case DT_INT32:     return "int32";
        case DT_INT64:     return "int64";
        case DT_BOOL:      return "bool";
        case DT_STRING:    return "string";
        default:          return "unknown";
    }
}

std::vector<std::string> TFSavedModelParser::getModelOutputs() {
    std::vector<std::string> outputs;
    if (!saved_model_ || saved_model_->meta_graphs_size() == 0) {
        return outputs;
    }

    const MetaGraphDef* meta_graph = &saved_model_->meta_graphs(0);
    if (meta_graph->signature_def().count("serving_default")) {
        const auto& sdef = meta_graph->signature_def().at("serving_default");
        for (auto& output_elem : sdef.outputs()) {
            outputs.push_back(output_elem.second.name());
        }
    }
    return outputs;
}

bool TFSavedModelParser::get_tensor_value(const tensorflow::NodeDef* node,
                                         std::vector<uint8_t>& tensor_values,
                                         std::vector<int64_t>& shape,
                                         std::string& dtype) {
    if (!node) return false;
    if (node->op() == "Const") {
        return extract_tensor_from_attr(node, "value", tensor_values, shape, dtype);
    }
    if (node->op() == "VarHandleOp" || node->op() == "ReadVariableOp") {
        for (const auto& var_tensor : saved_model_->meta_graphs(0).collection_def()) {
            const auto& collection = var_tensor.second;
            if (collection.node_list().value_size() > 0) {
                for (int i = 0; i < collection.node_list().value_size(); ++i) {
                    const auto& node_name = collection.node_list().value(i);
                    if (node_name == node->name()) {
                        return extract_tensor_from_attr(node, "value", tensor_values, shape, dtype);
                    }
                }
            }
        }
    }
    
    return false;
}

bool TFSavedModelParser::extract_tensor_from_attr(const tensorflow::NodeDef* node,
                                                const std::string& attr_name,
                                                std::vector<uint8_t>& tensor_values,
                                                std::vector<int64_t>& shape,
                                                std::string& dtype) {
    auto attr_it = node->attr().find(attr_name);
    if (attr_it == node->attr().end()) {
        return false;
    }
    
    const tensorflow::TensorProto& tensor_proto = attr_it->second.tensor();
    dtype = get_data_type(node);

    shape.clear();
    for (int i = 0; i < tensor_proto.tensor_shape().dim_size(); ++i) {
        shape.push_back(tensor_proto.tensor_shape().dim(i).size());
    }
    
    tensor_values.clear();
    if (tensor_proto.tensor_content().size() > 0) {

        const char* data = tensor_proto.tensor_content().data();
        size_t size = tensor_proto.tensor_content().size();
        tensor_values.insert(tensor_values.end(), data, data + size);
    } else {
        if (dtype == "float32") {
            for (int i = 0; i < tensor_proto.float_val_size(); ++i) {
                const float val = tensor_proto.float_val(i);
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&val);
                tensor_values.insert(tensor_values.end(), bytes, bytes + sizeof(float));
            }
        } else if (dtype == "int32") {
            for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
                const int32_t val = tensor_proto.int_val(i);
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&val);
                tensor_values.insert(tensor_values.end(), bytes, bytes + sizeof(int32_t));
            }
        } else if (dtype == "int64") {
            for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
                const int64_t val = tensor_proto.int_val(i);
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&val);
                tensor_values.insert(tensor_values.end(), bytes, bytes + sizeof(int64_t));
            }
        } else if (dtype == "bool") {
            for (int i = 0; i < tensor_proto.bool_val_size(); ++i) {
                const uint8_t val = static_cast<uint8_t>(tensor_proto.bool_val(i));
                tensor_values.push_back(val);
            }
        } else if (dtype == "string") {
            for (int i = 0; i < tensor_proto.string_val_size(); ++i) {
                const std::string& str = tensor_proto.string_val(i);
                const char* data = str.data();
                size_t size = str.size();
                tensor_values.insert(tensor_values.end(), data, data + size);
            }
        }
    }
    
    return !tensor_values.empty();
}

std::vector<std::string> TFSavedModelParser::getModelInputs() {
    std::vector<std::string> inputs;
    if (!gdef_) return inputs;
    
    for (const auto& node : gdef_->node()) {
        if (IsPlaceholder(node) || IsVariable(node)) {
            inputs.push_back(node.name());
        }
    }
    return inputs;
}

void TFSavedModelParser::prune_graph(
    const std::vector<std::string>& output_defs,
    std::unordered_map<std::string, const NodeDef*>& node_map, 
    std::vector<std::string>& pruned_nodes) {
    
    std::set<std::string> visited;
    std::queue<std::string> worklist;
    
    for (auto output_name : output_defs) {
        printf("output_name : %s\n", output_name.c_str());
        std::string name = get_input_name(output_name);
        if (node_map.count(name)) {
            visited.insert(name);
            worklist.push(name);
        }
    }

    while (!worklist.empty()) {
        std::string cur_name = worklist.front();
        worklist.pop();
        const NodeDef* node = node_map.at(cur_name);
        
        if (IsVariable(*node)) continue;
        
        for (const std::string& input_tensor : node->input()) {
            std::string input_name = get_input_name(input_tensor);
            printf("input name :  %s\n", input_name.c_str());
            if (input_name.empty() || input_name[0] == '^') continue;

            if (!visited.count(input_name)) {
                visited.insert(input_name);
                pruned_nodes.push_back(input_name);
                worklist.push(input_name);
            }
        }
    }
}

void TFSavedModelParser::process_identity_nodes(
    std::vector<std::string>& pruned_nodes, 
    const std::unordered_map<std::string, const NodeDef*>& node_map,
    std::unordered_map<std::string, std::string>& identity_redirect) {
    
    for (auto name : pruned_nodes) {
        name = get_input_name(name);
        printf("pruned name : %s\n" , name.c_str());
        if (node_map.count(name)) {
            const NodeDef* node = node_map.at(name);
            if (IsIdentity(*node)) {
                if (!node->input().empty()) {
                    std::string input_name = get_input_name(node->input(0));
                    if (!input_name.empty() && input_name[0] != '^') {
                        identity_redirect[name] = input_name;
                    }
                }
            }
        }
    }
    
    //  Identity  (A->identity->identity->B to A->B)
    bool changed;
    do {
        changed = false;
        for (auto& kv : identity_redirect) {
            std::string& target = kv.second;
            while (identity_redirect.count(target)) {
                std::string next = identity_redirect[target];
                target = next;
                changed = true;
            }
        }
    } while (changed);
}

void TFSavedModelParser::process_node(const NodeDef* node,
                                      std::unordered_map<std::string, std::string>& identity_redirect,
                                      NodeInfo& info) {
    for (int i = 0; i < node->input_size(); ++i) {
        std::string input_name = get_input_name(node->input(i));
        while (identity_redirect.count(input_name)) {
            input_name = identity_redirect[input_name];
        }
        info.inputs.push_back(input_name);
    }

    auto attr_it = node->attr().find("_output_shapes");
    if (attr_it != node->attr().end()) {
        const auto& shapes = attr_it->second.list().shape();
        for (int j = 0; j < shapes.size(); ++j) {
            std::vector<int64_t> shape = get_shape_from_proto(shapes[j]);
            std::string dtype = get_data_type(node);
            info.addOutput(j, node->name(), shape, dtype);
        }
    }
    
    //  variable  const  tensor 
    if (IsVariable(*node) || node->op() == "Const") {
        std::vector<uint8_t> tensor_values;
        std::vector<int64_t> shape;
        std::string dtype;
        
        if (get_tensor_value(node, tensor_values, shape, dtype)) {
            //  tensor  base64  raw_data
            std::string base64_data;
            base64_data.resize(tensor_values.size() * 4 / 3 + 4, '\0');
            
            const static char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            size_t i = 0;
            size_t j = 0;
            for (; i + 2 < tensor_values.size(); i += 3) {
                uint32_t value = (tensor_values[i] << 16) | (tensor_values[i+1] << 8) | tensor_values[i+2];
                base64_data[j++] = chars[(value >> 18) & 0x3F];
                base64_data[j++] = chars[(value >> 12) & 0x3F];
                base64_data[j++] = chars[(value >> 6) & 0x3F];
                base64_data[j++] = chars[value & 0x3F];
            }
            
            if (i < tensor_values.size()) {
                uint32_t value = tensor_values[i] << 16;
                if (i + 1 < tensor_values.size()) {
                    value |= tensor_values[i+1] << 8;
                }
                base64_data[j++] = chars[(value >> 18) & 0x3F];
                base64_data[j++] = chars[(value >> 12) & 0x3F];
                if (i + 1 < tensor_values.size()) {
                    base64_data[j++] = chars[(value >> 6) & 0x3F];
                } else {
                    base64_data[j++] = '=';
                }
                base64_data[j++] = '=';
            }
            
            base64_data.resize(j);
            info.raw_data = base64_data;
        }
    }
}

std::vector<NodeInfo> TFSavedModelParser::parse(const std::vector<std::string>& output_defs) {
    std::vector<NodeInfo> nodes;
    
    if (!gdef_) {
        return nodes;
    }

    std::unordered_map<std::string, const NodeDef*> node_map;
    for (const auto& node : gdef_->node()) {
        node_map[node.name()] = &node;
    }

    std::vector<std::string> pruned_nodes;
    prune_graph(output_defs, node_map, pruned_nodes);
    printf("num of pruned nodes: %d\n", pruned_nodes.size());
    //  Identity 
    std::unordered_map<std::string, std::string> identity_redirect;
    process_identity_nodes(pruned_nodes, node_map, identity_redirect);
    printf("num of pruned nodes: %d\n", pruned_nodes.size());
    std::vector<std::string> processed_output_defs = output_defs;
    process_identity_nodes(processed_output_defs, node_map, identity_redirect);
    for (auto& output_name : processed_output_defs) {
        output_name = get_input_name(output_name);
        if (!node_map.count(output_name)) continue;
        const NodeDef* node = node_map.at(output_name);
        std::string name = output_name;
        while (identity_redirect.count(name)) {
            name = identity_redirect[name];
        }
        NodeInfo info;
        info.name = name;
        info.op_type = (node->op() == "AddV2") ? "Add" : node->op();
        info.isOutputNode = true;
        
        process_node(node, identity_redirect, info);
        nodes.push_back(std::move(info));
    }

    for (const auto& node_name : pruned_nodes) {
        if (!node_map.count(node_name)) continue;

        const NodeDef* node = node_map.at(node_name);
        std::string name = node_name;
        while (identity_redirect.count(name)) {
            name = identity_redirect[name];
        }

        NodeInfo info;
        info.name = name;
        info.op_type = (node->op() == "AddV2") ? "Add" : node->op();

        if (IsIdentity(*node)) continue;
        if (IsPlaceholder(*node) || IsVariable(*node)) {
            info.isInputNode = true;
        }

        process_node(node, identity_redirect, info);
        nodes.push_back(std::move(info));
    }
    
    return nodes;
}

json TFSavedModelParser::toJson(const std::vector<NodeInfo>& nodes) {
    json result;
    std::vector<NodeInfo> sorted_nodes = topologicalSort(nodes);

    result["name"] = "main";

    json inputs_array = json::array();
    json outputs_array = json::array();
    json nodes_array = json::array();
    for (const auto& node : sorted_nodes) {
        if (node.isInputNode) {
            const auto& output = node.outputs[0];
            json input_json = {
                {"name", output.name},
                {"shape", output.shape},
                {"dtype",  output.dtype}
            };
            inputs_array.push_back(input_json);
        } else if (node.isOutputNode) {
            json output_json = {
                {"name", node.name},
                {"shape", node.outputs[0].shape},
                {"dtype", node.outputs[0].dtype}
            };
            outputs_array.push_back(output_json);
        } else {
            json node_json;
            node_json["name"] = node.name;
            node_json["type"] = node.op_type;
            node_json["attributes"] = json::array();
            node_json["inputs"] = node.inputs;

            json node_outputs_array = json::array();
            for (const auto& output : node.outputs) {
                json output_json = {
                    {"name", output.name},
                    {"dtype", output.dtype},
                    {"shape", output.shape}
                };
                node_outputs_array.push_back(output_json);
            }
            node_json["outputs"] = node_outputs_array;
            nodes_array.push_back(node_json);
        }
    }
    result["inputs"] = inputs_array;
    result["outputs"] = outputs_array;
    result["nodes"] = nodes_array;
    return result;
}

void TFSavedModelParser::print_tensor_data() {
    if (!gdef_) {
        std::cout << "Graph is not loaded\n";
        return;
    }
    
    std::cout << "=== Tensor Data for Variables and Const Nodes ===\n";
    
    for (const auto& node : gdef_->node()) {
        if (IsVariable(node) || node.op() == "Const") {
            std::vector<uint8_t> tensor_values;
            std::vector<int64_t> shape;
            std::string dtype;
            
            if (get_tensor_value(&node, tensor_values, shape, dtype)) {
                std::cout << "Node: " << node.name() << "\n";
                std::cout << "  Op: " << node.op() << "\n";
                std::cout << "  Shape: [";
                for (size_t i = 0; i < shape.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << shape[i];
                }
                std::cout << "]\n";
                std::cout << "  DataType: " << dtype << "\n";
                std::cout << "  Data Size: " << tensor_values.size() << " bytes\n";
                if (!tensor_values.empty()) {
                    std::cout << "  Sample Data (first 8 bytes): ";
                    for (int i = 0; i < std::min(8, (int)tensor_values.size()); ++i) {
                        std::cout << std::hex << (int)tensor_values[i] << " ";
                    }
                    std::cout << std::dec << "\n";
                }
                std::cout << "\n";
            } else {
                std::cout << "Node: " << node.name() << " - No tensor data found\n\n";
            }
        }
    }
}
    
std::vector<NodeInfo> TFSavedModelParser::topologicalSort(const std::vector<NodeInfo>& nodes) {
        std::unordered_map<std::string, size_t> node_to_index;
        for (size_t i = 0; i < nodes.size(); ++i) {
            node_to_index[nodes[i].name] = i;
        }

        std::vector<int> in_degree(nodes.size(), 0);

        std::vector<std::vector<size_t>> adj_list(nodes.size());

        for (size_t i = 0; i < nodes.size(); ++i) {
            const NodeInfo& node = nodes[i];
            for (const std::string& input_name : node.inputs) {
                if (node_to_index.find(input_name) != node_to_index.end()) {
                    size_t input_index = node_to_index[input_name];
                    adj_list[input_index].push_back(i);
                    in_degree[i]++;
                }
            }
        }

        std::queue<size_t> q;
        std::vector<NodeInfo> sorted_nodes;

        for (size_t i = 0; i < nodes.size(); ++i) {
            if (in_degree[i] == 0) {
                q.push(i);
            }
        }

        while (!q.empty()) {
            size_t current = q.front();
            NodeInfo node = nodes[current];
            printf("sorted node : %s\n", node.name.c_str());
            q.pop();
            sorted_nodes.push_back(nodes[current]);

            for (size_t neighbor : adj_list[current]) {
                in_degree[neighbor]--;
                if (in_degree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }

        if (sorted_nodes.size() != nodes.size()) {
            std::cerr << "Warning: Cycle detected in graph, original node order used" << std::endl;
            return nodes;
        } 
        return sorted_nodes;
}

#include "standalone_pb_parser.h"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace tensorflow;
using annc::NodeInfo;
using annc::OutputInfo;

// 构造函数
StandalonePbParser::StandalonePbParser(const std::string& model_path) 
    : model_path_(model_path) {}

bool StandalonePbParser::loadModel() {
    saved_model_ = std::make_unique<SavedModel>();
    std::string pb_path = model_path_ + "/saved_model.pb";
    
    std::ifstream pb_file(pb_path, std::ios::binary);
    if (!pb_file) {
        std::cerr << "Error: Cannot open SavedModel at " << pb_path << std::endl;
        return false;
    }
    if (!saved_model_->ParseFromIstream(&pb_file)) {
        std::cerr << "Error: Failed to parse SavedModel at " << pb_path << std::endl;
        return false;
    }
    if (saved_model_->meta_graphs_size() == 0) {
        std::cerr << "Error: No meta graphs found in saved model\n";
        return false;
    }
    gdef_ = &(saved_model_->meta_graphs(0).graph_def());
    return true;
}

bool StandalonePbParser::parse() {
    if (!loadModel()) return false;
    
    std::unordered_map<std::string, const NodeDef*> node_map;
    for (const auto& node : gdef_->node()) {
        node_map[node.name()] = &node;
    }
    
    std::vector<std::string> output_defs;
    const auto& meta_graph = saved_model_->meta_graphs(0);
    if (meta_graph.signature_def().count("serving_default")) {
        const auto& sdef = meta_graph.signature_def().at("serving_default");
        for (const auto& output_elem : sdef.outputs()) {
            output_defs.push_back(output_elem.second.name());
        }
    }
    
    if (output_defs.empty()) {
        for (auto it = gdef_->node().rbegin(); it != gdef_->node().rend(); ++it) {
            if (!isPlaceholder(*it) && it->op() != "Const") {
                output_defs.push_back(it->name());
                break;
            }
        }
    }

    std::vector<std::string> pruned_nodes;
    pruneGraph(output_defs, node_map, pruned_nodes);

    std::unordered_map<std::string, std::string> identity_redirect;
    processIdentityNodes(pruned_nodes, node_map, identity_redirect);
    processIdentityNodes(output_defs, node_map, identity_redirect);

    // 处理输出节点
    for (auto& output_name : output_defs) {
        output_name = getInputName(output_name);
        if (!node_map.count(output_name)) {
            size_t colon_pos = output_name.find(':');
            if (colon_pos != std::string::npos) {
                std::string base_name = output_name.substr(0, colon_pos);
                int out_idx = std::stoi(output_name.substr(colon_pos + 1));
                if (node_map.count(base_name)) {
                    NodeInfo base_info;
                    base_info.name = base_name;
                    base_info.op_type = (node_map.at(base_name)->op() == "AddV2") ? "Add" : node_map.at(base_name)->op();
                    processNode(node_map.at(base_name), identity_redirect, base_info);
                    if (out_idx >= 0 && out_idx < static_cast<int>(base_info.outputs.size())) {
                        NodeInfo alias_out;
                        alias_out.name = output_name;
                        alias_out.op_type = "Identity";
                        alias_out.isOutputNode = true;
                        alias_out.inputs.push_back(output_name);
                        OutputInfo out = base_info.outputs[static_cast<size_t>(out_idx)];
                        out.name = alias_out.name;
                        alias_out.outputs.push_back(std::move(out));
                        nodes_.push_back(std::move(alias_out));
                    }
                }
            }
            continue;
        }
        
        const NodeDef* node = node_map.at(output_name);
        std::string name = output_name;
        while (identity_redirect.count(name)) {
            name = identity_redirect[name];
        }

        bool redirectedToIndexedTensor = (name.find(':') != std::string::npos);
        const NodeDef* target_node = node;
        std::string final_name = output_name;
        if (!redirectedToIndexedTensor && name != output_name && node_map.count(name)) {
            target_node = node_map.at(name);
            final_name = name;
        }

        NodeInfo info;
        info.name = final_name;
        info.op_type = (target_node->op() == "AddV2") ? "Add" : target_node->op();
        info.isOutputNode = true;
        processNode(target_node, identity_redirect, info);
        nodes_.push_back(std::move(info));
    }

    // 处理其他节点
    std::set<std::string> processed_names;
    for (const auto& node : nodes_) {
        processed_names.insert(node.name);
    }
    
    for (const auto& node_name : pruned_nodes) {
        if (!node_map.count(node_name)) continue;
        const NodeDef* node = node_map.at(node_name);
        std::string name = node_name;
        while (identity_redirect.count(name)) {
            name = identity_redirect[name];
        }
        if (isIdentity(*node) || processed_names.count(name)) continue;

        NodeInfo info;
        info.name = name;
        info.op_type = (node->op() == "AddV2") ? "Add" : node->op();
        if (isPlaceholder(*node) || isVariable(*node)) {
            info.isInputNode = true;
        }
        processNode(node, identity_redirect, info);
        nodes_.push_back(std::move(info));
    }

    nodes_ = topologicalSort(nodes_);
    valid_ = true;
    return true;
}

void StandalonePbParser::pruneGraph(
    const std::vector<std::string>& output_defs,
    std::unordered_map<std::string, const NodeDef*>& node_map,
    std::vector<std::string>& pruned_nodes) {
    std::set<std::string> visited;
    std::queue<std::string> worklist;
    for (const auto& output_name : output_defs) {
        std::string name = getInputName(output_name);
        if (!node_map.count(name)) {
            size_t colon_pos = name.find(':');
            if (colon_pos != std::string::npos) {
                name = name.substr(0, colon_pos);
            }
        }
        if (node_map.count(name) && !visited.count(name)) {
            visited.insert(name);
            pruned_nodes.push_back(name);
            worklist.push(name);
        }
    }
    while (!worklist.empty()) {
        std::string cur_name = worklist.front();
        worklist.pop();
        const NodeDef* node = node_map.at(cur_name);
        if (isVariable(*node)) continue;
        for (const std::string& input_tensor : node->input()) {
            std::string input_name = getInputName(input_tensor);
            if (input_name.empty() || input_name[0] == '^') continue;
            size_t colon_pos = input_name.find(':');
            if (colon_pos != std::string::npos) {
                input_name = input_name.substr(0, colon_pos);
            }
            if (!visited.count(input_name) && node_map.count(input_name)) {
                visited.insert(input_name);
                pruned_nodes.push_back(input_name);
                worklist.push(input_name);
            }
        }
    }
}

void StandalonePbParser::processIdentityNodes(
    std::vector<std::string>& pruned_nodes,
    const std::unordered_map<std::string, const NodeDef*>& node_map,
    std::unordered_map<std::string, std::string>& identity_redirect) {
    
    for (const auto& node_name : pruned_nodes) {
        std::string name = getInputName(node_name);
        if (!node_map.count(name)) continue;
        const NodeDef* node = node_map.at(name);
        if (isIdentity(*node) && !node->input().empty()) {
            std::string input_name = getInputName(node->input(0));
            if (!input_name.empty() && input_name[0] != '^') {
                identity_redirect[name] = input_name;
            }
        }
    }

    bool changed;
    do {
        changed = false;
        for (auto& kv : identity_redirect) {
            std::string& target = kv.second;
            while (identity_redirect.count(target)) {
                target = identity_redirect[target];
                changed = true;
            }
        }
    } while (changed);
}

void StandalonePbParser::processNode(
    const NodeDef* node,
    const std::unordered_map<std::string, std::string>& identity_redirect,
    annc::NodeInfo& info) {
    info.attrs.clear();
    for (const auto& kv : node->attr()) {
        const std::string& attrName = kv.first;
        const auto& attrVal = kv.second;
        switch (attrVal.value_case()) {
            case tensorflow::AttrValue::kI:
                info.attrs[attrName] = static_cast<int64_t>(attrVal.i());
                break;
            case tensorflow::AttrValue::kF:
                info.attrs[attrName] = static_cast<double>(attrVal.f());
                break;
            case tensorflow::AttrValue::kB:
                info.attrs[attrName] = static_cast<bool>(attrVal.b());
                break;
            case tensorflow::AttrValue::kS:
                info.attrs[attrName] = attrVal.s();
                break;
            case tensorflow::AttrValue::kList: {
                const auto& lv = attrVal.list();
                if (lv.i_size() > 0) {
                    std::vector<int64_t> v;
                    v.reserve(static_cast<size_t>(lv.i_size()));
                    for (auto x : lv.i()) v.push_back(static_cast<int64_t>(x));
                    info.attrs[attrName] = std::move(v);
                } else if (lv.f_size() > 0) {
                    std::vector<double> v;
                    v.reserve(static_cast<size_t>(lv.f_size()));
                    for (auto x : lv.f()) v.push_back(static_cast<double>(x));
                    info.attrs[attrName] = std::move(v);
                } else if (lv.b_size() > 0) {
                    std::vector<bool> v;
                    v.reserve(static_cast<size_t>(lv.b_size()));
                    for (int i = 0; i < lv.b_size(); ++i) v.push_back(lv.b(i));
                    info.attrs[attrName] = std::move(v);
                } else if (lv.s_size() > 0) {
                    std::vector<std::string> v;
                    v.reserve(static_cast<size_t>(lv.s_size()));
                    for (int i = 0; i < lv.s_size(); ++i) v.push_back(lv.s(i));
                    info.attrs[attrName] = std::move(v);
                }
                break;
            }
            default:
                break;
        }
    }
    for (int i = 0; i < node->input_size(); ++i) {
        std::string input_name = getInputName(node->input(i));
        while (identity_redirect.count(input_name)) {
            input_name = identity_redirect.at(input_name);
        }
        info.inputs.push_back(input_name);
    }
    auto attr_it = node->attr().find("_output_shapes");
    if (attr_it != node->attr().end()) {
        const auto& shapes = attr_it->second.list().shape();
        const int outputCount = shapes.size();
        for (int j = 0; j < outputCount; ++j) {
            std::vector<int64_t> shape = shapeFromProto(shapes[j]);
            DataType tf_dtype = DT_FLOAT;
            auto getTypeAttr = [&](const char *key, DataType &out) -> bool {
                auto it = node->attr().find(key);
                if (it == node->attr().end()) return false;
                out = it->second.type();
                return true;
            };
            auto getTypeListAttr = [&](const char *key, int outIdx, DataType &out) -> bool {
                auto it = node->attr().find(key);
                if (it == node->attr().end()) return false;
                const auto &typeList = it->second.list().type();
                if (outIdx < 0 || outIdx >= typeList.size()) return false;
                out = static_cast<DataType>(typeList.Get(outIdx));
                return true;
            };
            if (isPlaceholder(*node)) {
                (void)getTypeAttr("dtype", tf_dtype);
            } else if (getTypeAttr("DstT", tf_dtype) ||
                       getTypeAttr("out_type", tf_dtype) ||
                       getTypeAttr("output_type", tf_dtype) ||
                       getTypeListAttr("Tout", j, tf_dtype) ||
                       getTypeAttr("Tout", tf_dtype) ||
                       getTypeAttr("T", tf_dtype)) {
            }
            std::string dtype = getDataTypeStr(tf_dtype);     
            OutputInfo output;
            output.name = (outputCount > 1 && j > 0) ? (info.name + ":" + std::to_string(j)) : info.name;
            output.shape = shape;
            output.dtype = dtype;
            info.outputs.push_back(output);
        }
    } else {
        OutputInfo output;
        output.name = info.name;
        output.dtype = "float32";
        info.outputs.push_back(output);
    }
    if (node->op() == "SparseFillEmptyRows" && info.outputs.size() >= 4) {
        std::string valuesDtype = "float32";
        if (node->attr().count("T")) {
            valuesDtype = getDataTypeStr(node->attr().at("T").type());
        }
        info.outputs[0].dtype = "int64";
        info.outputs[1].dtype = valuesDtype;
        info.outputs[2].dtype = "bool";
        info.outputs[3].dtype = "int64";
    }
    if (node->op() == "Where" && info.outputs.size() >= 1) {
        info.outputs[0].dtype = "int64";
    }
    if ((node->op() == "TopK" || node->op() == "TopKV2") && info.outputs.size() >= 2) {
        std::string valuesDtype = "float32";
        if (node->attr().count("T")) {
            valuesDtype = getDataTypeStr(node->attr().at("T").type());
        }
        std::string indicesDtype = "int32";
        auto it = node->attr().find("index_type");
        if (it != node->attr().end()) {
            indicesDtype = getDataTypeStr(it->second.type());
        }
        info.outputs[0].dtype = valuesDtype;
        info.outputs[1].dtype = indicesDtype;
    }
    if (node->op() == "Const") {
        std::vector<uint8_t> tensor_values;
        std::vector<int64_t> shape;
        std::string dtype;
        if (getTensorValue(node, tensor_values, shape, dtype)) {
            info.raw_data = base64Encode(tensor_values);
            if (!info.outputs.empty()) {
                info.outputs[0].shape = shape;
                info.outputs[0].dtype = dtype;
            }
        }
    }
}

// 张量数据处理
bool StandalonePbParser::getTensorValue(
    const NodeDef* node,
    std::vector<uint8_t>& tensor_values,
    std::vector<int64_t>& shape,
    std::string& dtype) {
    if (!node || node->op() != "Const") return false;
    return extractTensorFromAttr(node, "value", tensor_values, shape, dtype);
}

bool StandalonePbParser::extractTensorFromAttr(
    const NodeDef* node,
    const std::string& attr_name,
    std::vector<uint8_t>& tensor_values,
    std::vector<int64_t>& shape,
    std::string& dtype) {
    auto attr_it = node->attr().find(attr_name);
    if (attr_it == node->attr().end()) return false;
    const TensorProto& tensor_proto = attr_it->second.tensor();
    dtype = getDataTypeStr(tensor_proto.dtype());
    shape.clear();
    for (int i = 0; i < tensor_proto.tensor_shape().dim_size(); ++i) {
        shape.push_back(tensor_proto.tensor_shape().dim(i).size());
    }
    auto appendBytes = [&](const void* p, size_t n) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(p);
        tensor_values.insert(tensor_values.end(), bytes, bytes + n);
    };
    auto elementCount = [&]() -> size_t {
        if (shape.empty()) return 1;
        size_t n = 1;
        for (int64_t d : shape) {
            if (d < 0) return 0;
            n *= static_cast<size_t>(d);
        }
        return n;
    };
    auto appendZeroFilled = [&](size_t elemSize) {
        size_t n = elementCount();
        if (n == 0) return;
        tensor_values.resize(n * elemSize, 0);
    };
    tensor_values.clear();
    if (tensor_proto.tensor_content().size() > 0) {
        const char* data = tensor_proto.tensor_content().data();
        size_t size = tensor_proto.tensor_content().size();
        tensor_values.insert(tensor_values.end(), data, data + size);
    } else {
        if (dtype == "float32") {
            for (int i = 0; i < tensor_proto.float_val_size(); ++i) {
                const float val = tensor_proto.float_val(i);
                appendBytes(&val, sizeof(float));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(float));
        } else if (dtype == "float64") {
            for (int i = 0; i < tensor_proto.double_val_size(); ++i) {
                const double val = tensor_proto.double_val(i);
                appendBytes(&val, sizeof(double));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(double));
        } else if (dtype == "float16") {
            for (int i = 0; i < tensor_proto.half_val_size(); ++i) {
                const uint16_t val = static_cast<uint16_t>(tensor_proto.half_val(i));
                appendBytes(&val, sizeof(uint16_t));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(uint16_t));
        } else if (dtype == "int32") {
            for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
                const int32_t val = tensor_proto.int_val(i);
                appendBytes(&val, sizeof(int32_t));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(int32_t));
        } else if (dtype == "int64") {
            for (int i = 0; i < tensor_proto.int64_val_size(); ++i) {
                const int64_t val = tensor_proto.int64_val(i);
                appendBytes(&val, sizeof(int64_t));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(int64_t));
        } else if (dtype == "int16") {
            for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
                const int16_t val = static_cast<int16_t>(tensor_proto.int_val(i));
                appendBytes(&val, sizeof(int16_t));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(int16_t));
        } else if (dtype == "int8") {
            for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
                const int8_t val = static_cast<int8_t>(tensor_proto.int_val(i));
                appendBytes(&val, sizeof(int8_t));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(int8_t));
        } else if (dtype == "uint8") {
            for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
                const uint8_t val = static_cast<uint8_t>(tensor_proto.int_val(i));
                appendBytes(&val, sizeof(uint8_t));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(uint8_t));
        } else if (dtype == "bool") {
            for (int i = 0; i < tensor_proto.bool_val_size(); ++i) {
                const uint8_t val = tensor_proto.bool_val(i) ? 1 : 0;
                appendBytes(&val, sizeof(uint8_t));
            }
            if (tensor_values.empty()) appendZeroFilled(sizeof(uint8_t));
        }
    }
    return true;
}

std::vector<annc::NodeInfo> StandalonePbParser::topologicalSort(const std::vector<annc::NodeInfo>& nodes) const {
    std::unordered_map<std::string, size_t> node_to_index;
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_to_index[nodes[i].name] = i;
    }
    std::vector<int> in_degree(nodes.size(), 0);
    std::vector<std::vector<size_t>> adj_list(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        const annc::NodeInfo& node = nodes[i];
        for (const std::string& input_name : node.inputs) {
            auto it = node_to_index.find(input_name);
            if (it == node_to_index.end()) {
                size_t colonPos = input_name.find(':');
                if (colonPos != std::string::npos) {
                    std::string base = input_name.substr(0, colonPos);
                    it = node_to_index.find(base);
                }
            }
            if (it != node_to_index.end()) {
                size_t input_index = it->second;
                adj_list[input_index].push_back(i);
                in_degree[i]++;
            }
        }
    }
    std::queue<size_t> q;
    std::vector<annc::NodeInfo> sorted_nodes;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (in_degree[i] == 0) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        size_t current = q.front();
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
        std::cerr << "Warning: Cycle detected in graph" << std::endl;
        return nodes;
    }
    
    return sorted_nodes;
}

std::string StandalonePbParser::base64Encode(const std::vector<uint8_t>& data) {
    static const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::string result;
    result.reserve((data.size() + 2) / 3 * 4);
    
    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        uint32_t value = (data[i] << 16) | (data[i+1] << 8) | data[i+2];
        result.push_back(chars[(value >> 18) & 0x3F]);
        result.push_back(chars[(value >> 12) & 0x3F]);
        result.push_back(chars[(value >> 6) & 0x3F]);
        result.push_back(chars[value & 0x3F]);
    }

    if (i < data.size()) {
        uint32_t value = data[i] << 16;
        if (i + 1 < data.size()) {
            value |= data[i+1] << 8;
        }
        result.push_back(chars[(value >> 18) & 0x3F]);
        result.push_back(chars[(value >> 12) & 0x3F]);
        if (i + 1 < data.size()) {
            result.push_back(chars[(value >> 6) & 0x3F]);
        } else {
            result.push_back('=');
        }
        result.push_back('=');
    }

    return result;
}

std::string StandalonePbParser::getInputName(const std::string& name) const {
    size_t colon_pos = name.find(':');
    if (colon_pos != std::string::npos) {
        std::string suffix = name.substr(colon_pos + 1);
        if (suffix == "0") {
            return name.substr(0, colon_pos);
        }
        return name;
    }
    return name;
}

std::string StandalonePbParser::getDataTypeStr(int dtype) const {
    switch (dtype) {
        case DT_FLOAT:     return "float32";
        case DT_DOUBLE:    return "float64";
        case DT_INT32:     return "int32";
        case DT_INT64:     return "int64";
        case DT_INT16:     return "int16";
        case DT_INT8:      return "int8";
        case DT_UINT8:     return "uint8";
        case DT_HALF:      return "float16";
        case DT_BOOL:      return "bool";
        case DT_STRING:    return "string";
        case DT_COMPLEX64: return "complex64";
        case DT_COMPLEX128:return "complex128";
        default:           return "float32";
    }
}

std::vector<int64_t> StandalonePbParser::shapeFromProto(const TensorShapeProto& proto) {
    std::vector<int64_t> shape;
    for (int i = 0; i < proto.dim_size(); ++i) {
        int64_t dim = proto.dim(i).size();
        shape.push_back(dim == -1 ? -1 : dim);
    }
    return shape;
}

bool StandalonePbParser::isPlaceholder(const NodeDef& node) {
    return node.op() == "Placeholder" || node.op() == "PlaceholderV2";
}

bool StandalonePbParser::isVariable(const NodeDef& node) {
    return node.op() == "Variable" || node.op() == "VariableV2" || 
           node.op() == "VarHandleOp" || node.op() == "ReadVariableOp";
}

bool StandalonePbParser::isIdentity(const NodeDef& node) {
    return node.op() == "Identity" || node.op() == "IdentityN";
}

#ifndef STANDALONE_PB_PARSER_H
#define STANDALONE_PB_PARSER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <set>

// 包含 protoc 生成的核心头文件
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

// 引入Builder中的NodeInfo定义
#include "Builder/MLIROpBuilder.h"

class StandalonePbParser {
public:
    explicit StandalonePbParser(const std::string& model_path);
    bool parse();

    bool isValid() const { return valid_; }
    const std::vector<annc::NodeInfo>& getNodes() const { return nodes_; }

private:
    std::string model_path_;
    bool valid_ = false;
    std::vector<annc::NodeInfo> nodes_;
    
    // SavedModel 数据
    std::unique_ptr<tensorflow::SavedModel> saved_model_;
    std::unique_ptr<tensorflow::GraphDef> graph_def_;
    const tensorflow::GraphDef* gdef_ = nullptr;

    // 内部方法
    bool loadModel();
    std::string getDataTypeStr(int dtype) const;
    static std::vector<int64_t> shapeFromProto(const tensorflow::TensorShapeProto& proto);
    std::string getInputName(const std::string& name) const;
    
    // 图处理方法
    void pruneGraph(const std::vector<std::string>& output_defs,
                    std::unordered_map<std::string, const tensorflow::NodeDef*>& node_map,
                    std::vector<std::string>& pruned_nodes);
    void processIdentityNodes(std::vector<std::string>& pruned_nodes,
                              const std::unordered_map<std::string, const tensorflow::NodeDef*>& node_map,
                              std::unordered_map<std::string, std::string>& identity_redirect);
    void processNode(const tensorflow::NodeDef* node,
                     const std::unordered_map<std::string, std::string>& identity_redirect,
                     annc::NodeInfo& info);
    bool isAtirConvertibleNode(const annc::NodeInfo& node) const;
    void filterConvertibleNodes();
    void inferMvpMatMulAddReluShapes();
    
    // 张量数据处理
    bool getTensorValue(const tensorflow::NodeDef* node,
                        std::vector<uint8_t>& tensor_values,
                        std::vector<int64_t>& shape,
                        std::string& dtype);
    bool extractTensorFromAttr(const tensorflow::NodeDef* node,
                               const std::string& attr_name,
                               std::vector<uint8_t>& tensor_values,
                               std::vector<int64_t>& shape,
                               std::string& dtype);
    
    // 工具方法
    std::vector<annc::NodeInfo> topologicalSort(const std::vector<annc::NodeInfo>& nodes) const;
    static std::string base64Encode(const std::vector<uint8_t>& data);
    
    // 判断节点类型
    static bool isPlaceholder(const tensorflow::NodeDef& node);
    static bool isVariable(const tensorflow::NodeDef& node);
    static bool isIdentity(const tensorflow::NodeDef& node);
};

#endif
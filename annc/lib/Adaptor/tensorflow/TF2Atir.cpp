#include "Adaptor/tensorflow/TF2Atir.h"


TFGraphDefAdaptor::TFGraphDefAdaptor(const std::string& model_path)
    : parser_(std::make_unique<TFSavedModelParser>(model_path)) {
}

std::vector<NodeInfo> TFGraphDefAdaptor::parse(std::vector<std::string>& output_defs) {
    if (!parser_->isValid()) {
        return std::vector<NodeInfo>();
    }
    
    return parser_->parse(output_defs);
}

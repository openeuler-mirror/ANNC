#include "Adaptor/tensorflow/TFSavedModelParser.h"
#include <vector>

class TFGraphDefAdaptor{
public:
    explicit TFGraphDefAdaptor(const std::string& model_path);
    std::vector<NodeInfo> parse(std::vector<std::string>& output_defs);

private:
   std::unique_ptr<TFSavedModelParser> parser_;
};


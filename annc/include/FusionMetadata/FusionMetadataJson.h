#ifndef ANNC_FUSION_METADATA_FUSION_METADATA_JSON_H
#define ANNC_FUSION_METADATA_FUSION_METADATA_JSON_H

#include <string>
#include <vector>

#include "FusionMetadata/FusionMetadata.h"
#include "llvm/Support/Error.h"
#include "nlohmann/json.hpp"

namespace annc::fusion {

nlohmann::json fusionInfoToJson(const FusionInfo &info);
nlohmann::json fusionInfosToJson(const std::vector<FusionInfo> &infos);

llvm::Expected<FusionInfo> fusionInfoFromJson(const nlohmann::json &json);
llvm::Expected<std::vector<FusionInfo>> fusionInfosFromJson(
    const nlohmann::json &json);

llvm::Error writeFusionMetadataJson(const std::vector<FusionInfo> &infos,
                                    const std::string &path);
llvm::Expected<std::vector<FusionInfo>> readFusionMetadataJson(
    const std::string &path);

}  // namespace annc::fusion

#endif  // ANNC_FUSION_METADATA_FUSION_METADATA_JSON_H

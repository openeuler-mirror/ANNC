#ifndef ANNC_FUSION_METADATA_TENSOR_ENDPOINT_H_
#define ANNC_FUSION_METADATA_TENSOR_ENDPOINT_H_

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace annc::fusion {

struct TensorEndpoint {
  std::string node;
  int64_t port = 0;
  bool control = false;

  std::string canonicalDataName() const;
};

llvm::Expected<TensorEndpoint> parseTensorEndpoint(llvm::StringRef name);

}  // namespace annc::fusion

#endif  // ANNC_FUSION_METADATA_TENSOR_ENDPOINT_H_

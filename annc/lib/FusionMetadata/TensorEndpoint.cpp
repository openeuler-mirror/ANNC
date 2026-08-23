#include "FusionMetadata/TensorEndpoint.h"

#include <cerrno>
#include <cstdlib>
#include <system_error>

namespace annc::fusion {

std::string TensorEndpoint::canonicalDataName() const {
  return node + ":" + std::to_string(port);
}

llvm::Expected<TensorEndpoint> parseTensorEndpoint(llvm::StringRef input) {
  if (input.empty()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "tensor endpoint is empty");
  }

  TensorEndpoint endpoint;
  llvm::StringRef value = input;
  if (value.front() == '^') {
    endpoint.control = true;
    value = value.drop_front();
    if (value.empty() || value.contains(':')) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "control endpoint must not contain a data port: %s",
          input.str().c_str());
    }
    endpoint.node = value.str();
    return endpoint;
  }

  size_t colon = value.find(':');
  if (colon == llvm::StringRef::npos) {
    endpoint.node = value.str();
    return endpoint;
  }
  if (colon == 0 || value.find(':', colon + 1) != llvm::StringRef::npos ||
      colon + 1 == value.size()) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "malformed tensor endpoint: %s",
                                   input.str().c_str());
  }
  endpoint.node = value.take_front(colon).str();
  llvm::StringRef portText = value.drop_front(colon + 1);
  for (char c : portText) {
    if (c < '0' || c > '9') {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "malformed tensor endpoint port: %s",
                                     input.str().c_str());
    }
  }
  errno = 0;
  char *end = nullptr;
  std::string portString = portText.str();
  long long parsed = std::strtoll(portString.c_str(), &end, 10);
  if (errno == ERANGE || end != portString.c_str() + portString.size() ||
      parsed < 0) {
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invalid tensor endpoint port: %s",
                                   input.str().c_str());
  }
  endpoint.port = parsed;
  return endpoint;
}

}  // namespace annc::fusion

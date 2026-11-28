#ifndef ANNC_PACKED_RHS_H
#define ANNC_PACKED_RHS_H
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Kernel/PackedRhs.h"
namespace annc::jit {
struct PackedRhsBuffer {
  ~PackedRhsBuffer();
  void* data = nullptr;
};
// Owned by one immutable model node. Entries outlive code cache eviction.
class PackedRhsCache {
 public:
  std::shared_ptr<PackedRhsBuffer> GetOrPack(const AnncPackedRhsSpec& spec,
                                             const float* source,
                                             AnncPreparePackedRhs prepare,
                                             int64_t index, std::string* error);
  size_t size() const;

 private:
  struct Entry {
    AnncPackedRhsSpec spec;
    std::shared_ptr<PackedRhsBuffer> buffer;
  };
  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
};
bool ValidatePackedRhsSpec(const AnncPackedRhsSpec& spec);
bool SamePackedRhsLayout(const AnncPackedRhsSpec& a,
                         const AnncPackedRhsSpec& b);
}  // namespace annc::jit
#endif

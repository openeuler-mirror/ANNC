#include "annc_packed_rhs.h"

#include <cstdlib>
#include <limits>
#include <tuple>
namespace annc::jit {
static_assert(sizeof(AnncPackedRhsSpec) == 10 * sizeof(int64_t),
              "Packed RHS metadata must match generated C layout");
PackedRhsBuffer::~PackedRhsBuffer() { std::free(data); }
bool ValidatePackedRhsSpec(const AnncPackedRhsSpec& s) {
  if (s.version != 1 || s.argument < 0 || s.k <= 0 || s.n <= 0 ||
      s.k > INT32_MAX || s.n > INT32_MAX || s.kc <= 0 || s.nc <= 0 ||
      s.kc > INT32_MAX || s.nc > INT32_MAX || s.isa < 0 || s.isa > 1 ||
      s.panel <= 0 || s.panel > 256 || (s.isa == 0 && s.panel != 4) ||
      (s.isa == 1 && (s.panel < 16 || s.panel % 16 != 0)) ||
      s.alignment != 64 || s.elements <= 0 ||
      static_cast<uint64_t>(s.elements) >
          std::numeric_limits<size_t>::max() / sizeof(float))
    return false;
  const uint64_t full = s.n / s.nc, tail = s.n % s.nc;
  const uint64_t padded = full * ((s.nc + s.panel - 1) / s.panel * s.panel) +
                          ((tail + s.panel - 1) / s.panel * s.panel);
  return padded <= static_cast<uint64_t>(INT64_MAX) / s.k &&
         padded * s.k == static_cast<uint64_t>(s.elements);
}
bool SamePackedRhsLayout(const AnncPackedRhsSpec& a,
                         const AnncPackedRhsSpec& b) {
  return std::tie(a.version, a.argument, a.k, a.n, a.kc, a.nc, a.isa, a.panel,
                  a.elements, a.alignment) ==
         std::tie(b.version, b.argument, b.k, b.n, b.kc, b.nc, b.isa, b.panel,
                  b.elements, b.alignment);
}
std::shared_ptr<PackedRhsBuffer> PackedRhsCache::GetOrPack(
    const AnncPackedRhsSpec& spec, const float* source,
    AnncPreparePackedRhs prepare, int64_t index, std::string* error) {
  error->clear();
  if (!ValidatePackedRhsSpec(spec) || !source || !prepare || index < 0) {
    *error = "Invalid packed RHS contract or preparation entry";
    return nullptr;
  }
  // Cold path only. No process-wide lock or partially published buffers.
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& entry : entries_)
    if (SamePackedRhsLayout(entry.spec, spec)) return entry.buffer;
  auto buffer = std::make_shared<PackedRhsBuffer>();
  if (posix_memalign(&buffer->data, spec.alignment,
                     spec.elements * sizeof(float)) != 0) {
    *error = "Cannot allocate packed RHS";
    return nullptr;
  }
  prepare(index, source, static_cast<float*>(buffer->data));
  entries_.push_back({spec, buffer});
  return buffer;
}
size_t PackedRhsCache::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}
}  // namespace annc::jit

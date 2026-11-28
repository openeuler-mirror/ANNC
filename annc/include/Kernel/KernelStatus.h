#ifndef ANNC_KERNEL_STATUS_H
#define ANNC_KERNEL_STATUS_H

#include <cstdint>

namespace annc {
namespace kernels {

enum class KernelStatus : std::int32_t {
    Success = 0,
    InvalidArgument = 1,
    RuntimeError = 2,
    UnknownError = 3,
};

inline bool succeeded(KernelStatus status) noexcept {
    return status == KernelStatus::Success;
}

inline bool failed(KernelStatus status) noexcept {
    return !succeeded(status);
}

inline std::int32_t statusCode(KernelStatus status) noexcept {
    return static_cast<std::int32_t>(status);
}

} // namespace kernels
} // namespace annc

#endif // ANNC_KERNEL_STATUS_H

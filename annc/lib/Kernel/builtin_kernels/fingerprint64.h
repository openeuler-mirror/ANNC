#ifndef ANNC_BUILTIN_KERNELS_FINGERPRINT64_H
#define ANNC_BUILTIN_KERNELS_FINGERPRINT64_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace annc::kernels {
namespace fingerprint_detail {

constexpr uint64_t k0 = 0xc3a5c85c97cb3127ULL;
constexpr uint64_t k1 = 0xb492b66fbe98f273ULL;
constexpr uint64_t k2 = 0x9ae16a3b2f90404fULL;
constexpr uint64_t kMul = 0x9ddfea08eb382d69ULL;

inline uint64_t fetch64(const char* p) {
    uint64_t result;
    std::memcpy(&result, p, sizeof(result));
    return result;
}

inline uint32_t fetch32(const char* p) {
    uint32_t result;
    std::memcpy(&result, p, sizeof(result));
    return result;
}

inline uint64_t rotate(uint64_t value, int shift) {
    return shift == 0 ? value : ((value >> shift) | (value << (64 - shift)));
}

inline uint64_t shiftMix(uint64_t value) {
    return value ^ (value >> 47);
}

inline uint64_t hashLen16(uint64_t u, uint64_t v, uint64_t mul) {
    uint64_t a = (u ^ v) * mul;
    a ^= (a >> 47);
    uint64_t b = (v ^ a) * mul;
    b ^= (b >> 47);
    b *= mul;
    return b;
}

inline uint64_t hashLen16(uint64_t u, uint64_t v) {
    return hashLen16(u, v, kMul);
}

inline uint64_t hashLen0To16(const char* s, size_t len) {
    if (len >= 8) {
        const uint64_t mul = k2 + len * 2;
        const uint64_t a = fetch64(s) + k2;
        const uint64_t b = fetch64(s + len - 8);
        const uint64_t c = rotate(b, 37) * mul + a;
        const uint64_t d = (rotate(a, 25) + b) * mul;
        return hashLen16(c, d, mul);
    }
    if (len >= 4) {
        const uint64_t mul = k2 + len * 2;
        const uint64_t a = fetch32(s);
        return hashLen16(len + (a << 3), fetch32(s + len - 4), mul);
    }
    if (len > 0) {
        const uint8_t a = static_cast<uint8_t>(s[0]);
        const uint8_t b = static_cast<uint8_t>(s[len >> 1]);
        const uint8_t c = static_cast<uint8_t>(s[len - 1]);
        const uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
        const uint32_t z = static_cast<uint32_t>(len) + (static_cast<uint32_t>(c) << 2);
        return shiftMix(y * k2 ^ z * k0) * k2;
    }
    return k2;
}

inline uint64_t hashLen17To32(const char* s, size_t len) {
    const uint64_t mul = k2 + len * 2;
    const uint64_t a = fetch64(s) * k1;
    const uint64_t b = fetch64(s + 8);
    const uint64_t c = fetch64(s + len - 8) * mul;
    const uint64_t d = fetch64(s + len - 16) * k2;
    return hashLen16(rotate(a + b, 43) + rotate(c, 30) + d,
                     a + rotate(b + k2, 18) + c, mul);
}

inline uint64_t hashLen33To64(const char* s, size_t len) {
    const uint64_t mul = k2 + len * 2;
    uint64_t a = fetch64(s) * k2;
    uint64_t b = fetch64(s + 8);
    uint64_t c = fetch64(s + len - 8) * mul;
    uint64_t d = fetch64(s + len - 16) * k2;
    uint64_t y = rotate(a + b, 43) + rotate(c, 30) + d;
    uint64_t z = hashLen16(y, a + rotate(b + k2, 18) + c, mul);
    uint64_t e = fetch64(s + 16) * mul;
    uint64_t f = fetch64(s + 24);
    uint64_t g = (y + fetch64(s + len - 32)) * mul;
    uint64_t h = (z + fetch64(s + len - 24)) * mul;
    return hashLen16(rotate(e + f, 43) + rotate(g, 30) + h,
                     e + rotate(f + a, 18) + g, mul);
}

inline std::pair<uint64_t, uint64_t> weakHashLen32WithSeeds(
    uint64_t w, uint64_t x, uint64_t y, uint64_t z,
    uint64_t a, uint64_t b) {
    a += w;
    b = rotate(b + a + z, 21);
    const uint64_t c = a;
    a += x;
    a += y;
    b += rotate(a, 44);
    return {a + z, b + c};
}

inline std::pair<uint64_t, uint64_t> weakHashLen32WithSeeds(
    const char* s, uint64_t a, uint64_t b) {
    return weakHashLen32WithSeeds(fetch64(s), fetch64(s + 8),
                                  fetch64(s + 16), fetch64(s + 24), a, b);
}

} // namespace fingerprint_detail

inline uint64_t Fingerprint64(const char* data, size_t size) {
    using namespace fingerprint_detail;

    if (size <= 16) return hashLen0To16(data, size);
    if (size <= 32) return hashLen17To32(data, size);
    if (size <= 64) return hashLen33To64(data, size);

    uint64_t x = fetch64(data + size - 40);
    uint64_t y = fetch64(data + size - 16) + fetch64(data + size - 56);
    uint64_t z = hashLen16(fetch64(data + size - 48) + size,
                           fetch64(data + size - 24));
    auto v = weakHashLen32WithSeeds(data + size - 64, size, z);
    auto w = weakHashLen32WithSeeds(data + size - 32, y + k1, x);
    x = x * k1 + fetch64(data);

    size = (size - 1) & ~static_cast<size_t>(63);
    do {
        x = rotate(x + y + v.first + fetch64(data + 8), 37) * k1;
        y = rotate(y + v.second + fetch64(data + 48), 42) * k1;
        x ^= w.second;
        y += v.first + fetch64(data + 40);
        z = rotate(z + w.first, 33) * k1;
        v = weakHashLen32WithSeeds(data, v.second * k1, x + w.first);
        w = weakHashLen32WithSeeds(data + 32, z + w.second,
                                   y + fetch64(data + 16));
        std::swap(z, x);
        data += 64;
        size -= 64;
    } while (size != 0);

    return hashLen16(hashLen16(v.first, w.first) + shiftMix(y) * k1 + z,
                     hashLen16(v.second, w.second) + x);
}

} // namespace annc::kernels

#endif // ANNC_BUILTIN_KERNELS_FINGERPRINT64_H

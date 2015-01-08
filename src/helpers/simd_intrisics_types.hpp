#ifndef SIMD_INTRISICS_TYPES_HPP
#define SIMD_INTRISICS_TYPES_HPP

/// Helper types that make the life easier when
/// handling SIMD related code

#include <boost/cstdint.hpp>

// emmintrin will define __m128i
// and include the SSE2 intrinsics
#include <emmintrin.h>

namespace doppia {


typedef __m128i m128i;
typedef __m64 m64;

typedef union {
    m128i m;
    boost::uint8_t v[16];
} v16qi; // gives access to __v16qi

typedef union {
    m64 m;
    boost::uint8_t v[8];
} v8qi; // gives access to __v8qi

typedef union {
    m64 m;
    boost::uint16_t v[4];
} v4hi; // gives access to __v4hi


typedef union {
    m128i m;
    boost::uint32_t v[4];
} v2di; // gives access to __v2di


} // end of namespace doppia

#endif // SIMD_INTRISICS_TYPES_HPP

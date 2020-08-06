/*
 * --- xoshiro128+/256+ ---
 *
 * Lightweight API for the Blackman/Vigna XOSHIRO256+ (public domain)
 * pseudorandom number generator for U[0,1) double-precision floats.
 * Also API for XOSHIRO128+ for U[0,1) single-precision floats.
 *
 * Includes facilities for jump-ahead in PRNG stream.
 * And includes utility for state initialization from seed.
 *
 * Adapted from: http://prng.di.unimi.it/
 *
 */

#ifndef __XOSHIRO1N2PLUS_H__
#define __XOSHIRO1N2PLUS_H__

#include <stdint.h>

/* --- splitmix initializer tools --- */

typedef struct __splitmix64_state {
  uint64_t s;
} __splitmix64_state;

uint64_t __splitmix64(__splitmix64_state* state) {
  uint64_t result = state->s;
  state->s = result + 0x9E3779B97f4A7C15;
  result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
  result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
  return result ^ (result >> 31);
}

/* --- PRNG state holder structs --- */

typedef struct xoshiro128plus_state {
  uint32_t s[4];
} xoshiro128plus_state;

typedef struct xoshiro256plus_state {
  uint64_t s[4];
} xoshiro256plus_state;

/* --- common helper routines --- */

static inline uint32_t rotl32(const uint32_t x, int k) {
  return (x << k) | (x >> (32 - k));
}

static inline uint64_t rotl64(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

/* --- xoshiro128+ --- */

uint32_t xoshiro128plus_next(xoshiro128plus_state* xst) {
  uint32_t* s = xst->s;
  const uint32_t result = s[0] + s[3];
  const uint32_t t = s[1] << 9;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl32(s[3], 11);
  return result;
}

/* Returns an uniform random float from the range [0, 1) */
static inline float xoshiro128plus_next_float(xoshiro128plus_state* xst) {
  const uint32_t x = xoshiro128plus_next(xst);
  return ((x >> 8) * 0x1.0p-24);
}

/* Initialize state using a single 64 bit integer seed */
void xoshiro128plus_init(xoshiro128plus_state* xst, uint64_t seed) {
  __splitmix64_state smstate = {seed};
  uint64_t tmp = __splitmix64(&smstate);
  xst->s[0] = (uint32_t)(tmp);
  xst->s[1] = (uint32_t)(tmp >> 32);
  tmp = __splitmix64(&smstate);
  xst->s[2] = (uint32_t)(tmp);
  xst->s[3] = (uint32_t)(tmp >> 32);
}

/* Jump ahead 2^64 steps in PRNG stream */
void xoshiro128plus_jump(xoshiro128plus_state* xst) {
  static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
  uint32_t s0 = 0;
  uint32_t s1 = 0;
  uint32_t s2 = 0;
  uint32_t s3 = 0;
  for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++) {
    for(int b = 0; b < 32; b++) {
      if (JUMP[i] & UINT32_C(1) << b) {
        s0 ^= xst->s[0];
        s1 ^= xst->s[1];
        s2 ^= xst->s[2];
        s3 ^= xst->s[3];
      }
      xoshiro128plus_next(xst); 
    }
  }
  xst->s[0] = s0;
  xst->s[1] = s1;
  xst->s[2] = s2;
  xst->s[3] = s3;
}

/* Jump ahead 2^96 steps in PRNG stream */
void xoshiro128plus_long_jump(xoshiro128plus_state* xst) {
  static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };
  uint32_t s0 = 0;
  uint32_t s1 = 0;
  uint32_t s2 = 0;
  uint32_t s3 = 0;
  for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++) {
    for(int b = 0; b < 32; b++) {
      if (LONG_JUMP[i] & UINT32_C(1) << b) {
        s0 ^= xst->s[0];
        s1 ^= xst->s[1];
        s2 ^= xst->s[2];
        s3 ^= xst->s[3];
      }
      xoshiro128plus_next(xst);
    }
  }
  xst->s[0] = s0;
  xst->s[1] = s1;
  xst->s[2] = s2;
  xst->s[3] = s3;
}

/* --- xoshiro256+ --- */

uint64_t xoshiro256plus_next(xoshiro256plus_state* xst) {
  uint64_t* s = xst->s;
  const uint64_t result = s[0] + s[3];
  const uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl64(s[3], 45);
  return result;
}

static inline double xoshiro256plus_next_double(xoshiro256plus_state* xst) {
  const uint64_t x = xoshiro256plus_next(xst);
  return ((x >> 11) * 0x1.0p-53);
}

/* Initialize state using a single 64 bit integer seed */
void xoshiro256plus_init(xoshiro256plus_state* xst, uint64_t seed) {
  __splitmix64_state smstate = {seed};
  uint64_t tmp = __splitmix64(&smstate);
  xst->s[0] = tmp;
  tmp = __splitmix64(&smstate);
  xst->s[1] = tmp;
  tmp = __splitmix64(&smstate);
  xst->s[2] = tmp;
  tmp = __splitmix64(&smstate);
  xst->s[3] = tmp;
}

/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */
void xoshiro256plus_jump(xoshiro256plus_state* xst) {
  static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };
  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++) {
    for(int b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b) {
        s0 ^= xst->s[0];
        s1 ^= xst->s[1];
        s2 ^= xst->s[2];
        s3 ^= xst->s[3];
      }
      xoshiro256plus_next(xst); 
    }
  }
  xst->s[0] = s0;
  xst->s[1] = s1;
  xst->s[2] = s2;
  xst->s[3] = s3;
}


/* This is the long-jump function for the generator. It is equivalent to
   2^192 calls to next(); it can be used to generate 2^64 starting points,
   from each of which jump() will generate 2^64 non-overlapping
   subsequences for parallel distributed computations. */
void xoshiro256plus_long_jump(xoshiro256plus_state* xst) {
  static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };
  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++) {
    for(int b = 0; b < 64; b++) {
      if (LONG_JUMP[i] & UINT64_C(1) << b) {
        s0 ^= xst->s[0];
        s1 ^= xst->s[1];
        s2 ^= xst->s[2];
        s3 ^= xst->s[3];
      }
      xoshiro256plus_next(xst); 
    }
  }
  xst->s[0] = s0;
  xst->s[1] = s1;
  xst->s[2] = s2;
  xst->s[3] = s3;
}

#endif

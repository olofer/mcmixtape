#ifndef __SHA2_H_
#define __SHA2_H_

/*
 * sha2.h
 *
 * Simple standalone SHA-256 / SHA-224 cryptographic hash implementation.
 * Data to be hashed is required to come in multiples of 8 bits (whole bytes).
 *
 * usage:
 *   sha2_constants S;
 *   sha2_initialize_constants_256(&S);    // only needed once
 *   ...
 *   uint32_t hash[8];
 *   sha2_digest(&S, data1, data1lengthinbytes, hash);
 *   ...
 *   sha2_digest(&S, data2, data2lengthinbytes, hash);
 *   ...
 *
 * alternatively: 
 *   sha2_initialize_constants_224(&S);   // for SHA-224
 *
 */

#ifdef __SHA2_HEXSTRING_

void __sha2_sub_hexstring(const uint32_t* h, char* str, bool upper, int jmax) {
  const uint8_t* bytes = (uint8_t*) h;
  const char* frmtstr = (upper ? "%02X%02X%02X%02X" : "%02x%02x%02x%02x");
  for (int j = 0, i = 0; j < jmax; j++) {
    sprintf(str, frmtstr, bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]);
    str += 8;
    i += 4;
  }
  *str = '\0';
}

void sha2_hexstring_lower(const uint32_t* h, char* str) {
  __sha2_sub_hexstring(h, str, false, 8);
}

void sha2_hexstring_upper(const uint32_t* h, char* str) {
  __sha2_sub_hexstring(h, str, true, 8);
}

void sha2_hexstring_lower_224(const uint32_t* h, char* str) {
  __sha2_sub_hexstring(h, str, false, 7);
}

void sha2_hexstring_upper_224(const uint32_t* h, char* str) {
  __sha2_sub_hexstring(h, str, true, 7);
}

#endif

typedef struct sha2_constants {
  uint32_t h[8];
  uint32_t k[64];
} sha2_constants;

uint32_t __sha2_sub_rotate_right(uint32_t x, uint8_t c) {
  return (x >> c) | (x << (32 - c));
}

uint32_t __sha2_sub_sigma0(uint32_t x) {
  uint32_t y = __sha2_sub_rotate_right(x, 7);
  y ^= __sha2_sub_rotate_right(x, 18);
  return (y ^ (x >> 3)); 
}

uint32_t __sha2_sub_sigma1(uint32_t x) {
  uint32_t y = __sha2_sub_rotate_right(x, 17);
  y ^= __sha2_sub_rotate_right(x, 19);
  return (y ^ (x >> 10));
}

uint32_t __sha2_sub_Sigma0(uint32_t x) {
  uint32_t y = __sha2_sub_rotate_right(x, 2);
  y ^= __sha2_sub_rotate_right(x, 13);
  return (y ^ __sha2_sub_rotate_right(x, 22));
}

uint32_t __sha2_sub_Sigma1(uint32_t x) {
  uint32_t y = __sha2_sub_rotate_right(x, 6);
  y ^= __sha2_sub_rotate_right(x, 11);
  return (y ^ __sha2_sub_rotate_right(x, 25));
} 

// Absorb a single full chunk of data (fixed length = 64 bytes, 512 bits) into the hash state h
void __sha2_sub_consume(const uint32_t* k, uint32_t* h, const uint8_t* d) {
  uint32_t lh[8] = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]};
  uint32_t w[16];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 16; j++) {
      if (i == 0) {
        w[j] = ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) | ((uint32_t)d[2] << 8) | ((uint32_t)d[3]);
        d += 4;
      } else {
        const uint32_t s0 = __sha2_sub_sigma0(w[(j + 1) & 0xf]);
        const uint32_t s1 = __sha2_sub_sigma1(w[(j + 14) & 0xf]);
        w[j] += s0 + w[(j + 9) & 0xf] + s1;
      }
      const uint32_t S1 = __sha2_sub_Sigma1(lh[4]);
      const uint32_t ch = (lh[4] & lh[5]) ^ (~lh[4] & lh[6]);
      const uint32_t temp1 = lh[7] + S1 + ch + k[i << 4 | j] + w[j];
      const uint32_t S0 = __sha2_sub_Sigma0(lh[0]);
      const uint32_t maj = (lh[0] & lh[1]) ^ (lh[0] & lh[2]) ^ (lh[1] & lh[2]);
      const uint32_t temp2 = S0 + maj;
      lh[7] = lh[6];
      lh[6] = lh[5];
      lh[5] = lh[4];
      lh[4] = lh[3] + temp1;
      lh[3] = lh[2];
      lh[2] = lh[1];
      lh[1] = lh[0];
      lh[0] = temp1 + temp2;
    }
  }
  for (int i = 0; i < 8; i++) {
    h[i] += lh[i];
  }
}

void sha2_initialize_constants_hc(sha2_constants* s, bool sha256) {
  const uint32_t h256[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
  const uint32_t h224[8] = {
    0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939, 0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4 };
  const uint32_t* h = (sha256 ? h256 : h224);
  for (int i = 0; i < 8; i++)
    s->h[i] = h[i];
  const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
  for (int i = 0; i < 64; i++)
    s->k[i] = k[i];
}

void sha2_initialize_constants_256(sha2_constants* s) {
  sha2_initialize_constants_hc(s, true);
}

void sha2_initialize_constants_224(sha2_constants* s) {
  sha2_initialize_constants_hc(s, false);
}

uint64_t sha2_digest(const sha2_constants* s, 
                     const uint8_t* bytes, 
                     uint64_t numbytes, 
                     uint32_t* digest)
{
  uint8_t buffer[128]; // 1024 bit buffer
  uint32_t h[8] = {s->h[0], s->h[1], s->h[2], s->h[3], s->h[4], s->h[5], s->h[6], s->h[7]};

  // 512 bits per chunk is 64 bytes; this implementation's "atomic" unit is 8 bits
  uint64_t fullchunks = 0;
  const uint8_t* data = bytes;
  uint64_t msg_bytes_to_go = numbytes;
  while (msg_bytes_to_go >= 64) {
    __sha2_sub_consume(s->k, h, data);
    msg_bytes_to_go -= 64;
    data += 64;
    fullchunks++;
  }

  /*
  begin with the original message of length L bits
  append a single '1' bit
  append K '0' bits, where K is the minimum number >= 0 such that L + 1 + K + 64 is a multiple of 512
  append L as a 64-bit big-endian integer, making the total post-processed length a multiple of 512 bits
  such that the bits in the message are L 1 00..<K 0's>..00 <L as 64 bit integer> = k*512 total bits
  */

  int b = 0;
  while (msg_bytes_to_go--)
    buffer[b++] = *data++;
  buffer[b++] = 0x80;

  if (b + 8 < 64) {
    while (b + 8 < 64)
      buffer[b++] = 0x00;
  } else {
    while (b + 8 < 128)
      buffer[b++] = 0x00;
  }

  const uint64_t total_bits = numbytes << 3;
  uint8_t* l = (uint8_t*) &buffer[b];
  l[0] = (uint8_t) (total_bits >> 56); // big end of 64 bit word
  l[1] = (uint8_t) (total_bits >> 48);
  l[2] = (uint8_t) (total_bits >> 40);
  l[3] = (uint8_t) (total_bits >> 32);
  l[4] = (uint8_t) (total_bits >> 24);
  l[5] = (uint8_t) (total_bits >> 16);
  l[6] = (uint8_t) (total_bits >> 8);
  l[7] = (uint8_t) (total_bits);

  if (b < 64) { // must hold b == 56, 56+7 = 63
    __sha2_sub_consume(s->k, h, buffer);
    fullchunks++;
  } else { // else b == 120
    __sha2_sub_consume(s->k, h, buffer);
    __sha2_sub_consume(s->k, h, &buffer[64]);
    fullchunks += 2;
  }

  uint8_t* hash = (uint8_t*) digest;
  for (int i = 0, j = 0; i < 8; i++) {
    hash[j++] = (uint8_t) (h[i] >> 24); // big end
    hash[j++] = (uint8_t) (h[i] >> 16);
    hash[j++] = (uint8_t) (h[i] >> 8);
    hash[j++] = (uint8_t) (h[i]);
  }

  return fullchunks;
}

#endif

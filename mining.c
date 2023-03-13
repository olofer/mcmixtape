/*
  mining.c

  Hash mining illustration.
  Find a seed code such that a specific number of consecutive zeros start the SHA-256 digest.
  Run parallel 'miners', one per thread (OpenMP). 
  The thread which first finds the correct 'nonce' gets rewarded a coin.
  This is a cartoon of the "proof-of-work" consensus mechanism.

  USAGE:
    mining [what-to-do] [parameters ...]
    (./mining.exe ...)

    what-to-do          parameters                           comment  

      blockchain          numthreads ...

      constants           (none)                               SHA256 constants printout
      bitprint            (none)                               Old test code
      endian              (none)                               Little or big endian? 
      digest              text1 text2 ...                      Generate SHA256 hash for text
      digestfile          file1 file2 ...                      Generate SHA256 hash for data in file
      tests               (none)                               Run a few builtin SHA256 test-cases

  BUILD:
    gcc -O2 -Wall -o mining.exe mining.c -fopenmp

 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include "xoshiro1n2plus.h"

#define __SHA2_HEXSTRING_
#include "sha2.h"

bool little_endian() {
  const uint32_t x = 0x01;
  return *((uint8_t *)(&x)) == 1;
}

void make_bit_string(uint32_t x, char* str) {
  for (int b = 0; b < 32; b++) {
    const bool hasbitb = x & (0x01 << b); 
    str[32 - b - 1] = (hasbitb ? '1' : '0');
  }
  str[32] = '\0';
}

typedef struct tBlock {
  uint64_t index;
  uint32_t hash[8]; // SHA256 hash from previous block
  uint64_t nonce;
  uint64_t payload[8];
} tBlock;

uint32_t hash_ctz(const uint32_t* x, int w) // count trailing zeros
{
  uint32_t c = 0;
  for (int i = 0; i < w; i++) {
    if (x[i] == 0) {
      c += 32;
      continue;
    }
    c += __builtin_ctz(x[i]);
    break;
  }
  return c;
}

int main(int argc, char** argv)
{
  if (argc == 1) {
    printf("USAGE: %s [what-to-do] [parameters ...]\n", argv[0]);
    printf("See comment block in source: \"%s\"\n", __FILE__);
    return 0;
  }

  const char* what_to_do = argv[1];

  if (strcmp(what_to_do, "constants") == 0) {
    sha2_constants A;
    sha2_initialize_constants_256(&A);
    for (int i = 0; i < 8; i++) {
      printf("h[%02i] = 0x%08x\n", i, A.h[i]);
    }
    for (int i = 0; i < 64; i++) {
      printf("k[%02i] = 0x%08x\n", i, A.k[i]);
    }
    return 0;
  }

  if (strcmp(what_to_do, "bitprint") == 0) {
    char str[33];
    for (uint32_t i = 0; i < 256; i++) {
      make_bit_string(i, str);
      printf("%04i | %s | 0x%08x\n", i, str, i);
    }
    return 0;
  }

  if (strcmp(what_to_do, "endian") == 0) {
    printf("endian-ness = %s\n", (little_endian() ? "little" : "big"));
    return 0;
  }

  if (strcmp(what_to_do, "digest") == 0) {
    sha2_constants A;
    sha2_initialize_constants_256(&A);
    uint32_t hash[8];
    char str[65]; // only need space for 8*8+1 chars

    if (argc == 2) {
      sha2_digest(&A, (const uint8_t *) NULL, (uint64_t) 0, hash);
      sha2_hexstring_lower(hash, str);
      printf("bytes = %03i | sha256 = %s\n", (int) 0, str);
      return 0;
    }

    for (int i = 2; i < argc; i++) {
      sha2_digest(&A, (const uint8_t *) argv[i], (uint64_t) strlen(argv[i]), hash);
      sha2_hexstring_lower(hash, str);
      printf("bytes = %03i | sha256 = %s\n", (int) strlen(argv[i]), str);
    }
    return 0;
  }

  if (strcmp(what_to_do, "digestfile") == 0) {
    sha2_constants A;
    sha2_initialize_constants_256(&A);
    uint32_t hash[8];
    char str[65];
    if (argc == 2) {
      printf("no filename(s) provided.\n");
      return 1;
    }

    for (int i = 2; i < argc; i++) {
      uint8_t* buffer = NULL;
      int64_t nread = 0;
      FILE* pf = fopen(argv[i], "rb"); 
      if (pf != NULL) {
        fseek(pf, 0, SEEK_END);
        int64_t sz = ftell(pf);
        fseek(pf, 0, SEEK_SET);
        buffer = malloc(sz);
        if (buffer != NULL) {
          nread = fread(buffer, 1, sz, pf);
        }
        fclose(pf);
        if (buffer != NULL && nread == sz) {
          sha2_digest(&A, buffer, sz, hash);
          sha2_hexstring_lower(hash, str);
          printf("sha256 = %s | file = \"%s\" (%lld bytes)\n", str, argv[i], sz);
        } else {
          printf("opened \"%s\" (%lld bytes) but failed to read into buffer.\n", argv[i], sz);
        }
        if (buffer != NULL)
          free(buffer);
      } else {
        printf("failed to open \"%s\"\n", argv[i]);
      }
    }

    return 0;
  }

  if (strcmp(what_to_do, "tests") == 0) {
    // https://www.cosic.esat.kuleuven.be/nessie/testvectors/hash/sha/Sha-2-256.unverified.test-vectors
    sha2_constants A;
    sha2_initialize_constants_256(&A);
    uint32_t hash[8];
    char str[65];
    uint8_t msg[128];
    memset(msg, 0, 32);
    const char hashstr1[] = "66687AADF862BD776C8FC18B8E9F8E20089714856EE233B3902A591D0D5F2925";
    const char hashstr2[] = "B422BC9C0646A432433C2410991C95E2D89758E3B4F540ACA863389F28A11379"; 
    sha2_digest(&A, msg, 32, hash);
    sha2_hexstring_upper(hash, str);

    printf("sha256:       %s\n", str);
    printf("should be:    %s | match = %i\n", hashstr1, strcmp(str, hashstr1) == 0);

    const int niters = 100000;
    for (int i = 1; i < niters; i++) {
      memcpy(msg, hash, 32);
      sha2_digest(&A, msg, 32, hash);
    }

    sha2_hexstring_upper(hash, str);
    printf("sha256:       %s | iterated %i times\n", str, niters);
    printf("should be:    %s | match = %i\n", hashstr2, strcmp(str, hashstr2) == 0);

    memset(msg, 0, 128);

    const int numZeroBytes[] = {
      0,
      1,
      2,
      3,
      4, 
      11,
      17,
      32,
      64,
      127
    };

    const char* hashes[] = {
      "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
      "6E340B9CFFB37A989CA544E6BB780A2C78901D3FB33738768511A30617AFA01D",
      "96A296D224F285C67BEE93C30F8A309157F0DAA35DC5B87E410B78630A09CFC7",
      "709E80C88487A2411E1EE4DFB9F22A861492D20C4765150C0C794ABD70F8147C",
      "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119",
      "71B6C1D53832F789A7F2435A7C629245FA3761AD8487775EBF4957330213A706",
      "0A88111852095CAE045340EA1F0B279944B2A756A213D9B50107D7489771E159",
      "66687AADF862BD776C8FC18B8E9F8E20089714856EE233B3902A591D0D5F2925",
      "F5A5FD42D16A20302798EF6ED309979B43003D2320D9F0E8EA9831A92759FB4B",
      "15DAE5979058BFBF4F9166029B6E340EA3CA374FEF578A11DC9E6E923860D7AE"
    };

    for (int i = 0; i < 10; i++) {
      sha2_digest(&A, msg, numZeroBytes[i], hash); 
      sha2_hexstring_upper(hash, str);
      printf("[%02i] sha256:  %s | %i bytes of zeros\n", i, str, numZeroBytes[i]);
      printf("should be:    %s | match = %i\n", hashes[i], strcmp(str, hashes[i]) == 0);
    }

    return 0;
  }

  if (strcmp(what_to_do, "blockchain") != 0) {
    printf("unrecognized argument: \"%s\": stopping.\n", what_to_do);
    return 1;
  }

  if (argc != 3) {
    printf("USAGE: %s blockchain [nworkers]\n", argv[0]);
    return 1;
  }

  const int maxworkers = omp_get_max_threads();
  int nworkers = (int) strtod(argv[2], NULL);
  nworkers = (nworkers < 1 ? 1 : (nworkers > maxworkers ? maxworkers : nworkers));

  printf("num. parallel workers/miners =  %i\n", nworkers);

  const uint32_t requiredTrailingZeros = 24;
  const int64_t maxWorkerHashes = 100000000;

  int64_t workerCoins[nworkers];
  memset(workerCoins, 0, sizeof(int64_t) * nworkers);

  int64_t workerHashes[nworkers];
  memset(workerHashes, 0, sizeof(int64_t) * nworkers);

  int64_t workerCopies[nworkers];
  memset(workerCopies, 0, sizeof(int64_t) * nworkers);

  omp_set_num_threads(nworkers);

  uint64_t seed = 0x1234abcd4321efab;
  //uint64_t seed = 0x4321abcd4321efab;

  tBlock currentBlock; // all threads compete to mine this block in parallel
  memset(&currentBlock, 0, sizeof(tBlock)); // initial block (ignoring nonce) is just zeros

  printf("will run %lliM hashes per miner, requiring %i trailing zeros or more\n", maxWorkerHashes / 1000000, requiredTrailingZeros);

  #pragma omp parallel
  {
    const int worker_id = omp_get_thread_num();
    const int worker_count = omp_get_num_threads();
    if (worker_id == 0) {
      printf("worker %i reports that there are %i workers in total.\n", worker_id, worker_count);
    }

    xoshiro256plus_state local_rng;

    xoshiro256plus_init(&local_rng, seed);
    for (int j = 0; j < worker_id; j++)
      xoshiro256plus_long_jump(&local_rng);

    sha2_constants A;
    sha2_initialize_constants_256(&A);
    uint32_t localHash[8];

    tBlock localBlock;
    localBlock.index = (uint64_t) -1;

    //uint64_t localNonce = 0;
    uint64_t localNonce = xoshiro256plus_next(&local_rng);

    while (true)
    {
      if (workerHashes[worker_id] == maxWorkerHashes) {
        printf("worker %i hashed out\n", worker_id);
        break;
      }

      if (localBlock.index != currentBlock.index) {
        #pragma omp critical(shared_problem)
        {
          memcpy(&localBlock, &currentBlock, sizeof(tBlock));
        }
        workerCopies[worker_id]++;
      }

      if (localBlock.index == currentBlock.index) {
        localBlock.nonce = localNonce++;
        sha2_digest(&A, (const uint8_t *) &localBlock, (uint64_t) sizeof(tBlock), localHash);
        const uint32_t ntz = hash_ctz(localHash, 8);

        if (ntz >= requiredTrailingZeros) {
          #pragma omp critical(shared_problem)
          {
            if (localBlock.index == currentBlock.index) {
              memcpy(currentBlock.hash, localHash, sizeof(uint32_t) * 8); // chain block
              for (int i = 0; i < 8; i++) currentBlock.payload[i] = xoshiro256plus_next(&local_rng); // random payload
              currentBlock.index++;
              currentBlock.nonce = 0;
              workerCoins[worker_id]++;
            }
          }

          if (localBlock.index != currentBlock.index) {
            char str[65];
            sha2_hexstring_upper(localHash, str);
            printf("good hash = \"%s\" @ worker %i\n", str, worker_id);
          }

        }

        workerHashes[worker_id]++;
      }
    }

  }

  int totalCoins = 0;
  for (int i = 0; i < nworkers; i++) {
    printf("worker/thread [%03i]: copies=%lli, hashes=%lliM, coins=%lli\n", 
           i, 
           workerCopies[i], 
           workerHashes[i] / 1000000, 
           workerCoins[i]);
    totalCoins += workerCoins[i];
  }

  printf("total coins awarded = %i\n", totalCoins);

  return 0;
}

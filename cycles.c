/*
 * Sampler for maximum cycle lengths (and number of cycles) of random permutations.
 *
 * BUILD:
 *   gcc -Wall -O2 -o cycles.exe cycles.c
 *
 * USAGE:
 *   cycles n r fy        n slots, r repeats, Fisher-Yates;   max cycle sampled r times
 *   cycles n r sw 15     n slots, r repeats, k random swaps; max cycle sampled r times
 *   cycles n r sa        (degenerate stats. only a single full length cycle permutation)
 *   cycles n r rs 6      6 riffle-shuffles should be equivalent to fisher-yates if n=52
 *
 * Randomization options:
 *   fy = Fisher-Yates (once)
 *   sw = random element swaps (repeated k times)
 *   rs = riffle-shuffle (repeated k times)
 *   sa = Sattolo (once)
 *
 * EXAMPLE:
 *   cycles 100 1e6 fy > cycle-histogram-fy.txt 
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "xoshiro1n2plus.h"

void ResetArray(int* a, int n) {
  for (int i = 0; i < n; i++) {
    a[i] = i;
  }
}

// b is uses as temporary array
void riffle_shuffle(xoshiro256plus_state* rng, int* a, int n, int* b) {
  const int cp = n / 2;
  int lp = 0;
  int rp = cp;
  int i = 0;
  for (;;) {
    const double p = xoshiro256plus_next_double(rng);
    b[i++] = (p < 0.5 ? a[lp++] : a[rp++]);
    if (lp == cp || rp == n) break;
  }
  while (lp != cp) b[i++] = a[lp++];
  while (rp != n) b[i++] = a[rp++];
  for (int j = 0; j < n; j++) 
    a[j] = b[j];
}

void two_element_swap(xoshiro256plus_state* rng, int* a, int n) {
  const int i = (int) (xoshiro256plus_next_double(rng) * n);
  const int j = (int) (xoshiro256plus_next_double(rng) * n);
  if (i == j) return;
  const int tmp = a[i];
  a[i] = a[j];
  a[j] = tmp;
}

/* in-place uniform random permutation of a */
void FisherYatesForward(xoshiro256plus_state* rng, int* a, int n) {
  for (int i = 0; i < n - 1; i++) {
    const int j = i + (int) (xoshiro256plus_next_double(rng) * (n - i));
    const int x = a[i];
    a[i] = a[j];
    a[j] = x;
  }
}

// Generate a random n-cycle permutation
void SattoloFullCycle(xoshiro256plus_state* rng, int* a, int n) {
  int i = n;
  while (i > 1) {
    i--;
    const int j = (int) ((i - 1) * xoshiro256plus_next_double(rng));
    const int tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }
}

// start at slot a[i], then a[a[i]] and so on, until a[j] = i, closing the cycle.
// return the length of the cycle; mark each slot visited in array b with i
int traverse_cycle(const int* a, int n, int i, int* b) {
  int j = i;
  int len = 0;
  for (;;) {
    b[j] = i;
    j = a[j];
    len ++;
    if (j == i) break; 
  }
  return len;
}

// return maximum length of the cycles, and the number of cycles, for array a 
int max_cycle_length(const int* a, int n, int* b, int* nc) {
  for (int i = 0; i < n; i++) b[i] = -1;
  *nc = 0;
  int maxlen = 0;
  int i = 0;
  for (;;) {
    while (b[i] != -1 && i < n) i++;
    if (i == n) break;
    int len = traverse_cycle(a, n, i, b);
    if (len > maxlen) maxlen = len;
    (*nc)++;
  }
  return maxlen;
}

int main(int argc, const char** argv)
{
  if (argc < 4 || argc > 5) {
    printf("USAGE: %s numslots numsamples shuffletype [numshuffles]\n", argv[0]);
    return 0;
  }

  uint64_t seed = 0x1234abcd4321efab;
  xoshiro256plus_state xrng;

  xoshiro256plus_init(&xrng, seed);

  int n = (int) strtod(argv[1], NULL);
  int r = (int) strtod(argv[2], NULL);
  int k = 0;

  const bool use_fisher_yates = (strcmpi(argv[3], "fy") == 0);
  const bool use_riffle_shuffle = (strcmpi(argv[3], "rs") == 0);
  const bool use_swaps = (strcmpi(argv[3], "sw") == 0);
  const bool use_sattolo = (strcmpi(argv[3], "sa") == 0);

  if (!use_fisher_yates && !use_riffle_shuffle && !use_swaps && !use_sattolo) {
    printf("shuffle identifier not recognized among: fy, rs, sw, sa\n");
    return 1;
  }

  if (argc == 5) {
    k = (int) strtod(argv[4], NULL);
  }

  if ((n <= 0 || r <= 0) || k < 0) {
    printf("invalid input: n, r, or k\n");
    return 1;
  }

  int* A = (int*) malloc(sizeof(int) * n);
  int* B = (int*) malloc(sizeof(int) * n); // helper array to mark already visted cycles

  int* L = (int*) malloc(sizeof(int) * (n + 1)); // histogram for maximum cycle length
  memset(L, 0, sizeof(int) * (n + 1));

  int* C = (int*) malloc(sizeof(int) * (n + 1)); // histogram for number of cycles
  memset(C, 0, sizeof(int) * (n + 1));

  for (int i = 0; i < r; i++) {
    ResetArray(A, n);

    int nc;
    int l1 = max_cycle_length(A, n, B, &nc);

    if (l1 != 1 || nc != n) {
      printf("self-test failure (on reset array)\n");
      break;
    }

    if (use_fisher_yates) {
      // This generates a fresh "complete" uniform randomization of the elements
      FisherYatesForward(&xrng, A, n);
    } else if (use_swaps) {
      // This creates an "incomplete" randomization (none at all for k = 0)
      for (int s = 0; s < k; s++)
        two_element_swap(&xrng, A, n);
    } else if (use_riffle_shuffle) {
      for (int s = 0; s < k; s++)
        riffle_shuffle(&xrng, A, n, B);
    } else if (use_sattolo) {
      SattoloFullCycle(&xrng, A, n);
    }

    l1 = max_cycle_length(A, n, B, &nc);

    if (l1 > n || l1 < 1) {
      printf("maximum cycle length out of bounds\n"); 
      break;
    }

    if (nc > n || nc < 1) {
      printf("number of cycles out of bounds\n"); 
      break;
    }

    L[l1]++;
    C[nc]++;
  }

  // sum-up L and check it is equal to r...
  uint64_t suml = 0;
  uint64_t sumc = 0;
  for (int i = 0; i <= n; i++) {
    suml += L[i];
    sumc += C[i];
  }

  if (suml == r && sumc == r) {
    // The sample is complete; write histograms to standard output
    for (int i = 0; i <= n; i++) {
      printf("%16i %16i %16i\n", i, L[i], C[i]);
    }
  } 

  free(A);
  free(B);
  free(L);
  free(C);

  return 0;
}

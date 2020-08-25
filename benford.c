/*
    benford.c

    Explore the Benford-Newcomb distribution: does it emerge from ratio of uniforms?

    USAGE:
      benford numsamples base
      benford numsamples base a1 a2 b1 b2 [numthreads]

    Program will print out two columns of numbers, CSV style:
      (1) the Benford distribution of the first digit
      (2) the empirical distriution of the samples x (if requested)

    Here x = (a1 * u1 + a2) / (b1 * u2 + b2) is a ratio of uniform
    distributions with u1, u2 uniform on (0,1)

    The program takes numsamples * numthreads total samples.
    If numthreads is omitted its default is 1.

    BUILD:
      gcc -O2 -Wall -o benford.exe benford.c -fopenmp

    EXAMPLE:
      benford 1e8 10 2 0 0.5 0 2
      benford 1e8 16 1 0 0.125 0 4
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

#include "xoshiro1n2plus.h"

#define maxdigits 32

double logbase(int b, double x) {
  switch (b) {
    case 2: return log2(x);
    case 10: return log10(x);
    default: return log(x) / log((double) b);
  }
}

double benford(int b, int d) {
  if (d < 1 || d >= b) return 0.0;
  return logbase(b, 1.0 + 1.0 / d);
}

/* b > 1 assumed; returns integer 0 .. b - 1, 0 if failure (e.g. x = 0) */
/* remember finite precision .. */
int firstdigit(int b, double x) {
  if (x == 0.0) return 0;
  if (x < 0.0) x *= -1.0;
  const double logbx = logbase(b, x);
  const int floor_logbx = (int) logbx;
  const double z = logbx - (double) floor_logbx;
  const int dig = (int) pow((double) b, z);
  if (dig < 1 || dig > b - 1) return 0;
  return dig;
}

int main(int argc, const char** argv)
{
  if (argc < 7 || argc > 8) {
    printf("USAGE: %s numsamples base a1 a2 b1 b2 [numthreads]\n", argv[0]);
    return 0;
  }

  const int numsamples = (int) floor(strtod(argv[1], NULL));
  const int digitbase = (int) floor(strtod(argv[2], NULL));
  const double a1 = strtod(argv[3], NULL);
  const double a2 = strtod(argv[4], NULL);
  const double b1 = strtod(argv[5], NULL);
  const double b2 = strtod(argv[6], NULL);

  int numthreads = 1;
  if (argc == 8) {
    numthreads = (int) floor(strtod(argv[7], NULL));
  }

  if (digitbase < 2 || digitbase > maxdigits) {
    printf("(digit) base >= 2 and <= %i, required\n", maxdigits);
    return 0;
  }

  if (numsamples < 5) {
    printf("specify at least 5 samples\n");
    return 0;
  }  

  const int maxthreads = omp_get_max_threads();
  if (numthreads < 1 || numthreads > maxthreads) {
    printf("requested thread count (%i) invalid (min = 1, max = %i)\n", numthreads, maxthreads);
    return 0;
  }

  uint64_t seed = 0x1234abcd4321efab;

  uint64_t totalsamples = 0;
  uint64_t counts[maxdigits];
  memset(counts, 0, maxdigits * sizeof(uint64_t));

  omp_set_num_threads(numthreads);

  #pragma omp parallel
  {
    xoshiro256plus_state local_xrng;
    xoshiro256plus_init(&local_xrng, seed);

    const int tid = omp_get_thread_num();
    const int tnum = omp_get_num_threads();

    if (tid == 0 && tnum != numthreads) {
      printf("did not get requested (%i) threads (%i)\n", tnum, numthreads);
    }

    for (int j = 0; j < tid; j++) {
      xoshiro256plus_long_jump(&local_xrng);
    }

    uint64_t local_counts[maxdigits];
    memset(local_counts, 0, maxdigits * sizeof(uint64_t));

    for (int j = 0; j < numsamples; j++) {
      const double A = a1 * xoshiro256plus_next_double(&local_xrng) + a2;
      const double B = b1 * xoshiro256plus_next_double(&local_xrng) + b2;
      const double xj = A / B;
      const int dj = firstdigit(digitbase, xj);
      local_counts[dj]++;
    }

    #pragma omp critical
    {
      totalsamples += numsamples;
      for (int j = 0; j < digitbase; j++)
        counts[j] += local_counts[j];
    }
  }

  for (int i = 1; i < digitbase; i++) {
    printf("%.6f, %.6f\n", benford(digitbase, i), (double) counts[i] / totalsamples);
  }

  return 0;
}

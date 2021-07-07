/*
 * ctmc.c - Simulation of Continuous Time Markov Chains
 *
 * USAGE:
 *  ctmc [num-samples] [textfile|mode] [chain-name] [chain-parameter(s)]
 *
 *   chain-name | parameters :
 *
 *   erlang     | k lambda
 *   coxian     | k lambda alpha
 *   rw1        | B lambdaL lambdaR
 *   uniform    | N lambda
 *   collatz    | N qinit qabsorb qeven qodd
 *
 * EXAMPLES:
 *   ctmc 10 --verbose-- erlang 1 2.00
 *   ctmc 1e4 --summary-- erlang 5 0.25
 *   ctmc 1e2 --verbose-- erlang 5 0.25
 *   ctmc 1e3 output.txt erlang 3 1.25
 *   ctmc 3 --traject-- erlang 7 0.50
 *   ctmc 1e7 --summary-- coxian 3 1.0 0.95
 *   ctmc 1e6 --summary-- collatz 200 100.0 1e-3 1.0 1.0
 *
 * METHOD: 
 *   Uses the Doob-Gillespie algorithm. Each new jump requires two uniform
 *   psuedorandom numbers. A chain is realized by implementing a callback
 *   function which returns all the jump options and the associated jump
 *   (outgoing) intensities for a given current state. 
 *
 * LIMITATIONS:
 *   Only a few example distributions are implemented.
 *   The random variable generated is the "time to absorption" T > 0.
 *   (the sampling engine is more general).
 *   The samples are drawn internally and summary stats are printed out.
 *   (optional text file sample output).
 *   Only single-threaded computation at this time.
 *   The PRNG seed is hardcoded.
 *   There is a (silly) limit to the number of jumps from a current state.
 *   The states are enumerated with a single integer.
 *
 * (GRAPHICAL) TEST & ILLUSTRATION:
 *   Rscript --vanilla ctmc-example.R 1e6
 *
 * COMPILE:
 *   gcc -Wall -O2 ctmc.c -o ctmc.exe
 *
 */

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

//#include <omp.h>

#include "xoshiro1n2plus.h"

/* Callback parameters:

   (int currentstate, 
    double currenttime, 
    int buffersize,
    int* numstates, 
    int* states,
    double* intensities,
    const void* params)
 */

typedef void (*ctmc_jump_func_ptr)(int, double, int, int*, int*, double*, const void*);

/* --- Erlang(k, lambda) --- */

typedef struct erlang_params {
  int k;
  double lambda;
} erlang_params;

void erlang_jump_function(
  int x, 
  double t, 
  int maxstates,
  int* nj, 
  int* yj, 
  double* lambdaj, 
  const void* params)
{
  const erlang_params* p = (const erlang_params*) params;
  if (x == p->k) {
    *nj = 0;  // absorbing state
    return;
  }
  *nj = 1;
  if (*nj > maxstates) return;
  *yj = x + 1;
  *lambdaj = p->lambda;
}

/* --- Random Walk on a discrete bounded line (start at 0, domain [-B, +B]) --- */

typedef struct rw1_params {
  int B;
  double lambda_down;
  double lambda_up;
} rw1_params;

void rw1_jump_function(
  int x, 
  double t,
  int maxstates, 
  int* nj, 
  int* yj, 
  double* lambdaj, 
  const void* params)
{
  const rw1_params* p = (const rw1_params*) params;
  if (x == p->B || x == -p->B) {
    *nj = 0;  // absorbing domain boundary
    return;
  }
  *nj = 2;
  if (*nj > maxstates) return;
  yj[0] = x - 1;
  lambdaj[0] = p->lambda_down;
  yj[1] = x + 1;
  lambdaj[1] = p->lambda_up;
}

/* --- Basic Coxian distribution / process (k, lambda, alpha) --- */

typedef struct coxian_params {
  int k;
  double lambda;
  double alpha;
} coxian_params;

void coxian_jump_function(
  int x, 
  double t, 
  int maxstates,
  int* nj, 
  int* yj, 
  double* lambdaj, 
  const void* params)
{
  const coxian_params* p = (const coxian_params*) params;
  if (x == p->k) {
    *nj = 0;
    return;
  }
  *nj = 2;
  if (*nj > maxstates) return;
  yj[0] = x + 1;
  lambdaj[0] = p->lambda * p->alpha;
  yj[1] = p->k;
  lambdaj[1] = p->lambda * (1.0 - p->alpha);
}

/* --- Uniformly all-to-all jumping w one absorbing state --- */

typedef struct uniform_params {
  int N; // states 0..N-1 are equivalent; state N is absorbing
  double lambda; // each outgoing intensity is lambda / (N + 1), from every state
} uniform_params;

void uniform_jump_function(
  int x, 
  double t, 
  int maxstates,
  int* nj, 
  int* yj, 
  double* lambdaj, 
  const void* params)
{
  const uniform_params* p = (const uniform_params*) params;
  if (x == p->N) {
    *nj = 0;
    return;
  }
  const int N1 = p->N + 1;
  *nj = N1;
  if (*nj > maxstates) return;
  const double lambda_hat = p->lambda / N1;
  for (int i = 0; i < N1; i++) {
    yj[i] = i;
    lambdaj[i] = lambda_hat;
  }
}

/* --- Collatz --- */

typedef struct collatz_params {
  int N;
  double lambda_init;
  double lambda_absorb;
  double lambda_even;
  double lambda_odd;
} collatz_params;

void collatz_jump_function(
  int x, 
  double t, 
  int maxstates,
  int* nj, 
  int* yj, 
  double* lambdaj, 
  const void* params)
{
  const collatz_params* p = (const collatz_params*) params;
  if (x == 0) { // scatter onto 1..N uniformly
    *nj = p->N;
    if (*nj > maxstates) return;
    const double lambda_scatter = p->lambda_init / *nj;
    for (int i = 0; i < *nj; i++) {
      yj[i] = i + 1;
      lambdaj[i] = lambda_scatter;
    }
    return;
  }
  if (x == 1) {
    *nj = 0;
    return;
  }
  *nj = 2;  // Follow the Collatz sequence, or get absorbed early..
  yj[0] = 1;
  lambdaj[0] = p->lambda_absorb;
  if (x & 0x01) {  // x is odd
    yj[1] = 3 * x + 1;
    lambdaj[1] = p->lambda_odd;
  } else {  // x is even
    yj[1] = x >> 1;
    lambdaj[1] = p->lambda_even;
  }
}

/* --- CTMC simulation engine --- */

/* returns 1 if stepped forward; 0 if absorbed; -1 if error  */
int next_ctmc_state(
  ctmc_jump_func_ptr ctmcfcn,
  const void* params,
  xoshiro256plus_state* xst,
  int buffersize, // do not put too large numbers here
  int state, 
  double time, 
  int* nextstate, 
  double* deltatime)
{
  int ny;
  int y[buffersize];            // VLA
  double lambda[buffersize];    // VLA
  double lambdaSum[buffersize]; // VLA
  // Step 1: apply callback to get all jump options
  (*ctmcfcn)(state, time, buffersize, &ny, y, lambda, params);
  if (ny < 0 || ny > buffersize) {
    *deltatime = -1.0;
    *nextstate = -1;
    return -1;  // pathological
  }
  if (ny == 0) {
    *deltatime = 0.0;
    *nextstate = state;
    return 0;  // absorbed
  }
  // Step 2: sum up intensities; draw 2 uniform
  lambdaSum[0] = lambda[0];
  for (int i = 1; i < ny; i++) {
    lambdaSum[i] = lambdaSum[i - 1] + lambda[i];
  }
  const double r1 = xoshiro256plus_next_double(xst);
  const double r2 = xoshiro256plus_next_double(xst); 
  // Step 3: evaluate deltatime & nextstate
  *deltatime = -1.0 * log(r1) / lambdaSum[ny - 1];  // sojourn time
  const double threshold = r2 * lambdaSum[ny - 1];
  for (int i = 0; i < ny; i++) {
    if (lambdaSum[i] > threshold) {
      *nextstate = y[i];
      return 1;  // healthy exit
    }
  }
  *deltatime = -1.0;
  *nextstate = -1;
  return -1;  // pathological
}

/* --- helper functions --- */

int max_of(int a, int b) {
  return (a >= b ? a : b);
}

/* --- program entrypoint --- */

int main(int argc, const char** argv)
{
  if (argc < 5) {
    printf("usage: %s [num-samples] [textfile|mode] [chain-name] [chain-parameter(s)]\n", argv[0]);
    return 1;
  }

  const int64_t numSamples = (int64_t) floor(strtod(argv[1], NULL));

  if (numSamples <= 0) {
    return 1;
  }

  const char* textfileName = argv[2];
  const char* chainName = argv[3];
  const int numChainParameters = argc - 4;
  const char** strParams = &argv[4];

  const bool verboseMode = strcmp(textfileName, "--verbose--") == 0;
  const bool summaryMode = strcmp(textfileName, "--summary--") == 0;
  const bool trajectMode = strcmp(textfileName, "--traject--") == 0;   // extremely verbose
  const bool writeFile = !verboseMode && ! summaryMode && !trajectMode; // assume textfileName is a writeable filename

  int theBufferSize = 256;

  ctmc_jump_func_ptr theJumpFunction = NULL;

  union {
    erlang_params A1;
    coxian_params A2;
    rw1_params A3;
    uniform_params A4;
    collatz_params A5;
  } chainParamsUnionBlock;

  int largestOf = max_of(sizeof(erlang_params), sizeof(coxian_params));
  largestOf = max_of(largestOf, sizeof(rw1_params));
  largestOf = max_of(largestOf, sizeof(uniform_params));
  largestOf = max_of(largestOf, sizeof(collatz_params));

  if (sizeof(chainParamsUnionBlock) != largestOf) {
    printf("assertion error; size of union block\n");
    return 1;
  }

  void* theJumpParameters = &chainParamsUnionBlock;

  if (strcmp(chainName, "Erlang") == 0 || strcmp(chainName, "erlang") == 0) {
    if (numChainParameters != 2)
      return 1;
    const int k = (int) floor(strtod(strParams[0], NULL));
    const double lambda = strtod(strParams[1], NULL);
    if (k <= 0 || lambda <= 0.0)
      return 1;
    ((erlang_params*)theJumpParameters)->k = k;
    ((erlang_params*)theJumpParameters)->lambda = lambda;
    theJumpFunction = &erlang_jump_function;
  }

  if (strcmp(chainName, "RW1") == 0 || strcmp(chainName, "rw1") == 0) {
    if (numChainParameters != 3)
      return 1;
    const int B = (int) floor(strtod(strParams[0], NULL));
    const double lambda_dec = strtod(strParams[1], NULL);
    const double lambda_inc = strtod(strParams[2], NULL);
    if (B <= 0 || lambda_dec <= 0.0 || lambda_inc <= 0.0)
      return 1;
    ((rw1_params*)theJumpParameters)->B = B;
    ((rw1_params*)theJumpParameters)->lambda_down = lambda_dec;
    ((rw1_params*)theJumpParameters)->lambda_up = lambda_inc;
    theJumpFunction = &rw1_jump_function;
  }

  if (strcmp(chainName, "Coxian") == 0 || strcmp(chainName, "coxian") == 0) {
    if (numChainParameters != 3)
      return 1;
    const int k = (int) floor(strtod(strParams[0], NULL));
    const double lambda = strtod(strParams[1], NULL);
    const double alpha = strtod(strParams[2], NULL);
    if (k <= 0 || lambda <= 0.0 || alpha <= 0.0 || alpha >= 1.0)
      return 1;
    ((coxian_params*)theJumpParameters)->k = k;
    ((coxian_params*)theJumpParameters)->lambda = lambda;
    ((coxian_params*)theJumpParameters)->alpha = alpha;
    theJumpFunction = &coxian_jump_function;
  }

  if (strcmp(chainName, "Uniform") == 0 || strcmp(chainName, "uniform") == 0) {
    if (numChainParameters != 2)
      return 1;
    const int N = (int) floor(strtod(strParams[0], NULL));
    const double lambda = strtod(strParams[1], NULL);
    if (N <= 0 || lambda <= 0.0)
      return 1;
    ((uniform_params*)theJumpParameters)->N = N;
    ((uniform_params*)theJumpParameters)->lambda = lambda;
    theJumpFunction = &uniform_jump_function;
  }

  if (strcmp(chainName, "Collatz") == 0 || strcmp(chainName, "collatz") == 0) {
    if (numChainParameters != 5)
      return 1;
    const int N = (int) floor(strtod(strParams[0], NULL));
    const double lambda_init = strtod(strParams[1], NULL);
    const double lambda_absorb = strtod(strParams[2], NULL);
    const double lambda_even = strtod(strParams[3], NULL);
    const double lambda_odd = strtod(strParams[4], NULL);
    if (N <= 0 || 
        lambda_init <= 0.0 || 
        lambda_absorb <= 0.0 ||
        lambda_even <= 0.0 || 
        lambda_odd <= 0.0)
      return 1;
    ((collatz_params*)theJumpParameters)->N = N;
    ((collatz_params*)theJumpParameters)->lambda_init = lambda_init;
    ((collatz_params*)theJumpParameters)->lambda_absorb = lambda_absorb;
    ((collatz_params*)theJumpParameters)->lambda_even = lambda_even;
    ((collatz_params*)theJumpParameters)->lambda_odd = lambda_odd;
    theJumpFunction = &collatz_jump_function;
  }

  FILE *pfile = NULL;

  if (theJumpFunction == NULL) {
    printf("Did not recognize chain-name.\n");
    return 1;
  }

  uint64_t seed = 0x1234abcd4321efab;
  xoshiro256plus_state local_rng;
  xoshiro256plus_init(&local_rng, seed);

  double mean_t = 0.0;
  double mean_tsq = 0.0;
  double mean_tcu = 0.0;
  double min_t = DBL_MAX;
  double max_t = 0.0;

  if (writeFile) {
    pfile = fopen(textfileName, "w");
    if (!pfile) {
      printf("Failed to open file \"%s\" for writing.\n", textfileName);
      return 1;
    }
  }

  int status = -1;

  for (int64_t i = 0; i < numSamples; i++) {
    int x = 0;
    double t = 0.0;

    if (trajectMode)
      printf("(x, t) = (%i, %f)\n", x, t);

    for (;;) {
      int x1;
      double dt;
      status = next_ctmc_state(theJumpFunction, 
                               theJumpParameters, 
                               &local_rng, 
                               theBufferSize,
                               x, 
                               t, 
                               &x1, 
                               &dt);
      if (status == 0 || status == -1)
        break;
      if (status == 1) {
        x = x1;
        t += dt;
      }
      if (trajectMode)
        printf("(%i, %f)\n", x, t);
    }

    if (status == -1) {
      printf("sampling failed.\n");
      break;
    }

    if (summaryMode) {
      const double c1 = ((double) i) / (i + 1);
      const double c2 = 1.0 / (i + 1);
      mean_t = c1 * mean_t + c2 * t;
      const double tt = t * t;
      mean_tsq = c1 * mean_tsq + c2 * tt;
      mean_tcu = c1 * mean_tcu + c2 * (tt * t);
      if (t < min_t) min_t = t;
      if (t > max_t) max_t = t;
    }

    if (writeFile) {
      fprintf(pfile, "%.15e\n", t);
    }

    if (verboseMode) {
      printf("[%06lli] (x = %i)\tt = %.10f \n", i, x, t);
    }
  }

  if (writeFile) {
    fclose(pfile);
  }

  if (summaryMode && status == 0) {
    const double stdev_t_sq = mean_tsq - mean_t * mean_t;
    const double stdev_t = sqrt(stdev_t_sq);
    const double skew_t = (mean_tcu - mean_t * (3.0 * stdev_t_sq + mean_t * mean_t)) / (stdev_t_sq * stdev_t);

    printf(" mean = %.8f\n", mean_t);
    printf("stdev = %.8f\n", stdev_t);
    printf("c.o.v = %.8f\n", stdev_t / mean_t);  // coef. of variation
    printf(" skew = %.8f\n", skew_t);
    printf("  min = %.8f\n", min_t);
    printf("  max = %.8f\n", max_t);
  }

  return 0;
}

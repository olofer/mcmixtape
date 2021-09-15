/*
 *  Simulator of Kelly style repeated betting processes.
 *  Uses (optionally) OpenMP (for fun) to draw trajectories in parallel.
 *
 *  The following process is being simulated for k = 0 ... K (with W(0) > 0):
 *   
 *    W(k + 1) = (1 + b * f) * W(k), with prob. p
 *    W(k + 1) = (1 - a * f) * W(k), with prob. q = 1 - p
 *
 *  Kelly betting: f(kelly) = p / a - q / b 
 *
 *  In logspace X = log(W) this is instead:
 *    
 *    X(k + 1) = log(1 + b * f) + X(k), prob. p
 *    X(k + 1) = log(1 - a * f) + X(k), prob. q
 *
 *  The sampled results are compared to a binomial recombining tree style
 *  valuation (lattice) of the expected value and standard deviation.
 *
 *  BUILD:
 *    gcc -O2 -Wall -o kelly.exe kelly.c -fopenmp
 *
 *  RUN:
 *    kelly a b f p W K N [nthreads] 
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

#include "xoshiro1n2plus.h"

/* bucket with values to pass around */
typedef struct kellyprocess {
  double a;
  double b;
  double f;
  double p;
  double W;
  int K;
} kellyprocess;

double calc_growth_rate(const kellyprocess* params, double f) {
  const double uval = log(1.0 + params->b * f);
  const double dval = log(1.0 - params->a * f);
  const double p = params->p;
  const double q = 1 - p;
  return (p * uval + q * dval);
}

/* arithmetic sum of k = 0..K */
int arith_sum(int K) {
  return ((K * (K + 1)) >> 1);
}

/* k = 0..K, s = 0..k required */
int array_index(int k, int s) {
  if (k < 0) return 0;
  return (arith_sum(k) + s);
}

/* linear array index or time step k, # successes s = 0..k
   is given by array_index(k, s)

   Number of elements required for the array covering K time-steps is:
     1 + array_index(K, K)
*/
bool test_array_index(int K) {
  int q = 0;
  for (int k = 0; k <= K; k++) {
    for (int s = 0; s <= k; s++) {
      const int idx = array_index(k, s);
      // printf("(k=%i, s=%i): idx = %i, q = %i\n", k, s, idx, q);
      if (idx != q) return false;
      q++;
    }
  }
  return (q == 1 + array_index(K, K));
}

/* allocate value lattice for size K and set terminal values as a function of s
   then do backward-time propagation; when element 0 is reached return its value;
   this works for any moment of the value.
*/
double lattice_evaluation(const kellyprocess* params, int powexp) {
  const double uval = log(1.0 + params->b * params->f);
  const double dval = log(1.0 - params->a * params->f);
  const double val0 = log(params->W);
  const double p = params->p;
  const int K = params->K;
  const int bufsz = 1 + array_index(K, K);
  double* X = malloc(sizeof(double) * bufsz);
  for (int s = 0; s <= K; s++) {
    const double XKs = val0 + s * uval + (K - s) * dval;
    X[array_index(K, s)] = pow(XKs, powexp);       /* initialize terminal value */
  }
  for (int k = K - 1; k >= 0; k--) {  /* backward recursion */
    for (int s = 0; s <= k; s++) {
      X[array_index(k, s)] = p * X[array_index(k + 1, s + 1)] + (1.0 - p) * X[array_index(k + 1, s)];
    }
  }
  const double value = X[array_index(0, 0)];
  free(X); 
  return value;
}

/* Draw random path for K steps; return final log-value */
double one_path_sample(xoshiro256plus_state* xst, const kellyprocess* params) {
  const double uval = log(1.0 + params->b * params->f);
  const double dval = log(1.0 - params->a * params->f);
  const double val0 = log(params->W);
  const double p = params->p;
  const int K = params->K;
  double X = val0;
  for (int k = 0; k < K; k++) {
    const double r = xoshiro256plus_next_double(xst);
    X += (r < p ? uval : dval);
  }
  return X;
}

/* program entry point */
int main(int argc, const char** argv)
{
  if (argc != 8 && argc != 9) {
    printf("USAGE: %s a b f p W K N [nthread]\n", argv[0]);
    return 0;
  }

  kellyprocess params;
  memset(&params, 0, sizeof(kellyprocess));

  params.a = strtod(argv[1], NULL);
  params.b = strtod(argv[2], NULL);
  params.f = strtod(argv[3], NULL);
  params.p = strtod(argv[4], NULL);
  params.W = strtod(argv[5], NULL);
  params.K = (int) floor(strtod(argv[6], NULL));

  const int N = (int) floor(strtod(argv[7], NULL));
  int nthread = 0;
  if (argc == 9) {
    nthread = (int) floor(strtod(argv[8], NULL));
  }

  if (params.b < 0.0 || 
      params.a < 0.0 || 
      params.W <= 0.0 || 
      params.K <= 0 || 
      params.p < 0 || 
      1.0 - params.p < 0 ||
      params.a * params.f >= 1.0 ||
      params.f < 0 || 
      N <= 0)
  {
    printf("(please check input parameters)\n");
    return -1;
  }

  const bool array_index_test = test_array_index(params.K);
  if (!array_index_test) {
    printf("(lattice indexing test failed)\n");
    return -1; 
  }

  const double fkelly = params.p / params.a - (1.0 - params.p) / params.b;

  printf("using f     = %f\n", params.f);
  printf("rate        = %.6e\n", calc_growth_rate(&params, params.f));
  printf("f(Kelly)    = %f\n", fkelly);
  printf("rate(Kelly) = %.6e\n", calc_growth_rate(&params, fkelly));

  printf("K = %i steps\n", params.K);

  const double EX = lattice_evaluation(&params, 1);
  const double EX2 = lattice_evaluation(&params, 2);
  printf("(lattice) E[X]   = %.6e\n", EX);
  printf("(lattice) E[X*X] = %.6e\n", EX2);
  printf("(lattice) std(X) = %.6e\n", sqrt(EX2 - EX * EX));
  printf("(lattice) rate   = %.6e\n", (EX - log(params.W)) / params.K);

  const int maxthreads = omp_get_max_threads();
  nthread = (nthread < 1 ? 1 : (nthread > maxthreads ? maxthreads : nthread));
  uint64_t seed = 0x1234abcd4321efab;
  omp_set_num_threads(nthread);

  printf("drawing N = %i samples / thread (%ix)\n", N, nthread);

  double omp_EX = 0.0;
  double omp_EX2 = 0.0;

  #pragma omp parallel
  {
    xoshiro256plus_state local_xrng;
    xoshiro256plus_init(&local_xrng, seed);

    const int tid = omp_get_thread_num();
    const int tnum = omp_get_num_threads();

    if (tid == 0 && tnum != nthread) {
      printf("requested (%i) != nthread (%i)\n", tnum, nthread);
    }

    for (int j = 0; j < tid; j++) {
      xoshiro256plus_long_jump(&local_xrng);
    }

    double sumx = 0.0;
    double sumx2 = 0.0;
    for (int i = 0; i < N; i++) {
      const double x = one_path_sample(&local_xrng, &params);
      sumx  += x;
      sumx2 += x * x;
    }

    const double local_EX = sumx / N;
    const double local_EX2 = sumx2 / N;

    #pragma omp atomic
    omp_EX += local_EX / tnum;

    #pragma omp atomic
    omp_EX2 += local_EX2 / tnum;
  }

  printf("(sample)  E[X]   = %.6e\n", omp_EX);
  printf("(sample)  E[X*X] = %.6e\n", omp_EX2);
  printf("(sample)  std(X) = %.6e\n", sqrt(omp_EX2 - omp_EX * omp_EX));
  printf("(sample)  rate   = %.6e\n", (omp_EX - log(params.W)) / params.K);

  return 0;
}

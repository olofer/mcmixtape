/*
 * Fundamental patterns for Random Walk Metropolis.
 * Using normal variates as move proposal vector.
 * Walks based on a general (non-normalized) log-likelihood function.
 *
 * Test application samples from a Laplace density (which has known properties).
 * Only run a single MCMC chain.
 *
 * BUILD: gcc -Wall -O2 -o metropolis.exe metropolis.c -lm
 * EXECUTE: ./metropolis.exe 5 500000
 * CHECK: python3 metropolis.py
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "xoshiro1n2plus.h"
#include "ppnd16.h"

// Provide log-prob for current position, and log-prob for the proposed move.
// Return true if the move is accepted, otherwise false.
bool accept_move(
    xoshiro256plus_state *rng,
    double logpx,
    double logpxstar)
{
  if (logpxstar >= logpx)
    return true;
  const double alfa = exp(logpxstar - logpx);
  const double unif = xoshiro256plus_next_double(rng);
  return (unif < alfa);
}

typedef double (*loglikl_func_ptr)(int, const double *, const void *);

// Basic MCMC sampler; take starting sample xinit as input, and a fixed proposal vector parameter lambda
// Return the number of proposal moves that were accepted.
// If X != NULL, save every sample (requires storage nx*nsamples).
// If xfinal != NULL, store the last walk position (nx storage).
// If F != NULL, store log-likelihood values for the chain (nsamples storage).
int random_walk_metropolis(
    xoshiro256plus_state *rng,
    int nx,
    const double *xinit,
    loglikl_func_ptr llikl,
    const double *lambda,
    int nsamples,
    int iprint,
    double *F,
    double *X,
    double *xfinal)
{
  const double sqrtnx = sqrt((double)nx);
  double gamm[nx];
  for (int j = 0; j < nx; j++)
    gamm[j] = lambda[j] / sqrtnx;
  int num_accepted = 0;
  double x[nx];
  double xstar[nx];
  memcpy(x, xinit, nx * sizeof(double));
  double fx = (*llikl)(nx, x, NULL);
  for (int i = 0; i < nsamples; i++)
  {
    if ((iprint > 0) && (i % iprint == 0) && (i > 0))
      printf("[%06i]: f=%f, accpt=%.3f\n", i, fx, ((double)num_accepted) / i);
    if (F != NULL)
      F[i] = fx;
    if (X != NULL)
      memcpy(&X[i * nx], x, nx * sizeof(double));
    // draw proposal moves from N(0, lambda^2 / nx)
    for (int j = 0; j < nx; j++)
    {
      const double uj = xoshiro256plus_next_double(rng);
      xstar[j] = x[j] + gamm[j] * ppnd16(uj, NULL);
    }
    // Evaluate proposed move
    double fxstar = (*llikl)(nx, xstar, NULL);
    if (accept_move(rng, fx, fxstar))
    {
      fx = fxstar;
      memcpy(x, xstar, nx * sizeof(double));
      num_accepted++;
    }
  }
  if (xfinal != NULL)
    memcpy(xfinal, x, nx * sizeof(double));
  return num_accepted;
}

// nx-dimensional Laplace density with an offset (1,1,1...)
// and the spread parameter b = 1.0, for each dimension
double llikl_laplace(int nx, const double *x, const void *aux)
{
  double sum = 0.0;
  for (int i = 0; i < nx; i++)
    sum -= fabs(x[i] - 1.0);
  return sum;
}

// For the above specific hardcoded log-likelihood function:
//
// The sample file written below should have column means and medians = 1.
// It should also have column stdev = sqrt(2), and the MAD should be ln(2).
// And the columns should be uncorrelated.
//

int main(int argc, const char **argv)
{
  const uint64_t seed = 0x1234abcd4321efab;
  xoshiro256plus_state local_rng;
  xoshiro256plus_init(&local_rng, seed);

  if (argc != 3)
  {
    printf("USAGE: %s <dims> <samples>\n", argv[0]);
    return 1;
  }

  const int32_t numDims = (int32_t)floor(strtod(argv[1], NULL));
  const int64_t numSamples = (int64_t)floor(strtod(argv[2], NULL));

  if (numDims <= 0 || numSamples <= 0)
  {
    return 1;
  }

  const int nx = numDims;
  double x[nx];
  double lambda[nx];

  for (int i = 0; i < nx; i++)
  {
    x[i] = 2.0 * xoshiro256plus_next_double(&local_rng) - 1.0;
    lambda[i] = 2.0;
  }

  printf("=== burn-in ===\n");

  random_walk_metropolis(&local_rng, nx, x, llikl_laplace, lambda, 1000, 250, NULL, NULL, x);

  double *X = malloc(nx * numSamples * sizeof(double));

  if (X == NULL)
  {
    printf("Failed to allocate space for all samples requested.\n");
    return 1;
  }

  printf("=== running MCMC (%li samples) ===\n", numSamples);

  random_walk_metropolis(&local_rng, nx, x, llikl_laplace, lambda, numSamples, 2500, NULL, X, x);

  const char textfileName[] = "metropolis.txt";
  FILE *pfile = fopen(textfileName, "w");
  if (pfile != NULL)
  {
    int k = 0;
    for (int64_t i = 0; i < numSamples; i++)
    {
      for (int j = 0; j < nx; j++)
      {
        fprintf(pfile, " %.15e", X[k++]);
      }
      fprintf(pfile, "\n");
    }
    fclose(pfile);
    printf("Wrote %li dim-%i samples to file \"%s\".\n", numSamples, nx, textfileName);
  }
  else
  {
    printf("Failed to open file \"%s\" for writing.\n", textfileName);
  }

  free(X);

  return 0;
}

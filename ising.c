/*
  ising.c - sampling the classic 2D periodic square lattice Ising model

  DESCRIPTION:
    Configuration energy E(x) = -0.5 * sum_{i,j} J_{i,j}*x_i*x_j - H * sum_i x_i
    beta = 1 / (k_B * T); k_B = Boltzmann's constant, T = temperature [K]

    Let T[nondim] = k_B * T[Kelvin] / |J|;
    then sufficient to have J = 1, -1 and beta = 1 / T[nondim]

  USAGE:
    ising J H size sampler numwarm numstat nthread beta1 [beta2 ...]

  ARGUMENTS:
    J = scalar coupling factor, e.g. +1.0, -1.0
    H = scalar external field
    size = width, height of domain, e.g 20
    sampler = gibbs, metropolis
    nthread = # OpenMP workers to request (replications of the schedule)
    numwarm = initial # samples to discard for each beta
    numstat = # samples to get statistics from for each beta
    beta1 ... = sequence/schedule of inverse temperatures to simulate

  OUTPUT:
    CSV files (one per thread) with the statistics for the beta schedule

  BUILD:
    gcc -o ising.exe -Wall -O2 ising.c -fopenmp

  REFERENCES:
    David Chandler, "Introduction to Modern Statistical Mechanics", OUP 1987
    David J.C. Mackay, "Information Theory, Inference, and Learning Algorithms", CUP 2003

  EXAMPLE:
    Calculation similar to Mackay figure 31.5: see R-program 'ising-example.R'

 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

#include "xoshiro1n2plus.h"

int textio_write_coldata_csv(
  const char *filename,
  int numcols,
  const char **colnames,
  const double **coldata,
  int numrows,
  const char *formatspec);

void init_ising_state(
  int8_t* x,
  int size,
  xoshiro256plus_state* xst)
{
  const int N = size * size;
  for (int i = 0; i < N; i++)
    x[i] = (xoshiro256plus_next_double(xst) < 0.5 ? -1 : 1);
}

void next_ising_state(
  int8_t* x,
  int size,
  double J,
  double H,
  double beta,
  char sampler,
  double* deltaE,
  double* deltaM,
  xoshiro256plus_state* xst)
{
  const int i = (int) (xoshiro256plus_next_double(xst) * size);
  const int j = (int) (xoshiro256plus_next_double(xst) * size);
  const int deci = (i == 0 ? size - 1 : i - 1); // wrap around neighbours
  const int inci = (i == size - 1 ? 0 : i + 1);
  const int decj = (j == 0 ? size - 1 : j - 1);
  const int incj = (j == size - 1 ? 0 : j + 1);
  const int idx = i + j * size;
  const double B = J * (x[deci + j * size] + 
                        x[inci + j * size] + 
                        x[i + decj * size] + 
                        x[i + incj * size]) + H;
  const int8_t currx = x[idx];
  const double dE = 2.0 * currx * B;
  if (sampler == 'G') {
    const double P1 = 1.0 / (1.0 + exp(-2.0 * B * beta)); // Pr[x = +1]
    x[idx] = (xoshiro256plus_next_double(xst) < P1 ? +1 : -1);
  } else {
    const double Paccept = (dE <= 0.0 ? 1.0 : exp(-1.0 * beta * dE));  // Pr[flip]
    if (xoshiro256plus_next_double(xst) < Paccept) {
      x[idx] = -currx;
    }
  }
  const bool has_flipped = (currx != x[idx]);
  if (deltaE != NULL)
    *deltaE = (has_flipped ? dE : 0.0);
  if (deltaM != NULL)
    *deltaM = (has_flipped ? -2.0 * currx / (size * size) : 0.0);
}

double ising_magnetization(const int8_t* x, int size)
{
  const int N = size * size;
  double sum = 0.0;
  for (int i = 0; i < N; i++)
    sum += x[i];
  return sum / N;
}

double ising_energy(const int8_t* x, int size, double J, double H) {
  double E1 = 0.0;
  double E2 = 0.0;
  for (int i = 0; i < size; i++) {
    const int deci = (i == 0 ? size - 1 : i - 1);
    const int inci = (i == size - 1 ? 0 : i + 1);
    for (int j = 0; j < size; j++) {
      const int decj = (j == 0 ? size - 1 : j - 1);
      const int incj = (j == size - 1 ? 0 : j + 1);
      const int idx = i + j * size;
      E1 += H * x[idx];
      E2 += J * x[idx] * (x[inci + j * size] + 
                          x[deci + j * size] + 
                          x[i + decj * size] + 
                          x[i + incj * size]);
    }
  }
  return -1.0 * (0.5 * E2 + E1);
}

int main(int argc, const char** argv)
{
  if (argc < 9) {
    printf("usage: %s J H size sampler numwarm numstat nthread beta1 [beta2 ...]\n", argv[0]);
    return 1;
  }

  const double J = strtod(argv[1], NULL);
  const double H = strtod(argv[2], NULL);
  const int size = (int) floor(strtod(argv[3], NULL));
  char sampler = 0;
  if (strcmp(argv[4], "gibbs") == 0 || 
      strcmp(argv[4], "Gibbs") == 0 || 
      strcmp(argv[4], "g") == 0 ||
      strcmp(argv[4], "G") == 0)
  {
    sampler = 'G';
  } else if (strcmp(argv[4], "metropolis") == 0 ||
             strcmp(argv[4], "Metropolis") == 0 ||
             strcmp(argv[4], "m") == 0 ||
             strcmp(argv[4], "M") == 0)
  {
    sampler = 'M';
  } else {
    printf("Unrecognized sampler (not g nor m)\n");
    return 1;
  }
  const int64_t numwarm = (int64_t) floor(strtod(argv[5], NULL));
  const int64_t numstat = (int64_t) floor(strtod(argv[6], NULL));

  if (size <= 0 || numwarm < 0 || numstat < 0) {
    printf("Please specify positive size and non-negative number of samples\n");
    return 1;
  }

  printf("size N = %i-by-%i = %i; J = %.4f, H = %.4f, sampler = %s\n",
    size, size, size * size, J, H, (sampler == 'G' ? "Gibbs" : "Metropolis"));

  const int nthread = (int) floor(strtod(argv[7], NULL));
  const int maxthread = omp_get_max_threads();
  if (nthread < 1 || nthread > maxthread) {
    printf("requested thread count (%i) invalid (min = 1, max = %i)\n", nthread, maxthread);
    return 1;
  }

  printf("*** nthread =  %i ***\n", nthread);

  const int numbeta = argc - 8;
  double* beta_vec = malloc(sizeof(double) * numbeta);
  printf("*** numbeta =  %i ***\n", numbeta);
  for (int i = 0; i < numbeta; i++) {
    const double betai = strtod(argv[8 + i], NULL);
    beta_vec[i] = betai;
    //printf("%.8e\n", beta_vec[i]);
  }

  double* thread_tally_magn = malloc(nthread * numbeta * sizeof(double));
  double* thread_tally_magn_sq = malloc(nthread * numbeta * sizeof(double));
  double* thread_tally_magn_abs = malloc(nthread * numbeta * sizeof(double));
  double* thread_tally_energy = malloc(nthread * numbeta * sizeof(double));
  double* thread_tally_energy_sq = malloc(nthread * numbeta * sizeof(double));

  uint64_t seed = 0x1234abcd4321efab;

  int8_t* state = malloc(sizeof(int8_t) * size * size * nthread);

  omp_set_num_threads(nthread);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const int tnum = omp_get_num_threads();

    if (tid == 0 && tnum != nthread) {
      printf("did not get the requested (%i) threads (%i)\n", tnum, nthread);
    }

    xoshiro256plus_state local_rng;
    xoshiro256plus_state* this_rng = &local_rng;

    xoshiro256plus_init(this_rng, seed);
    for (int j = 0; j < tid; j++)
      xoshiro256plus_long_jump(this_rng);

    int8_t* this_state = &state[tid * size * size];
    init_ising_state(this_state, size, this_rng);

    for (int i = 0; i < numbeta; i++) {
      /* set new beta */
      const double beta = beta_vec[i];
      if (tid == 0) {
        printf("[tid = %i / tnum = %i]: beta[%i] = %.8e\n", tid, tnum, i, beta_vec[i]);
      }

      /* burn in */
      for (int j = 0; j < numwarm; j++)
        next_ising_state(this_state, size, J, H, beta, sampler, NULL, NULL, this_rng);

      /* these are updated incrementally for efficiency reasons.. */
      double this_magn = ising_magnetization(this_state, size);
      double this_energy = ising_energy(this_state, size, J, H);

      /* running statistics: <magn>, <magn^2>, <|magn|>, <E>, <E^2> */
      double mean_magn = 0.0;
      double mean_magn_sq = 0.0;
      double mean_magn_abs = 0.0;
      double mean_energy = 0.0;
      double mean_energy_sq = 0.0;

      for (int j = 0; j < numstat; j++) {
        double deltaE = 0.0;
        double deltaM = 0.0;
        next_ising_state(this_state, size, J, H, beta, sampler, &deltaE, &deltaM, this_rng);
        //const double this_magn = ising_magnetization(this_state, size);
        this_magn += deltaM;
        mean_magn = ((mean_magn * j) + this_magn) / (j + 1);
        mean_magn_sq = ((mean_magn_sq * j) + this_magn * this_magn) / (j + 1);
        mean_magn_abs = ((mean_magn_abs * j) + fabs(this_magn)) / (j + 1);
        //const double this_energy = ising_energy(this_state, size, J, H);
        this_energy += deltaE;
        mean_energy = ((mean_energy * j) + this_energy) / (j + 1);
        mean_energy_sq = ((mean_energy_sq * j) + this_energy * this_energy) / (j + 1);
      }

      const int tid_ofs = tid * numbeta;
      thread_tally_magn[tid_ofs + i] = mean_magn;
      thread_tally_magn_sq[tid_ofs + i] = mean_magn_sq;
      thread_tally_magn_abs[tid_ofs + i] = mean_magn_abs;
      thread_tally_energy[tid_ofs + i] = mean_energy;
      thread_tally_energy_sq[tid_ofs + i] = mean_energy_sq;

      if (tid == 0) {
        printf("[tid = %i]: <magn^2> = %.8e, <energy>/N = %.8e\n", tid, mean_magn_sq, mean_energy / (size * size));
      }

      /* repeat ... */
    }

  }

  /* finish by writing CSV files (one per thread) */
  for (int tid = 0; tid < nthread; tid++) {
    char csvfilename[64];
    sprintf(csvfilename, "ising_tally_%04i_%04i.csv", size, tid);
    const int tid_ofs = tid * numbeta;
    const int numcols = 6;
    const char *colnames[6] = { "beta",
                                "magn",
                                "magn_sq", 
                                "magn_abs", 
                                "energy", 
                                "energy_sq" };
    const double* coldata[6] = { beta_vec,
                                 &thread_tally_magn[tid_ofs],
                                 &thread_tally_magn_sq[tid_ofs],
                                 &thread_tally_magn_abs[tid_ofs],
                                 &thread_tally_energy[tid_ofs],
                                 &thread_tally_energy_sq[tid_ofs] };
    const int ok = textio_write_coldata_csv(csvfilename,
                                            numcols,
                                            colnames,
                                            coldata,
                                            numbeta,
                                            "%.16e");
    if (ok != 0)
      printf("failed to write: \"%s\"\n", csvfilename);
  }

  free(thread_tally_magn);
  free(thread_tally_magn_sq);
  free(thread_tally_magn_abs);
  free(thread_tally_energy);
  free(thread_tally_energy_sq);

  free(beta_vec);
  free(state);

  return 0;
}

int textio_write_coldata_csv(
  const char *filename,
  int numcols,
  const char **colnames,
  const double **coldata,
  int numrows,
  const char *formatspec)
{
  if (filename == NULL || coldata == NULL) return 1;
  if (numcols <= 0 || numrows < 0) return 2;
  char default_formatspec[] = "%.6e";
  char str1[8];
  char str2[8];
  if (formatspec != NULL) {
    sprintf(str1, "%s, ", formatspec);
    sprintf(str2, "%s\n", formatspec);
  } else {
    sprintf(str1, "%s, ", default_formatspec);
    sprintf(str2, "%s\n", default_formatspec);
  }
  FILE *pfile = NULL;
  pfile = fopen(filename, "w");
  if (!pfile) return 3;
  if (colnames != NULL) {
    for (int j = 0; j < (numcols - 1); j++) {
      fprintf(pfile, "%s, ", colnames[j]);
    }
    fprintf(pfile, "%s\n", colnames[numcols - 1]);
  }
  for (int i = 0; i < numrows; i++) {
    for (int j = 0; j < (numcols - 1); j++) {
      fprintf(pfile, str1, coldata[j][i]);
    }
    fprintf(pfile, str2, coldata[numcols - 1][i]);
  }
  fclose(pfile);
  return 0;
}

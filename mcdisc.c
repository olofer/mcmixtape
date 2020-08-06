/*
    Demonstration of Random Walk evaluation of
    Laplace's equation in a disc:

    Dirichlet problem:
      diff(diff(u, x), x) + diff(diff(u, y), y) = 0, |(x, y)| < a
      u(x, y) = Indicator{theta inside (-w/2, w/2)}, on boundary |(x, y)| = a

    WHAT GOES ON HERE:
      There is a classic integral solution for this problem:

      u(r, theta) = ...
        (1/2*pi)*integral_0^(2*pi) dz * h(z) *...
        {(a^2 - r^2)/(a^2 - 2*cos(theta - z)*a*r + r^2)}

      where h(theta) = Dirichlet boundary value,
      and x = cos(theta) * r, y = sin(theta) * r

      Compare its value with the Brownian motion method
      (special case of the Feynman-Kac equation).

    BUILD:
      gcc -O2 -Wall -o mcdisc.exe mcdisc.c -fopenmp

    RUN:
      mcdisc a w dt kmax threads ns nw x1 y1 [x2 y2 ...]

    a       = radius of disc > 0
    w       = arc (-w/2, w/2) of disc boundary takes value 1, the rest 0
    dt      = timestep for random walk
    kmax    = max number of timesteps (too small will bias results)
    threads = requested number of OMP threads to run when sampling
    ns      = number of standard Monte-Carlo path samples (per thread)
    nw      = number of intervals to use for trapezoidal integral
    x, y    = points to evaluate (must be inside disc)

    EXAMPLE:
      mcdisc 1.0 3.14159265359 .1e-3 1e5 2 1e4 1e6 0.0 0.0 0.5 0.0 -0.5 0.0

 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>  // memset, memcpy
#include <stdbool.h>
#include <omp.h>

#include "xoshiro1n2plus.h"

const double _pi = 3.14159265358979323846;
const double _twopi = 2.0 * _pi;

/* Marsaglia's polar method for random variates */
/* Initilize thread local variables has_spare = 0.0, has_spare = 0, then generate */
double randn_function_safe(
  xoshiro256plus_state* xst,
  double* spare,
  int* has_spare)
{
  if (*has_spare == 1) {
    *has_spare = 0;
    return *spare;
  } else {
    double u, v, s;
    do {
      u = 2.0 * xoshiro256plus_next_double(xst) - 1.0;
      v = 2.0 * xoshiro256plus_next_double(xst) - 1.0;
      s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0*log(s) / s);
    *spare = v * s;
    *has_spare = 1;
    return u * s;
  }
}

double mc_sample_standard_once(
  xoshiro256plus_state* xst,
  double x,
  double y,
  double a,
  double w,
  double dt,
  int kmax,
  int* steps)
{
  double spare = 0.0;
  int has_spare = 0;
  const double r2max = a * a;
  const double sqrtdt = sqrt(dt);
  int k = 0; 
  double r2 = x*x + y*y;
  while (k < kmax && r2 < r2max) {
    x += sqrtdt * randn_function_safe(xst, &spare, &has_spare);
    y += sqrtdt * randn_function_safe(xst, &spare, &has_spare);
    r2 = x*x + y*y;
    k++;
  }
  if (steps != NULL) *steps = k;
  double theta = atan2(y, x);
  return (fabs(theta) < w / 2.0 ? 1.0 : 0.0); // boundary indicator function
}

/* MC path sampler; draw n samples, optionally put results in n-array u */
void mc_sample_standard_many(
  xoshiro256plus_state* xst,
  int num_rngs,  // xst[0] .. xst[num_rngs - 1]
  double x,
  double y,
  double a,
  double w,
  double dt,
  int kmax,
  int n,
  double* u,
  int* steps,
  double* out_usum,
  int* out_nsum,
  bool print_rng_state)
{
  double usumtot = 0.0;
  int nsumtot = 0;

  #pragma omp parallel 
  {
    const int tid = omp_get_thread_num();
    const int tnum = omp_get_num_threads();
    if (tid == 0 && tnum != num_rngs) {
      printf("omp_get_num_threads() = %i != requested = %i\n", tnum, num_rngs);
    }

    xoshiro256plus_state* thread_xst = &xst[tid];
    
    if (print_rng_state) {
      printf("B:tid = %i (%i) | {%016lx %016lx %016lx %016lx}\n", 
        tid, tnum, thread_xst->s[0], thread_xst->s[1], thread_xst->s[2], thread_xst->s[3]);
    }

    int stepsi = -1;
    double usum = 0.0;
    int nsum = 0;

    for (int i = tid; i < n; i += tnum) {
      const double ui = mc_sample_standard_once(thread_xst, x, y, a, w, dt, kmax, &stepsi);
      if (stepsi > 0 && stepsi < kmax) {
        usum += ui;
        nsum ++;
      }
      if (steps != NULL) steps[i] = stepsi;
      if (u != NULL) u[i] = ui;
    }

    if (print_rng_state) {
      printf("E:tid = %i (%i) | {%016lx %016lx %016lx %016lx}\n", 
        tid, tnum, thread_xst->s[0], thread_xst->s[1], thread_xst->s[2], thread_xst->s[3]);
    }

    #pragma omp critical
    {
      usumtot += usum;
      nsumtot += nsum;
    }
  }

  if (out_usum != NULL) *out_usum = usumtot;
  if (out_nsum != NULL) *out_nsum = nsumtot;
}

double boundary_integrand(double a, double r, double theta, double z, double hz) {
  const double A = hz * (a*a - r*r);
  const double B = a*a + r*r - 2.0*cos(theta - z)*a*r;
  return A/B;
}

/* calculate the deterministic integral result for u(x, y);
   n intervals, theta = [-w/2, w/2], radius = a */
double trapezoidal_integral(double a, double w, double x, double y, int n) {
  const double theta = atan2(y, x);
  const double r = sqrt(x*x + y*y);
  const double dtheta = w / n;
  const double hz = 1.0 / _twopi;
  double sum = 0.0;
  sum += boundary_integrand(a, r, theta, -w/2, hz) / 2.0;
  for (int i = 1; i < n; i++) {
    const double z = -w/2 + i*dtheta;
    sum += boundary_integrand(a, r, theta, z, hz);
  }
  sum += boundary_integrand(a, r, theta, w/2, hz) / 2.0;
  return sum * dtheta;
}

/* program entry point */
int main(int argc, const char** argv)
{
  if (argc < 10) {
    printf("USAGE: %s r w dt kmax threads ns nw x1 y1 [x2 y2 ...]\n", argv[0]);
    return 0;
  }

  const double a = strtod(argv[1], NULL);
  const double w = strtod(argv[2], NULL);
  const double dt = strtod(argv[3], NULL);
  const int kmax = (int) floor(strtod(argv[4], NULL));
  const int req_threads = (int) floor(strtod(argv[5], NULL));
  const int ns = (int) floor(strtod(argv[6], NULL));
  const int nw = (int) floor(strtod(argv[7], NULL));

  if (a <= 0.0 || w <= 0.0 || w > _twopi || dt <= 0.0 || kmax <= 0) {
    printf("Incorrect argument(s).\n");
    return 0;
  }

  const int max_threads = omp_get_max_threads();
  printf("omp max threads = %i, requested = %i\n", max_threads, req_threads);

  if (req_threads < 1 || req_threads > max_threads) {
    printf("requested thread count invalid\n");
    return 0;
  }

  uint64_t seed = 0x1234abcd4321efab;

  xoshiro256plus_state* xrng = (xoshiro256plus_state*) malloc(req_threads * sizeof(xoshiro256plus_state));
  for (int i = 0; i < req_threads; i++) {
    if (i == 0) xoshiro256plus_init(&xrng[i], seed);
    if (i > 0) {
      memcpy(&xrng[i], &xrng[i - 1], sizeof(xoshiro256plus_state));
      xoshiro256plus_long_jump(&xrng[i]);
    }
  }

  printf("a = %.6f, w = %.6f\n", a, w);

  const double tmax = kmax * dt;
  printf("tmax = (kmax = %i) * (dt = %e) = %e\n", kmax, dt, tmax);

  const int startidx = 8;
  const int numxy = (argc - startidx) >> 1;

  printf("ns = %i samples/thread, for each of %i coordinates\n", ns, numxy);

  const int Ns = req_threads * ns;

  omp_set_num_threads(req_threads);

  for (int i = 0; i < numxy; i++) {
    const double xi = strtod(argv[startidx + i*2], NULL);
    const double yi = strtod(argv[startidx + i*2 + 1], NULL);

    if (xi*xi + yi*yi >= a*a) {
      printf("[%03i] : (x = %.6f, y = %.6f); skipping since r > a\n", i, xi, yi);
      continue;      
    }

    printf("[%03i] : (x = %.6f, y = %.6f)", i, xi, yi);
    if (nw > 0) {
      const double trapz_value = trapezoidal_integral(a, w, xi, yi, nw);
      printf("; trapz(%i pts) = %.6f", nw, trapz_value);
    }
    printf("\n");

    double usumi = -1.0;
    int nsumi = -1;

    mc_sample_standard_many(xrng, req_threads, xi, yi, a, w, dt, kmax, Ns, NULL, NULL, &usumi, &nsumi, false);

    const double ui = usumi / nsumi;
    const double sderr_alt = sqrt(ui*(1.0 - ui) / nsumi);

    printf("   u(x, y) ~ %.6f (se = %.6f); samples = %i/%i\n", ui, sderr_alt, nsumi, Ns);
  }

  free(xrng);

  return 0;
}

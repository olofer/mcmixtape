/* 
  trinomial.c

  Options valuation using a trinomial recombining tree.
  Using either a forward or a backward pass (both compared).
  Path sampling used as alternative demonstration.
  Compare with Black-Scholes-Merton when possible.
  Terminal value, value rate, and hazard rate supported.
  Also generates standard deviations using the trinomial tree.

  USAGE: trinomial mu sigma dt S0 Tmax q K option [nsamples nthreads sampler]
         trinomial (without any argument: run a few tests)

  Demonstration implementations of the classics (see EXAMPLES below):
    European Call (has a closed form solution for comparison)
    European Put  (also has ref. solution)
    American Call 
    American Put  

  NOTE:
  In the case of a geometric Brownian motion,
    dY = mu * Y * dt + sigma * Y * dW,
  and we want the path of X = log(Y), then by Ito's lemma,
    dX = (mu - sigma * sigma / 2) * dt + sigma * dW
  (this is used for the demo calculations below).

  Specifically x0 = log(S0)

  This code only uses constant mu, constant sigma processes to keep
  things simple; and this also allows an automatic choice of a good grid 
  spacing "delta" internally; given a requested timestep "dt".
  (the automatic delta will match the first three non-central moments
   for the drift-diffusion).

  These value functions f, h, F are internally function 
  pointers/callbacks on the lattice coordinates (t, s).
  K is the strike price of the option (parameter for terminal F).
  q is the continuous yield of the option (absorbed into tree probs.).
  The risk-free rate r will be taken as the drift parameter mu here.
  Recognized option arguments:
    'european-call', 'european-put', 'american-call', 'american-put'
    (also: 'european-test', and 'american-test')
  Recognized sampler arguments:
    'immortal', 'mortal'

  BUILD:
    gcc -O2 -Wall -o trinomial.exe trinomial.c -fopenmp

  EXAMPLES:
  { Comparison: http://www.math.columbia.edu/~smirnov/options13.html }
    trinomial 0.1 0.3 1e-3 100.0 1.0 0.03 95 european-call
    trinomial 0.1 0.3 1e-3 100.0 1.0 0.03 95 european-put 1e5 4 mortal
    trinomial 0.1 0.3 1e-3 100.0 1.0 0.03 95 american-call
    trinomial 0.1 0.3 1e-3 100.0 1.0 0.03 95 american-put
    trinomial 0.1 0.3 1e-3 100.0 1.0 0.03 95 american-put 1e5 4 mortal

 */

// ISSUE: the trinomial calc. of the stdev is incorrect if f != 0
// (applies to the test case, not the standard options)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

#include "ppnd16.h"
#include "xoshiro1n2plus.h"

/* ------------------------------------------------------------------------- */

typedef double (*func_ptr_value)(double, double, const void*); // y = func(t, x, data)
typedef double (*func_ptr_transform)(double); // y = func(x)

typedef struct value_parameters {
  func_ptr_value F; // terminal value
  func_ptr_value f; // value rate
  func_ptr_value h; // hazard rate
  const void* data_ptr;
} value_parameters;

typedef struct process_parameters {
  double mu;
  double sigma;
  double xnaught;
  double dt;
  double delta;  // automatically assigned (based on dt)
  double pu;
  double pd;
  double pc;
} process_parameters;

bool assign_delta_t(process_parameters* P, double dt) {
  if (dt <= 0.0 || P->sigma <= 0.0) return false;
  P->dt = dt;
  const double ep = P->mu / P->sigma;
  P->delta = P->sigma * sqrt(dt) * sqrt(3.0 + dt * ep * ep);
  const double B = (P->mu * dt) / P->delta;
  const double A = (1.0 + dt * ep * ep) / (3.0 + dt * ep * ep);
  P->pu = 0.5 * (B + A);
  P->pd = 0.5 * (-B + A);
  P->pc = 1.0 - P->pu - P->pd;
  return true;
}

/* ------------------------------------------------------------------------- */

/* t = 0..T, s = -t .. t required; array index is 0-based; no bounds checking */
int array_index(int t, int s) {
  return (t * (t + 1) + s);
}

// The above index function is used to access any element in the lattice
// specifically the 3 allowed jumps are (t, s) -> {(t + 1, s - 1), (t + 1, s), (t + 1, s + 1)}

bool test_array_index(int T) {
  int q = 0;
  for (int t = 0; t <= T; t++) {
    for (int s = -t; s <= t; s++) {
      const int idx = array_index(t, s);
      if (q != idx) return false;
      q++;
    }
  }
  return (q == array_index(T, T) + 1);
}

/* ------------------------------------------------------------------------- */

double zero_function(double t, double x, const void* data) {
  return 0.0;
}

double identity_transform(double a) {
  return a;
}

double square_transform(double a) {
  return a * a;
}

/* ------------------------------------------------------------------------- */

double backward_valuation(
  const process_parameters* P, 
  const value_parameters* V,
  double* A,
  int Nstep,
  func_ptr_transform g)
{
  if (g == NULL) g = &identity_transform;
  func_ptr_value F = (V->F == NULL ? &zero_function : V->F);
  func_ptr_value f = (V->f == NULL ? &zero_function : V->f);
  func_ptr_value h = (V->h == NULL ? &zero_function : V->h);  // h > 0 leads to decay of prob. mass
  for (int s = -Nstep; s <= Nstep; s++) {
    A[array_index(Nstep, s)] = (*g)((*F)(Nstep * P->dt, P->xnaught + s * P->delta, V->data_ptr));
  }
  for (int t = Nstep - 1; t >= 0; t--) {
    for (int s = -t; s <= t; s++) {
      const int idx = array_index(t, s);
      const double this_t = t * P->dt;
      const double this_x = P->xnaught + s * P->delta;
      const double thisSurv = exp(-1.0 * P->dt * (*h)(this_t, this_x, V->data_ptr));
      A[idx] = (P->pu) * A[array_index(t + 1, s + 1)] + 
               (P->pc) * A[array_index(t + 1, s)] +
               (P->pd) * A[array_index(t + 1, s - 1)]; 
      A[idx] += (*g)((*f)(this_t, this_x, V->data_ptr)) * P->dt;
      A[idx] *= thisSurv;
    }
  }
  return A[array_index(0, 0)];
}

double forward_valuation(
  const process_parameters* P,
  const value_parameters* V, 
  double* A,  // will hold the Fokker-Planck solution at exit
  int Nstep,
  func_ptr_transform g)
{
  double value = 0.0;
  if (g == NULL) g = &identity_transform;
  func_ptr_value f = (V->f == NULL ? &zero_function : V->f);
  func_ptr_value h = (V->h == NULL ? &zero_function : V->h);  // h = 0 implies conservation of prob. mass
  func_ptr_value F = (V->F == NULL ? &zero_function : V->F);
  memset(A, 0, sizeof(double) * (1 + array_index(Nstep, Nstep)));
  A[array_index(0, 0)] = 1.0;
  for (int t = 0; t < Nstep; t++) {
    for (int s = -t; s <= t; s++) {
      const int idx = array_index(t, s);
      const double thisProb = A[idx];
      const double this_t = t * P->dt;
      const double this_x = P->xnaught + s * P->delta;
      const double thisSurv = exp(-1.0 * P->dt * (*h)(this_t, this_x, V->data_ptr));
      const double gamma = thisProb * thisSurv;
      A[array_index(t + 1, s + 1)] += gamma * (P->pu);
      A[array_index(t + 1, s)]     += gamma * (P->pc);
      A[array_index(t + 1, s - 1)] += gamma * (P->pd);
      value += gamma * (*g)((*f)(this_t, this_x, V->data_ptr)) * P->dt;
    }
  }
  for (int s = -Nstep; s <= Nstep; s++) { // terminal contribution to value
    value +=  A[array_index(Nstep, s)] * (*g)((*F)(Nstep * P->dt, P->xnaught + s * P->delta, V->data_ptr));
  }
  return value;
}

// Dynamic programming backward evaluation: action = exercise option early, or not.
double backward_valuation_terminatable(
  const process_parameters* P, 
  const value_parameters* V,
  double* A,
  int Nstep,
  bool* X)
{
  func_ptr_value f = (V->f == NULL ? &zero_function : V->f);
  func_ptr_value h = (V->h == NULL ? &zero_function : V->h);  // h > 0 leads to decay of prob. mass
  func_ptr_value F = (V->F == NULL ? &zero_function : V->F);
  for (int s = -Nstep; s <= Nstep; s++) {
    const int idx = array_index(Nstep, s);
    A[idx] = (*F)(Nstep * P->dt, P->xnaught + s * P->delta, V->data_ptr);
    X[idx] = true;
  }
  for (int t = Nstep - 1; t >= 0; t--) {
    for (int s = -t; s <= t; s++) {
      const int idx = array_index(t, s);
      const double this_t = t * P->dt;
      const double this_x = P->xnaught + s * P->delta;
      const double thisSurv = exp(-1.0 * P->dt * (*h)(this_t, this_x, V->data_ptr));
      A[idx] = (P->pu) * A[array_index(t + 1, s + 1)] + 
               (P->pc) * A[array_index(t + 1, s)] +
               (P->pd) * A[array_index(t + 1, s - 1)]; 
      A[idx] += (*f)(this_t, this_x, V->data_ptr) * P->dt;
      A[idx] *= thisSurv;
      const double stopValue = (*F)(this_t, this_x, V->data_ptr);
      if (stopValue > A[idx]) {
        A[idx] = stopValue;
        X[idx] = true;
      } else {
        X[idx] = false;
      }
    }
  }
  return A[array_index(0, 0)];
}

// forward evaluation using the stopping rule X from above function..
double forward_valuation_terminatable(
  const process_parameters* P,
  const value_parameters* V, 
  double* A,
  int Nstep,
  const bool* X,
  func_ptr_transform g)
{
  double value = 0.0;
  if (g == NULL) g = &identity_transform;
  func_ptr_value f = (V->f == NULL ? &zero_function : V->f);
  func_ptr_value h = (V->h == NULL ? &zero_function : V->h);  // h = 0 implies conservation of prob. mass
  func_ptr_value F = (V->F == NULL ? &zero_function : V->F);
  memset(A, 0, sizeof(double) * (1 + array_index(Nstep, Nstep)));
  A[array_index(0, 0)] = 1.0;
  for (int t = 0; t < Nstep; t++) {
    for (int s = -t; s <= t; s++) {
      const int idx = array_index(t, s);
      const double thisProb = A[idx];
      const double this_t = t * P->dt;
      const double this_x = P->xnaught + s * P->delta;
      if (!X[idx]) {
        const double thisSurv = exp(-1.0 * P->dt * (*h)(this_t, this_x, V->data_ptr));
        const double gamma = thisProb * thisSurv;
        A[array_index(t + 1, s + 1)] += gamma * (P->pu);
        A[array_index(t + 1, s)]     += gamma * (P->pc);
        A[array_index(t + 1, s - 1)] += gamma * (P->pd);
        value += gamma * (*g)((*f)(this_t, this_x, V->data_ptr)) * P->dt;
      } else {
        value += thisProb * (*g)((*F)(this_t, this_x, V->data_ptr));
      }
    }
  }
  for (int s = -Nstep; s <= Nstep; s++) { // terminal contribution to value
    value +=  A[array_index(Nstep, s)] * (*g)((*F)(Nstep * P->dt, P->xnaught + s * P->delta, V->data_ptr));
  }
  return value;
}

/* ------------------------------------------------------------------------- */

double normalCDF(double x) {
  const double A = -1.0 / sqrt(2);
  return 0.5 * erfc(A * x);
}

double test_ppnd16_accuracy(xoshiro256plus_state* rng, int n) {
  double maxerr = 0.0;
  printf("maximum error |p - normalCDF(ppnd16(p))| over %.1fM samples p: ", n / 1.0e6);
  for (int k = 0; k < n; k++) {
    const double p = xoshiro256plus_next_double(rng);
    double errk = p - normalCDF(ppnd16(p, NULL));
    if (fabs(errk) > maxerr)
      maxerr = fabs(errk);
  }
  printf("%e\n", maxerr);
  return maxerr;
}

// draw n variates ~N(0,1); collect first 5 moments; compare to expected results (printf)
void test_ppnd16_samples(xoshiro256plus_state* rng, int n) {
  printf("drawing %.1fM samples X ~ N(0,1) using ppnd16(.) ...\n", n / 1.0e6);
  double x[6];
  memset(x, 0, 6 * sizeof(double));
  for (int k = 0; k < n; k++) {
    const double a = 1.0 / (k + 1);
    const double b = ((double) k) / (k + 1);
    const double w = ppnd16(xoshiro256plus_next_double(rng), NULL);
    x[0] = a * 1.0 + b * x[0];
    x[1] = a * w + b * x[1];
    const double w2 = w * w;
    x[2] = a * w2 + b * x[2];
    const double w3 = w2 * w;
    x[3] = a * w3 + b * x[3];
    x[4] = a * (w2 * w2) + b * x[4];
    x[5] = a * (w3 * w2) + b * x[5];
  }
  double xref[6] = {1.0, 0.0, 1.0, 0.0, 3.0, 0.0};
  for (int i = 0; i < 6; i++) {
    printf("sample E[X^%i] = %.8f\t(theory = %.1f)\n", i, x[i], xref[i]);
  }
}

/* ------------------------------------------------------------------------- */

bool fetch_decision(int t, double x, double delta, const bool* X) {
  int s = (int) (x / delta);  // round toward zero
  if (s < -t) s = -t;
  if (s > t) s = t;
  return X[array_index(t, s)];
}

// Basic Euler-Maruyama time-steppers (inaccurate unless dt is small)

typedef double (*path_func_ptr)(xoshiro256plus_state*, 
                                const process_parameters*, 
                                const value_parameters*, 
                                int,
                                const bool*);

double one_immortal_path_sample(
  xoshiro256plus_state* rng,
  const process_parameters* P,
  const value_parameters* V, 
  int Nstep,
  const bool* stoppingRule)
{
  func_ptr_value f = (V->f == NULL ? &zero_function : V->f);
  func_ptr_value h = (V->h == NULL ? &zero_function : V->h);
  func_ptr_value F = (V->F == NULL ? &zero_function : V->F);
  const bool hasStoppingArray = (stoppingRule != NULL);
  const double mu = P->mu;
  const double sigma = P->sigma;
  const double dt = P->dt;
  const double sqrtdt = sqrt(dt);
  double value = 0.0;
  double cumuh = 0.0;
  double x = P->xnaught;
  double t = 0.0;
  int k = 0;
  for (;;) {
    if (k == Nstep) break;
    if (hasStoppingArray && fetch_decision(k, x - P->xnaught, P->delta, stoppingRule)) break;
    const double this_h = (*h)(t, x, V->data_ptr);
    cumuh += dt * this_h;
    const double this_f = (*f)(t, x, V->data_ptr);
    value += exp(-1.0 * cumuh) * this_f * dt;
    const double r = xoshiro256plus_next_double(rng);
    const double w = ppnd16(r, NULL);
    x += mu * dt + sigma * sqrtdt * w;
    t += dt;
    k ++;
  }
  const double this_F = (*F)(t, x, V->data_ptr);
  value += exp(-1.0 * cumuh) * this_F;
  return value;
}

// The hazard function provides a probability of early death at each time-step.
// The probability of death implies discounting.
double one_mortal_path_sample(
  xoshiro256plus_state* rng,
  const process_parameters* P,
  const value_parameters* V, 
  int Nstep,
  const bool* stoppingRule)
{
  func_ptr_value f = (V->f == NULL ? &zero_function : V->f);
  func_ptr_value h = (V->h == NULL ? &zero_function : V->h);
  func_ptr_value F = (V->F == NULL ? &zero_function : V->F);
  const bool hasStoppingArray = (stoppingRule != NULL);
  const double mu = P->mu;
  const double sigma = P->sigma;
  const double dt = P->dt;
  const double sqrtdt = sqrt(dt);
  double value = 0.0;
  double x = P->xnaught;
  double t = 0.0;
  int k = 0;
  for (;;) {
    if (k == Nstep) {
      value += (*F)(t, x, V->data_ptr);
      break;
    }
    if (hasStoppingArray && fetch_decision(k, x - P->xnaught, P->delta, stoppingRule)) {
      value += (*F)(t, x, V->data_ptr);
      break;
    }
    const double this_h = (*h)(t, x, V->data_ptr);
    const double r1 = xoshiro256plus_next_double(rng);
    if (r1 > exp(-1.0 * dt * this_h)) break;
    const double this_f = (*f)(t, x, V->data_ptr);
    value += this_f * dt;
    const double r2 = xoshiro256plus_next_double(rng);
    const double w = ppnd16(r2, NULL);
    x += mu * dt + sigma * sqrtdt * w;
    t += dt;
    k ++;
  }
  return value;
}

/* ------------------------------------------------------------------------- */

// European call terminal value: F(t, s | K) = max(0, s - K); only called at t = T
double terminal_european_call(double t, double x, const void* data_ptr) {
  const double K = ((const double*)data_ptr)[0];
  const double s = exp(x); // log-space value x = log(s)
  return (s > K ? s - K : 0.0);
}

double terminal_european_put(double t, double x, const void* data_ptr) {
  const double K = ((const double*)data_ptr)[0];
  const double s = exp(x); // log-space value x = log(s)
  return (s > K ? 0.0 : K - s);
}

double constant_discount_function(double t, double x, const void* data_ptr) {
  const double r = ((const double*)data_ptr)[1];
  return r;
}

// closed form Black-Scholes value of European call/put option; set r = mu
double black_scholes_merton_european(
  double t, 
  double x, 
  double r, 
  double sigma, 
  double q,
  double K, 
  double T,
  bool call)
{
  const double s = exp(x);
  const double tau = T - t;
  const double d1 = (log(s / K) + (r - q + sigma * sigma / 2.0) * tau) / (sigma * sqrt(tau));
  const double d2 = d1 - sigma * sqrt(tau);
  const double exprtau = exp(-1.0 * r * tau);
  const double expqtau = exp(-1.0 * q * tau);
  if (call)
    return normalCDF(d1) * s * expqtau - normalCDF(d2) * K * exprtau;
  else
    return normalCDF(-d2) * K * exprtau - normalCDF(-d1) * s * expqtau;
}

/* ------------------------------------------------------------------------- */

double test_function_F(double t, double x, const void* data_ptr) {
  const double K = ((const double*)data_ptr)[0]; // use K as a length scale
  const double xhat = x / fabs(K);
  return fabs(sin(xhat));
}

double test_function_h(double t, double x, const void* data_ptr) {
  const double h = ((const double*)data_ptr)[1];
  return h;
}

double test_function_f(double t, double x, const void* data_ptr) {
  return cos(t + x);
}

/* ------------------------------------------------------------------------- */

void update_mean_and_meansquare(int i, double xi, double* ex, double* ex2) {
  const double a = 1.0 / (i + 1);
  const double b = ((double) i) / (i + 1);
  *ex = a * xi + b * (*ex);
  *ex2 = a * (xi * xi) + b * (*ex2);
}

/* ------------------------------------------------------------------------- */

int main(int argc, const char** argv)
{
  uint64_t seed = 0x1234abcd4321efab;

  if (argc == 1) {
    xoshiro256plus_state local_rng;
    xoshiro256plus_init(&local_rng, seed);
    test_ppnd16_accuracy(&local_rng, 1e7);
    test_ppnd16_samples(&local_rng, 1e7); 
    const bool ok = test_array_index(1e3);
    printf("index code test result = %i\n", (ok ? 0 : -1));
    return 0;
  }

  if (argc != 9 && argc != 12) {
    printf("USAGE: %s mu sigma dt S0 Tmax q K option [nsamples nthreads sampler]\n", argv[0]);
    return 0;
  }

  const bool run_sampler = (argc == 12);

  process_parameters P;
  memset(&P, 0, sizeof(process_parameters));

  const double mu    = strtod(argv[1], NULL);
  const double sigma = strtod(argv[2], NULL);
  const double dt    = strtod(argv[3], NULL);
  const double S0    = strtod(argv[4], NULL);
  const double Tmax  = strtod(argv[5], NULL);

  if (sigma <= 0 || S0 <= 0) {
    printf("sigma > 0 and S0 > 0 required.\n");
    return -1;
  }

  if (dt <= 0 || Tmax <= 0) {
    printf("dt > 0 and Tmax > 0 required.\n");
    return -1;
  }

  const int Nstep = (int) floor(Tmax / dt);
  const int Nelem = array_index(Nstep, Nstep) + 1;

  printf("dt = %e, Tmax = %e -> Nstep = %i, lattice array length = %i\n", dt, Tmax, Nstep, Nelem);

  P.sigma = sigma;
  P.mu = mu - sigma * sigma / 2.0; // required to convert the GBM parameters to log-space (Ito's lemma)
  P.xnaught = log(S0);

  const double q = strtod(argv[6], NULL);
  const double K = strtod(argv[7], NULL);
  const char* option_name_str = argv[8];

  printf("option = \"%s\": K = %f, r = %f, q = %f\n", option_name_str, K, mu, q);

  if (q != 0) {
    P.mu -= q;
  }

  if (!assign_delta_t(&P, dt)) {
    printf("failed to initialize lattice jump parameters.\n");
    return -1;
  }

  printf("tree delta   = %.6e\n", P.delta);
  printf("(pd, pc, pu) = (%.6e, %.6e, %.6e)\n", P.pd, P.pc, P.pu);

  double value_data[2];
  value_data[0] = K;
  value_data[1] = mu;  // perhaps should be renamed to r

  value_parameters V;
  memset(&V, 0, sizeof(value_parameters));
  V.data_ptr = value_data;

  const bool isAmerican = (strstr(option_name_str, "american") != NULL);

  if (strcmp(option_name_str, "european-call") == 0) {
    V.F = &terminal_european_call;
    V.h = &constant_discount_function;

    double ref_val = black_scholes_merton_european(0.0, P.xnaught, mu, sigma, q, K, Nstep * dt, true);
    printf("Black-Scholes-Merton solution     = %e (call)\n", ref_val);

  } else if (strcmp(option_name_str, "european-put") == 0) {
    V.F = &terminal_european_put;
    V.h = &constant_discount_function;

    double ref_val = black_scholes_merton_european(0.0, P.xnaught, mu, sigma, q, K, Nstep * dt, false);
    printf("Black-Scholes-Merton solution     = %e (put)\n", ref_val);

  } else if (strcmp(option_name_str, "american-call") == 0) {
    V.F = &terminal_european_call;
    V.h = &constant_discount_function;

  } else if (strcmp(option_name_str, "american-put") == 0) {
    V.F = &terminal_european_put;
    V.h = &constant_discount_function;

  } else if (strcmp(option_name_str, "european-test") == 0 ||
             strcmp(option_name_str, "american-test") == 0) {
    V.F = &test_function_F;
    V.h = &test_function_h;
  //  V.f = &test_function_f;

  } else {
    printf("unknown option name: \"%s\"\n", option_name_str);
    return -1;
  }

  double* A = malloc(sizeof(double) * Nelem);
  bool* X = (isAmerican ? malloc(sizeof(bool) * Nelem) : NULL);

  if (A == NULL || (isAmerican && X == NULL)) {
    printf("Array allocation(s) (%.1f MB) failed.\n", 
      ((sizeof(double) + sizeof(bool)) * Nelem) / (1024.0 * 1024.0));
    if (A != NULL) free(A);
    if (X != NULL) free(X);
    return -1;
  }

  if (isAmerican) {
    // Need different handling for American option put/call; since it requires a stopping rule
    // prior to the forward valuation (it comes out as byproduct of the backward valuation) 
    // We would need 2 arrays to provide the 2nd moment from the backward pass; so don't bother.

    double value_bwrd = backward_valuation_terminatable(&P, &V, A, Nstep, X);
    printf("Trinomial recomb. tree (backward) = %e (defines stopping rule)\n", value_bwrd);

    double value_fwrd = forward_valuation_terminatable(&P, &V, A, Nstep, X, NULL);
    double value_fwrd_sq = forward_valuation_terminatable(&P, &V, A, Nstep, X, &square_transform);
    printf("Trinomial recomb. tree (forward)  = %e (std = %e) (rule applied)\n", 
      value_fwrd, sqrt(value_fwrd_sq - value_fwrd * value_fwrd));

  } else {
    double value_bwrd = backward_valuation(&P, &V, A, Nstep, NULL);
    double value_bwrd_sq = backward_valuation(&P, &V, A, Nstep, &square_transform);
    printf("Trinomial recomb. tree (backward) = %e (std = %e)\n", 
      value_bwrd, sqrt(value_bwrd_sq - value_bwrd * value_bwrd));

    double value_fwrd = forward_valuation(&P, &V, A, Nstep, NULL);
    double value_fwrd_sq = forward_valuation(&P, &V, A, Nstep, &square_transform);
    printf("Trinomial recomb. tree (forward)  = %e (std = %e) (open-loop)\n", 
      value_fwrd, sqrt(value_fwrd_sq - value_fwrd * value_fwrd));
  }

  free(A);

  if (!run_sampler) {
    printf("done (no sampling requested).\n");
    if (isAmerican) free(X);
    return 0;
  }

  const int nsamples = (int) round(strtod(argv[9], NULL));
  int nthreads = (int) round(strtod(argv[10], NULL));

  path_func_ptr path_sampler = NULL;
  const char* sampler_name_str = argv[11];
  if (strcmp(sampler_name_str, "mortal") == 0) {
    path_sampler = &one_mortal_path_sample;
  } else if (strcmp(sampler_name_str, "immortal") == 0) {
    path_sampler = &one_immortal_path_sample;
  } else {
    printf("did not recognize sampler \"%s\"; stopping.\n", sampler_name_str);
    return -1;
  }

  const int maxthreads = omp_get_max_threads();
  nthreads = (nthreads < 1 ? 1 : (nthreads > maxthreads ? maxthreads : nthreads));
  omp_set_num_threads(nthreads);

  printf("drawing %i path samples (sampler = \"%s\") each for %i thread(s)..\n", nsamples, sampler_name_str, nthreads);

  double EX = 0.0;
  double EX2 = 0.0;

  #pragma omp parallel
  {
    xoshiro256plus_state local_rng;
    xoshiro256plus_init(&local_rng, seed);

    const int tid = omp_get_thread_num();
    const int tnum = omp_get_num_threads();

    if (tid == 0 && tnum != nthreads) {
      printf("warning: requested %i threads but got %i\n", nthreads, tnum);
    }

    for (int j = 0; j < tid; j++) {
      xoshiro256plus_long_jump(&local_rng);
    }

    int numtot = 0;
    double ex = 0.0;
    double ex2 = 0.0;
    while (numtot < nsamples) {
      double theSample = (*path_sampler)(&local_rng, &P, &V, Nstep, X);
      update_mean_and_meansquare(numtot++, theSample, &ex, &ex2);
    }

    #pragma omp atomic
    EX += ex / tnum;

    #pragma omp atomic
    EX2 += ex2 / tnum;
  }

  printf("E[X]   = %e\n", EX);
  printf("std[X] = %e\n", sqrt(EX2 - EX * EX));

  printf("done (%.1f M samples total).\n", nthreads * nsamples / 1.0e6);
  if (isAmerican) free(X);
  return 0;
}

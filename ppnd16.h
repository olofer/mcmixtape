/*
  Basic C implementation of the probit function PPND16.

  Purpose:
    Find the value of x such that normalCDF(x) = p, for any p in (0, 1),
    i.e. normalCDF(ppnd16(p)) = p.

  Simple C version of the normal cumulative density function:
    double normalCDF(double x) {
      return erfc(-1.0 * x / sqrt(2.0)) / 2.0;
    }

  Reference:
    M. J. Wichura, "Algorithm AS 241: The Percentage Points of the Normal Distribution",
    Journal of the Royal Statistical Society. Series C (Applied Statistics), vol. 37, 1988, pp. 477-484
    URL: http://www.jstor.org/stable/2347330
 */

#ifndef __PPND16_H__
#define __PPND16_H__

double ppnd16(double p, int* istatus) {
  const double ZERO = 0.0;
  const double ONE = 1.0;
  const double HALF = 0.5;
  const double SPLIT1 = 0.425;
  const double SPLIT2 = 5.0;
  const double CONST1 = 0.180625;
  const double CONST2 = 1.6;
  const double A[8] = {  /* A0..7 */
    3.3871328727963665,
    133.14166789178438,
    1971.5909503065514,
    13731.693765509461,
    45921.95393154987,
    67265.7709270087,
    33430.575583588128,
    2509.0809287301227
  };
  const double B[8] = {  /* B0=unused, B1..7 */
    0.0,
    42.313330701600911,
    687.18700749205789,
    5394.1960214247511,
    21213.794301586597,
    39307.895800092709,
    28729.085735721943,
    5226.4952788528544
  };
  const double C[8] = {  /* C0..7 */
    1.4234371107496835,
    4.6303378461565456,
    5.769497221460691,
    3.6478483247632045,
    1.2704582524523684,
    0.24178072517745061,
    0.022723844989269184,
    0.00077454501427834139
  };
  const double D[8] = {  /* D0=unused, D1..7 */
    0.0,
    2.053191626637759,
    1.6763848301838038,
    0.6897673349851,
    0.14810397642748008,
    0.015198666563616457,
    0.00054759380849953455,
    0.0000000010507500716444169
  };
  const double E[8] = {  /* E0..7 */
    6.6579046435011033,
    5.4637849111641144,
    1.7848265399172913,
    0.29656057182850487,
    0.026532189526576124,
    0.0012426609473880784,
    0.000027115555687434876,
    0.00000020103343992922882
  };
  const double F[8] = {  /* F0=unused, F1..7 */
    0.0,
    0.599832206555888,
    0.13692988092273581,
    0.014875361290850615,
    0.00078686913114561329,
    0.000018463183175100548,
    0.0000001421511758316446,
    0.0000000000000020442631033899397
  };

  double aa, bb, r;
  double q = p - HALF;
  double z = 0.0;
  if (istatus != NULL) *istatus = 0;
  if (fabs(q) <= SPLIT1) {
    r = CONST1 - q * q;
    aa = ((((((A[7] * r + A[6]) * r + A[5]) * r + A[4]) * r + A[3]) * r + A[2]) * r + A[1]) * r + A[0];
    bb = ((((((B[7] * r + B[6]) * r + B[5]) * r + B[4]) * r + B[3]) * r + B[2]) * r + B[1]) * r + ONE;
    z = q * (aa / bb);
    if (istatus != NULL) *istatus = 1;
  } else {
    if (q < ZERO) r = p; else r = 1.0 - p;
    if (r <= ZERO) return z;
    r = sqrt(-log(r));
    if (r <= SPLIT2) {
      r = r - CONST2;
      aa = ((((((C[7] * r + C[6]) * r + C[5]) * r + C[4]) * r + C[3]) * r + C[2]) * r + C[1]) * r + C[0];
      bb = ((((((D[7] * r + D[6]) * r + D[5]) * r + D[4]) * r + D[3]) * r + D[2]) * r + D[1]) * r + ONE;
      if (istatus != NULL) *istatus = 2;
    } else {
      r = r - SPLIT2;
      aa = ((((((E[7] * r + E[6]) * r + E[5]) * r + E[4]) * r + E[3]) * r + E[2]) * r + E[1]) * r + E[0];
      bb = ((((((F[7] * r + F[6]) * r + F[5]) * r + F[4]) * r + F[3]) * r + F[2]) * r + F[1]) * r + ONE;
      if (istatus != NULL) *istatus = 3;
    }
    z = aa / bb;
    if (q < ZERO) z = -z;
  }
  return z;
}

#endif

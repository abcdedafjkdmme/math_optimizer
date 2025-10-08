#ifndef APPROXIMATIONS_H
#define APPROXIMATIONS_H

#include "math.h"
#include "stddef.h"

double poly_approx(double x, double *coffs, size_t num_coffs) {
  // polynomial approximation
  double y = 0;
  for (size_t i = 0; i < num_coffs; i++) {
    double a = coffs[i];
    y += a * pow(x, i);
  }
  return y;
}

#endif
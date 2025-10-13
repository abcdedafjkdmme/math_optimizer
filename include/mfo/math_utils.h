#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "stdlib.h"
#include "math.h"

double nn_rand_range(double min, double max)
{
  double scale = rand() / (double)RAND_MAX; /* [0, 1.0] */
  return min + scale * (max - min);         /* [min, max] */
}
double nn_clamp(double d, double min, double max)
{
  const double t = d < min ? min : d;
  return t > max ? max : t;
}


double map_range(double x, double x_min, double x_max, double y_min,
                 double y_max) {
  return (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min;
}


#endif
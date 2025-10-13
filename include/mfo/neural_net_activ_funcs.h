#ifndef NEURAL_NET_ACTIV_FUNCS_H
#define NEURAL_NET_ACTIV_FUNCS_H

#include <math.h>

double nn_sigmoid(double x)
{
  return 1 / (1 + (exp(-x)));
}
double nn_relu(double x)
{
  return fmax(0, x);
}
double nn_leaky_relu(double x)
{
  double a = 0.1;
  return fmax(a * x, x);
}
double nn_identity(double x)
{
  return x;
}

#endif
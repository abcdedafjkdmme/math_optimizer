#ifndef MATH_OPTIMIZER_H
#define MATH_OPTIMIZER_H

#include "stddef.h"



typedef double (*optimizer_func_t)(double *args, size_t num_args,
                                   void *user_data);

void optimizer_minimize_func(optimizer_func_t func, void *func_data,
                             double *optimized_args, size_t num_args,
                             double delta, double rate, size_t n) {

  // randomize args first
  for (int i = 0; i < num_args; i++) {
    // wont randomize for now
    optimized_args[i] = 0.0;
  }
  // run the algorithm for n times
  for (size_t k = 0; k < n; k++) {
    // for each argument
    for (size_t i = 0; i < num_args; i++) {
      // calculate the return value of function
      double val = (*func)(optimized_args, num_args, func_data);
      // save current arg
      double arg = optimized_args[i];
      // increase arg by delta
      optimized_args[i] += delta;
      // calculate the new return value of function
      double new_val = (*func)(optimized_args, num_args, func_data);
      // calculate the derivative
      double deriv = (new_val - val) / delta;
      // increase arg by negative derivative multiplied by rate
      optimized_args[i] = arg - (deriv * rate);
    }
  }
}

#endif
#ifndef ERR_FUNC_H
#define ERR_FUNC_H

#include "assert.h"
#include "stddef.h"

typedef double (*err_func_t)(double input, void *user_data);

double err_func(err_func_t func, void *func_data, double *expected_inputs,
                size_t num_expected_inputs, double *expected_outputs,
                size_t num_expected_outputs) {

  assert(num_expected_inputs == num_expected_outputs);

  double tot_err = 0;
  // for each (input,output) pair calculate the err
  for (size_t i = 0; i < num_expected_inputs; i++) {
    double actual_output = func(expected_inputs[i], func_data);
    double err = (actual_output - expected_outputs[i]) / expected_outputs[i];
    double err_squared = err * err;
    // add to total error
    tot_err += err_squared;
  }
  return tot_err;
}

#endif
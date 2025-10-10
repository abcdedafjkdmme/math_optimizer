#include "assert.h"
#include "math.h"
#include "stdio.h"

#include "arrsize.h"
#include "err_func.h"

double exp_approximation(double x, void *user_data) {
  double y = 1.0 + x + (x * x) / 2.0 + +(x * x * x) / 6.0;
  return y;
}

int main() {
  // find the err of an exp(x) approximation
  // create sample data of e^x
  double fit_data_input[100];
  double fit_data_output[100];
  for (size_t i = 0; i < ARRAY_SIZE(fit_data_input); i++) {
    double x = i;
    fit_data_input[i] = x;
    double y = exp(x);
    fit_data_output[i] = y;
  }

  double err = err_func(&exp_approximation, NULL, fit_data_input,
                        ARRAY_SIZE(fit_data_input), fit_data_output,
                        ARRAY_SIZE(fit_data_output));
  printf("error of exp(x) approximation is %lf \n", err);
  return 0;
}
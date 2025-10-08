#include "assert.h"
#include "math.h"
#include "stdio.h"

#include "approximations.h"
#include "arrsize.h"
#include "err_func.h"
#include "optimizer.h"

double map_range(double x, double x_min, double x_max, double y_min,
                 double y_max) {
  return (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min;
}
// wrap polynomial approximation func so that it is usable with err_func()
typedef struct {
  double *coffs;
  size_t num_coffs;
} poly_approx_data_t;

double poly_approx_wrapper(double x, void *user_data) {
  poly_approx_data_t *d = (poly_approx_data_t *)(user_data);
  return poly_approx(x, d->coffs, d->num_coffs);
}

// wrap err_func so that it is usable with optimizer_minimize_func()
typedef struct {
  double *expected_inputs;
  size_t num_expected_inputs;
  double *expected_outputs;
  size_t num_expected_outputs;
} poly_approx_err_data_t;

double poly_approx_err_func_wrapper(double *args, size_t num_args,
                                    void *user_data) {
  poly_approx_err_data_t *d = (poly_approx_err_data_t *)(user_data);
  poly_approx_data_t poly_approx_data = {args, num_args};
  double err = err_func(&poly_approx_wrapper, &poly_approx_data,
                        d->expected_inputs, d->num_expected_inputs,
                        d->expected_outputs, d->num_expected_outputs);
  return err;
}

int main() {
  printf("program start \n");

  // create sample data of e^x
  double fit_data_input[100];
  double fit_data_output[100];
  for (size_t i = 0; i < OPT_ARRAY_SIZE(fit_data_input); i++) {
    double x = map_range(i, 0, OPT_ARRAY_SIZE(fit_data_input), 0, 10);
    fit_data_input[i] = x;
    double y = exp(x);
    fit_data_output[i] = y;
  }

  // optimize error function
  double new_coffs[4];
  poly_approx_err_data_t poly_approx_err_data = {
      fit_data_input, OPT_ARRAY_SIZE(fit_data_input), fit_data_output,
      OPT_ARRAY_SIZE(fit_data_output)};
  optimizer_minimize_func(&poly_approx_err_func_wrapper, &poly_approx_err_data,
                          new_coffs, OPT_ARRAY_SIZE(new_coffs), 0.1, 0.01,
                          10000);

  printf("new coffs are [%lf, %lf, %lf, %lf] \n", new_coffs[0], new_coffs[1],
         new_coffs[2], new_coffs[3]);


  // get the current error of our universal approximator
  poly_approx_data_t poly_approx_data = {new_coffs, OPT_ARRAY_SIZE(new_coffs)};
  double err_2 = err_func(&poly_approx_wrapper, &poly_approx_data,
                          fit_data_input, OPT_ARRAY_SIZE(fit_data_input),
                          fit_data_output, OPT_ARRAY_SIZE(fit_data_output));
  printf("error after optimization is %lf \n", err_2);

  printf("program end \n");
  return 0;
}
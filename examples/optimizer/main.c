#include "assert.h"
#include "math.h"
#include "stdio.h"

#include "arrsize.h"
#include "optimizer.h"

double example_func(double *args, size_t num_args, void *user_data) {
  assert(num_args == 1);
  double x = args[0];
  double y = (x * x) - (5 * x) + (6);
  return y;
}

double example_func_3d(double *args, size_t num_args, void *user_data) {
  assert(num_args == 2);
  double x = args[0];
  double y = args[1];
  double z = (x * x) + (y * y) + (x * y) + (5 * x) + (6 * y) + (3);
  return z;
}

int main(){
  // optimize first functions
  double args[1];
  optimizer_minimize_func(&example_func, NULL, args, 1, 0.01, 0.1, 200);
  double min_x = args[0];
  double min_y = example_func(args, 1, NULL);
  printf("min (x,y) = (%lf ,%lf) \n", min_x, min_y);

  // optimize second functions
  double args_2[2];
  optimizer_minimize_func(&example_func_3d, NULL, args_2, 2, 0.01, 0.1, 200);
  double min_x_2 = args_2[0];
  double min_y_2 = args_2[1];
  double min_z_2 = example_func_3d(args_2, 2, NULL);
  printf("min (x,y,z) = (%lf ,%lf ,%lf) \n", min_x_2, min_y_2, min_z_2);

  return 0;
}
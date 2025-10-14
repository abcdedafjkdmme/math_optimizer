#include "assert.h"
#include "math.h"
#include "stdio.h"
#include "time.h"
#include "memory.h"
#include "math.h"
#include <errno.h>

#include "arrsize.h"
#include "neural_net.h"
#include "neural_net_wrapper.h"
#include "neural_net_activ_funcs.h"
#include "gnuplot_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.7182818284590452354
#endif

/**
 * @warning only supports neural netoworks with 1 input and 1 output
 */
void create_training_data(nn_training_data *td, double *expected_inputs, size_t num_expected_inputs, double *expected_outputs, size_t num_expected_outputs, double range_min, double range_max)
{
  assert(num_expected_inputs == num_expected_outputs);
  size_t num_data_points = num_expected_inputs;

  for (size_t i = 0; i < num_data_points; i++)
  {
    double x = map_range((double)i, 0.0, (double)num_data_points, range_min, range_max);
    expected_inputs[i] = x;
    double y = 0; // not used
    expected_outputs[i] = y;
  }

  td->expected_inputs = (double *)expected_inputs;
  td->num_inputs = 1;
  td->num_expected_inputs = num_data_points;
  td->expected_outputs = (double *)expected_outputs;
  td->num_outputs = 1;
  td->num_expected_outputs = num_data_points;
}

// for optimizing the equation f(f(x))-x=0
double custom_cost_func_linear(nn_network_t *net, double *ex_input_data_point, double *ex_output_data_point)
{
  double x = ex_input_data_point[0];
  double f_x;
  nn_predict(net, ex_input_data_point, 1, &f_x, 1);
  double f_f_x;
  nn_predict(net, &f_x, 1, &f_f_x, 1);

  double err = (f_f_x - x);
  double err_sqr = 10.0 * err * err;
  return err_sqr;
}

// for optimizing the equation
double custom_cost_func_exp(nn_network_t *net, double *ex_input_data_point, double *ex_output_data_point)
{
  double x = ex_input_data_point[0];
  double f_x;
  nn_predict(net, &x, 1, &f_x, 1);

  double x_plus_one = x + 1;
  double f_x_plus_one;
  nn_predict(net, &x_plus_one, 1, &f_x_plus_one, 1);

  double err = (f_x_plus_one - f_x * M_E);
  if (abs(f_x) < 0.1)
  {
    err += 2;
  }
  double err_sqr = 10 * err * err;
  return err_sqr;
}

double custom_cost_func(nn_network_t *net, double *ex_input_data_point, double *ex_output_data_point)
{
  return custom_cost_func_exp(net, ex_input_data_point, ex_output_data_point);
}

int main()
{
  printf("program start\n");

  srand((unsigned int)time(NULL));

  nn_arena_t arena;
  nn_arena_create(&arena, 65536);

  nn_network_t knet = {};

  nn_add_layer(&arena, &knet, 1, NULL);
  nn_add_layer(&arena, &knet, 4, &nn_relu);
  nn_add_layer(&arena, &knet, 8, &nn_relu);
  nn_add_layer(&arena, &knet, 8, &nn_relu);
  nn_add_layer(&arena, &knet, 4, &nn_relu);
  nn_add_layer(&arena, &knet, 1, &nn_identity);

  printf("neural network created\n");

  double fit_range_min = -M_PI;
  double fit_range_max = M_PI;

  // create training data
  nn_training_data td;
  double expected_inputs[200];
  double expected_outputs[200];
  create_training_data(&td, expected_inputs, ARRAY_SIZE(expected_inputs), expected_outputs, ARRAY_SIZE(expected_outputs), fit_range_min, fit_range_max);
  printf("training data created\n");

  // optimize the network
  double clamp_min = -1.0;
  double clamp_max = 1.0;
  size_t num_iters = 1000;
  size_t num_mini_batches = 20;
  double delta = 0.001;
  double rate = 0.001;

  // data dump of cost
  double iters[num_iters];
  double costs[num_iters];

  // randomize weights and biases
  nn_randomize_net(&knet, clamp_min, clamp_max);

  // create an arena for sgd training data
  nn_arena_t arena_sgd;
  nn_arena_create(&arena_sgd, 10 * 1048576);

  for (size_t i = 0; i < num_iters; i++)
  {
    nn_training_data td_new;
    nn_create_sgd_traning_data(&arena_sgd, &td, &td_new, num_mini_batches);

    nn_optimize_iter(&knet, &td_new, &custom_cost_func, delta, rate, clamp_min, clamp_max);
    // write new costs of every iteration
    iters[i] = (double)i;
    costs[i] = nn_get_cost(&knet, &td, &custom_cost_func);
  }
  nn_arena_destroy(&arena_sgd);

  // dump cost vs iter plot data to file
  char cost_plot_data[100000];
  plt_create_plot_data(cost_plot_data, ARRAY_SIZE(cost_plot_data), iters, costs, num_iters);
  FILE *cost_plot_data_file = fopen("cost_plot_data.txt", "w");
  if (cost_plot_data_file == NULL)
  {
    printf("error opening cost plot data file");
    return -1;
  }
  fputs((const char *)&cost_plot_data, cost_plot_data_file);
  fclose(cost_plot_data_file);

  // get new error after training
  double err = nn_get_cost(&knet, &td, &custom_cost_func);
  printf("new err is %lf \n", err);

  // create plot data
  char plot_data_str[5000];
  nn_arena_t plot_arena;
  nn_arena_create(&plot_arena, 65536);
  nn_create_y_vs_x_plot(&plot_arena, &knet, plot_data_str, ARRAY_SIZE(plot_data_str), fit_range_min, fit_range_max, 200);
  nn_arena_destroy(&plot_arena);
  FILE *plot_data_file = fopen("plot_data.txt", "w");

  if (plot_data_file == NULL)
  {
    printf("ERROR: can't create file %s \n", strerror(errno));
    assert(1 == 0);
  }
  fputs((const char *)&plot_data_str, plot_data_file);
  fclose(plot_data_file);

  // open plot data in gnuplot
  int errcode = system("gnuplot -p data/plot.gp");
  assert(errcode == 0);

  // destroy stuff
  nn_network_destroy(&knet);
  nn_arena_destroy(&arena);
  printf("program end \n");
  return 0;
}
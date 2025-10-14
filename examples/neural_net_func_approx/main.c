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

double training_func(double x)
{
  return x * x - 3 * x + 5;
}
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
    double y = training_func(x);
    expected_outputs[i] = y;
  }

  td->expected_inputs = (double *)expected_inputs;
  td->num_inputs = 1;
  td->num_expected_inputs = num_data_points;
  td->expected_outputs = (double *)expected_outputs;
  td->num_outputs = 1;
  td->num_expected_outputs = num_data_points;
}

int main()
{
  printf("program start\n");

  srand((unsigned int)time(NULL));

  // new way of creating network
  nn_arena_t arena;
  nn_arena_create(&arena, 65536);

  //  nn_layer_t klayers[6];
  //  nn_network_t knet = {.layers = klayers, .num_layers = 0};
  nn_network_t knet = {};

  nn_add_layer(&arena, &knet, 1, NULL);
  nn_add_layer(&arena, &knet, 16, &nn_relu);
  nn_add_layer(&arena, &knet, 20, &nn_relu);
  nn_add_layer(&arena, &knet, 16, &nn_relu);
  nn_add_layer(&arena, &knet, 1, &nn_identity);

  printf("neural network created\n");

  // get new outputs
  double inputs[] = {2.0};
  double outputs[1];
  nn_predict(&knet, inputs, ARRAY_SIZE(inputs), outputs, ARRAY_SIZE(outputs));
  printf("new      (input,output) is (%lf, %lf)\n", inputs[0], outputs[0]);
  printf("expected (input,output) is (%lf, %lf)\n", inputs[0], training_func(inputs[0]));

  // create training data
  nn_training_data td;
  double expected_inputs[100];
  double expected_outputs[100];
  double fit_range_min = -20.0;
  double fit_range_max = 20.0;
  create_training_data(&td, expected_inputs, ARRAY_SIZE(expected_inputs), expected_outputs, ARRAY_SIZE(expected_outputs), fit_range_min, fit_range_max);
  printf("training data created\n");
  // get the current error based on training data
  double err = nn_get_cost(&knet, &td, &nn_cost_func_mse);
  printf("err is %lf \n", err);

  // optimize the network

  double clamp_min = -1.0;
  double clamp_max = 1.0;
#define NUM_ITERS 500
  size_t num_mini_batches = 20;
  double delta = 0.0001;
  double rate = 0.00001;

  // data dump of cost
  double iters[NUM_ITERS];
  double costs[NUM_ITERS];

  // randomize weights and biases
  nn_randomize_net(&knet, clamp_min, clamp_max);
  // create an arena for sgd traning data
  nn_arena_t arena_sgd;
  nn_arena_create(&arena_sgd, 10 * 1048576);
  for (size_t i = 0; i < NUM_ITERS; i++)
  {
    nn_training_data td_new;
    nn_create_sgd_traning_data(&arena_sgd, &td, &td_new, num_mini_batches);
    nn_optimize_iter(&knet, &td_new, &nn_cost_func_mse, delta, rate, clamp_min, clamp_max);
    // write new costs of every iteration
    iters[i] = (double)i;
    costs[i] = nn_get_cost(&knet, &td, &nn_cost_func_mse);
  }
  nn_arena_destroy(&arena_sgd);
  // dump cost vs iter plot data to file
  char cost_plot_data[10000];
  plt_create_plot_data(cost_plot_data, ARRAY_SIZE(cost_plot_data), iters, costs, NUM_ITERS);
  FILE *cost_plot_data_file = fopen("cost_plot_data.txt", "w");
  if (cost_plot_data_file == NULL)
  {
    printf("error opening cost plot data file");
    return -1;
  }
  fputs((const char *)&cost_plot_data, cost_plot_data_file);
  fclose(cost_plot_data_file);

  // get new error after training
  err = nn_get_cost(&knet, &td, &nn_cost_func_mse);
  printf("new err is %lf \n", err);

  // get new outputs after training
  double inputs2[] = {2.0};
  double outputs2[1];
  nn_predict(&knet, inputs2, ARRAY_SIZE(inputs2), outputs2, ARRAY_SIZE(outputs2));
  printf("new      (input,output) is (%lf, %lf)\n", inputs2[0], outputs2[0]);
  printf("expected (input,output) is (%lf, %lf)\n", inputs2[0], training_func(inputs2[0]));

  // create plot data
  char plot_data_str[5000];
  nn_arena_t plot_arena;
  nn_arena_create(&plot_arena, 65536);
  nn_create_y_vs_x_plot(&plot_arena, &knet, plot_data_str, ARRAY_SIZE(plot_data_str), fit_range_min, fit_range_max, 100);
  nn_arena_destroy(&plot_arena);
  FILE *plot_data_file = fopen("plot_data.txt", "w");
  if (plot_data_file == NULL)
  {
    printf("error opening plot data file");
    return -1;
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
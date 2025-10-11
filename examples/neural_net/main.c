#include "assert.h"
#include "math.h"
#include "stdio.h"
#include "time.h"
#include "memory.h"
#include "math.h"

#include "arrsize.h"
#include "neural_net.h"
#include "neural_net_wrapper.h"

double training_func(double x)
{
  return 10.0 * sin(x);
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

int main(int argc, char *argv[])
{
  printf("program start\n");

  srand((unsigned int)time(NULL));

  // new way of creating network
  nn_arena_t arena;
  nn_arena_create(&arena, 65536);

  nn_layer_t klayers[4];
  nn_network_t knet = {.layers = klayers, .num_layers = 0};

  nn_add_layer(&arena, &knet, ARRAY_SIZE(klayers), 1, NULL);
  nn_add_layer(&arena, &knet, ARRAY_SIZE(klayers), 32, &nn_relu);
  nn_add_layer(&arena, &knet, ARRAY_SIZE(klayers), 32, &nn_relu);
  nn_add_layer(&arena, &knet, ARRAY_SIZE(klayers), 1, &nn_identity);

  printf("neural network created\n");

  // get new outputs
  double inputs[] = {2.0};
  double outputs[1];
  nn_predict(&knet, inputs, ARRAY_SIZE(inputs), outputs, ARRAY_SIZE(outputs));
  printf("new      (input,output) is (%lf, %lf)\n", inputs[0], outputs[0]);
  printf("expected (input,output) is (%lf, %lf)\n", inputs[0], training_func(inputs[0]));

  nn_training_data td;
  double expected_inputs[100];
  double expected_outputs[100];
  create_training_data(&td, expected_inputs, ARRAY_SIZE(expected_inputs), expected_outputs, ARRAY_SIZE(expected_outputs), 0, 2.0 * M_PI);
  printf("training data created\n");
  double err = nn_get_cost(&knet, &td);
  printf("err is %lf \n", err);

  // optimize it now
  nn_fit(&knet, &td, -1.0, 1.0, 0.00001, 0.001, 1000);
  // get new error
  err = nn_get_cost(&knet, &td);
  printf("new err is %lf \n", err);

  // get new outputs
  double inputs2[] = {2.0};
  double outputs2[1];
  nn_predict(&knet, inputs2, ARRAY_SIZE(inputs2), outputs2, ARRAY_SIZE(outputs2));
  printf("new      (input,output) is (%lf, %lf)\n", inputs2[0], outputs2[0]);
  printf("expected (input,output) is (%lf, %lf)\n", inputs2[0], training_func(inputs2[0]));

  nn_arena_destroy(&arena);
  printf("program end \n");
  return 0;
}
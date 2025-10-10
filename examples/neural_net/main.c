#include "assert.h"
#include "math.h"
#include "stdio.h"
#include "time.h"

#include "arrsize.h"
#include "neural_net.h"

double map_range(double x, double x_min, double x_max, double y_min,
                 double y_max) {
  return (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min;
}

double training_func(double x){
  return x * x;
}
/**
 * @warning only supports neural netoworks with 1 input and 1 output
 */
void create_training_data(nn_training_data* td, double* expected_inputs, size_t num_expected_inputs, double* expected_outputs, size_t num_expected_outputs, double range_min, double range_max){
  assert(num_expected_inputs == num_expected_outputs);
  size_t num_data_points = num_expected_inputs;

  for (size_t i = 0; i < num_data_points; i++)
  {
    double x = map_range((double)i,0.0,(double)num_data_points,range_min,range_max);
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
  printf("program start \n\n");

  srand((unsigned int)time(NULL));
  // layer 1
  nn_node_t node_1 = {};
  node_1.output = 3; // input
  nn_node_t nodes_l1[] = {node_1};
  nn_layer_t layer_1 = {.nodes = nodes_l1, .num_nodes = ARRAY_SIZE(nodes_l1)};

  // layer 2
  nn_node_t node_2 = {};
  double node_2_weights[] = {2};
  node_2.weights = node_2_weights;
  node_2.num_weights = ARRAY_SIZE(node_2_weights);

  nn_node_t node_3 = {};
  double node_3_weights[] = {4};
  node_3.weights = node_3_weights;
  node_3.num_weights = ARRAY_SIZE(node_3_weights);

  nn_node_t nodes_l2[] = {node_2, node_3};
  nn_layer_t layer_2 = {.nodes = nodes_l2, .num_nodes = ARRAY_SIZE(nodes_l2), .activ_func = &nn_relu};

  // layer 3
  nn_node_t node_4 = {};
  double node_4_weights[] = {3, 1};
  node_4.weights = node_4_weights;
  node_4.num_weights = ARRAY_SIZE(node_4_weights);

  nn_node_t nodes_l3[] = {node_4};
  nn_layer_t layer_3 = {.nodes = nodes_l3, .num_nodes = ARRAY_SIZE(nodes_l3), .activ_func = &nn_relu};

  nn_layer_t layers[] = {layer_1, layer_2, layer_3};

  nn_network_t net = {.layers = layers, .num_layers = ARRAY_SIZE(layers)};

  nn_feedforward(&net);
  double output = net.layers[2].nodes[0].output;
  // printf("output is %lf \n", output);

  nn_training_data td;
  double expected_inputs[100];
  double expected_outputs[100];
  create_training_data(&td,expected_inputs,ARRAY_SIZE(expected_inputs),expected_outputs,ARRAY_SIZE(expected_outputs),-5.0,5.0);

  double err = nn_get_cost(&net, &td);
  printf("err is %lf \n", err);

  // optimize it now
  nn_optimize(&net, &td, -1.0, 1.0, 0.001, 0.001, 2000);
  // get new error
  err = nn_get_cost(&net, &td);
  printf("new err is %lf \n", err);

  // get new outputs
  double input = 3.0;
  net.layers[0].nodes[0].output = input;
  nn_feedforward(&net);
  output = net.layers[2].nodes[0].output;
  printf("\nnew      (input,output) is (%lf, %lf) \n", input, output);
  printf("\nexpected (input,output) is (%lf, %lf) \n",input,training_func(input));
  printf("program end \n");

  return 0;
}
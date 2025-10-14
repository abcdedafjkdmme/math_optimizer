#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "stddef.h"
#include "assert.h"
#include "stdlib.h"
#include "math_utils.h"
#include "arena.h"

typedef struct
{
  double *weights;
  size_t num_weights;
  double bias;
  double output;
} nn_node_t;

typedef double (*nn_activation_func_t)(double x);

typedef struct
{
  nn_node_t *nodes;
  size_t num_nodes;
  nn_activation_func_t activ_func;
} nn_layer_t;

typedef struct
{
  nn_layer_t *layers;
  size_t num_layers;
  size_t layer_capacity; // dynamic array
} nn_network_t;

nn_layer_t *nn_get_first_layer(nn_network_t *net)
{
  return &net->layers[0];
}

nn_layer_t *nn_get_last_layer(nn_network_t *net)
{
  assert(net->num_layers > 0);
  return &net->layers[net->num_layers - 1];
}
double nn_feedforward_node(nn_node_t *node, nn_layer_t *prev_layer, nn_activation_func_t activ_func)
{
  assert(prev_layer->num_nodes == node->num_weights);

  double output = node->bias;
  for (size_t i = 0; i < prev_layer->num_nodes; i++)
  {
    assert(node->weights != NULL);
    double w = node->weights[i];
    assert(prev_layer->nodes != NULL);
    double x = (prev_layer->nodes[i]).output;
    output += w * x;
  }
  return (*activ_func)(output);
}

void nn_feedforward(nn_network_t *net)
{
  // i is the layer
  // start from layer 1 since layer 0 is the inputs
  for (size_t i = 1; i < net->num_layers; i++)
  {
    nn_layer_t *layer = &net->layers[i];
    // j is the node index
    for (size_t j = 0; j < layer->num_nodes; j++)
    {
      nn_node_t *node = &layer->nodes[j];
      nn_layer_t *prev_layer = &net->layers[i - 1];
      double output = nn_feedforward_node(node, prev_layer, layer->activ_func);

      node->output = output;
    }
  }
}

void nn_set_inputs(nn_network_t *net, double *inputs, size_t num_inputs)
{

  nn_layer_t *first_layer = &net->layers[0];
  assert(first_layer->num_nodes == num_inputs);

  for (size_t i = 0; i < num_inputs; i++)
  {
    first_layer->nodes[i].output = inputs[i];
  }
}

void nn_predict(nn_network_t *net, double *inputs, size_t num_inputs, double *outputs, size_t num_outputs)
{
  nn_set_inputs(net, inputs, num_inputs);
  nn_feedforward(net);
  nn_layer_t *last_layer = &net->layers[net->num_layers - 1];
  assert(last_layer->num_nodes == num_outputs);
  for (size_t i = 0; i < last_layer->num_nodes; i++)
  {
    outputs[i] = last_layer->nodes[i].output;
  }
}

double nn_get_output(nn_network_t *net, size_t node_index)
{
  nn_layer_t *last_layer = &net->layers[net->num_layers - 1];
  double output = last_layer->nodes[node_index].output;
  return output;
}

typedef struct
{
  double *expected_inputs;     // 2d array
  size_t num_inputs;           // num cols
  size_t num_expected_inputs;  // num rows
  double *expected_outputs;    // 2d array
  size_t num_outputs;          // num cols
  size_t num_expected_outputs; // num rows
} nn_training_data;

typedef double (*nn_cost_func_impl_t)(nn_network_t *net, double *ex_input_data_point, double *ex_output_data_point);

double nn_cost_func_mse(nn_network_t *net, double *ex_input_data_point, double *ex_output_data_point)
{

  size_t num_inputs = nn_get_first_layer(net)->num_nodes;
  nn_set_inputs(net, ex_input_data_point, num_inputs);
  nn_feedforward(net);

  // for each value in neural network outputs
  double tot_err = 0;
  for (size_t j = 0; j < nn_get_last_layer(net)->num_nodes; j++)
  {

    double actual_output = nn_get_output(net, j);
    double expected_output = ex_output_data_point[j];
    double err = (actual_output - expected_output);
    double err_squared = err * err;
    // add to total error
    tot_err += err_squared;
  }
  return tot_err;
}

double nn_get_cost(nn_network_t *net, nn_training_data *td, nn_cost_func_impl_t cost_func)
{

  nn_layer_t *last_layer = &net->layers[net->num_layers - 1];
  nn_layer_t *first_layer = &net->layers[0];

  assert(td->num_inputs == first_layer->num_nodes);
  assert(td->num_outputs == last_layer->num_nodes);
  assert(td->num_expected_inputs == td->num_expected_outputs);

  double tot_err = 0;
  // for each (input,output) pair calculate the err
  for (size_t i = 0; i < td->num_expected_inputs; i++)
  {
    double *ex_input_data_point = (td->expected_inputs + i * td->num_inputs);
    double *ex_output_data_point = (td->expected_outputs + i * td->num_outputs);
    double err = (*cost_func)(net, ex_input_data_point, ex_output_data_point);
    tot_err += err;
  }
  return tot_err / (double)td->num_expected_outputs;
}

void nn_randomize_net(nn_network_t *net, double rand_range_min, double rand_range_max)
{
  // for each layer except for input layer
  for (size_t i = 1; i < net->num_layers; i++)
  {
    nn_layer_t *layer = &net->layers[i];
    // for each node
    for (size_t j = 0; j < layer->num_nodes; j++)
    {
      nn_node_t *node = &layer->nodes[j];
      // for bias
      node->bias = nn_rand_range(rand_range_min, rand_range_max);
      // for each weight
      for (size_t k = 0; k < node->num_weights; k++)
      {
        node->weights[k] = nn_rand_range(rand_range_min, rand_range_max);
      }
    }
  }
}

void nn_optimize_iter(nn_network_t *net, nn_training_data *td, nn_cost_func_impl_t cost_func, double delta, double rate, double clamp_min, double clamp_max)
{
  // for each layer except for input layer
  for (size_t i = 1; i < net->num_layers; i++)
  {
    nn_layer_t *layer = &net->layers[i];
    // for each node
    for (size_t j = 0; j < layer->num_nodes; j++)
    {
      nn_node_t *node = &layer->nodes[j];

      // for bias
      {
        double val = nn_get_cost(net, td, cost_func);
        double arg = node->bias;
        node->bias += delta;
        double new_val = nn_get_cost(net, td, cost_func);
        double deriv = (new_val - val) / delta;
        double new_bias = arg - (deriv * rate);
        node->bias = nn_clamp(new_bias, clamp_min, clamp_max);
      }
      // for each weight
      for (size_t k = 0; k < node->num_weights; k++)
      {
        double val = nn_get_cost(net, td, cost_func);
        double arg = node->weights[k];
        node->weights[k] += delta;
        double new_val = nn_get_cost(net, td, cost_func);
        double deriv = (new_val - val) / delta;
        double new_weight = arg - (deriv * rate);
        node->weights[k] = nn_clamp(new_weight, clamp_min, clamp_max);
      }
    }
  }
}

void nn_fit(nn_network_t *net, nn_training_data *td, nn_cost_func_impl_t cost_func, double delta, double rate, size_t num_iters, double clamp_min, double clamp_max)
{

  // randomize weights and biases
  nn_randomize_net(net, clamp_min, clamp_max);

  for (size_t i = 0; i < num_iters; i++)
  {
    nn_optimize_iter(net, td, cost_func, delta, rate, clamp_min, clamp_max);
 }
}

void nn_create_sgd_traning_data(nn_arena_t *arena, nn_training_data *td, nn_training_data *td_sgd, size_t num_mini_batches)
{
  size_t total_data_points = td->num_expected_inputs;
  assert(num_mini_batches < total_data_points);

  td_sgd->num_inputs = td->num_inputs;
  td_sgd->num_outputs = td->num_outputs;

  td_sgd->num_expected_inputs = num_mini_batches;
  td_sgd->num_expected_outputs = num_mini_batches;

  double *new_expected_inputs = nn_arena_alloc(arena, num_mini_batches * td->num_inputs * sizeof(double));
  double *new_expected_outputs = nn_arena_alloc(arena, num_mini_batches * td->num_outputs * sizeof(double));

  // bad implementatino of a sample
  for (size_t i = 0; i < num_mini_batches; i++)
  {
    size_t j = nn_rand_range(0.0, (double)(total_data_points - 1));
    new_expected_inputs[i] = td->expected_inputs[j];
    memcpy(new_expected_inputs + i * td->num_inputs, td->expected_inputs + j * td->num_inputs, td->num_inputs * sizeof(double));
    memcpy(new_expected_outputs + i * td->num_outputs, td->expected_outputs + j * td->num_outputs, td->num_outputs * sizeof(double));
  }

  td_sgd->expected_inputs = new_expected_inputs;
  td_sgd->expected_outputs = new_expected_outputs;
}

void nn_fit_sgd(nn_arena_t* arena, nn_network_t *net, nn_training_data *td, size_t num_mini_batches, nn_cost_func_impl_t cost_func, double delta, double rate, size_t num_iters, double clamp_min, double clamp_max)
{

  // randomize weights and biases
  nn_randomize_net(net, clamp_min, clamp_max);

  for (size_t i = 0; i < num_iters; i++)
  {
    nn_training_data td_new;
    nn_create_sgd_traning_data(arena, td, &td_new, num_mini_batches);
    nn_optimize_iter(net, &td_new, cost_func, delta, rate, clamp_min, clamp_max);
  }
}

#endif
#ifndef NEURAL_NET_WRAPPER_H
#define NEURAL_NET_WRAPPER_H

#include "stddef.h"
#include "assert.h"
#include "memory.h"
#include "stdlib.h"

#include "neural_net.h"
#include "arena.h"

void nn_create_layer(nn_arena_t *arena, nn_layer_t *created_layer, size_t num_prev_nodes, size_t num_nodes, nn_activation_func_t activ_func)
{
  nn_node_t *nodes_list = (nn_node_t *)nn_arena_alloc(arena, num_nodes * sizeof(nn_node_t));
  for (size_t i = 0; i < num_nodes; i++)
  {
    nn_node_t node;
    double *node_weights = (double *)nn_arena_alloc(arena, num_prev_nodes * sizeof(double));
    memset(node_weights, 0, num_prev_nodes * sizeof(double));
    node.weights = node_weights;
    node.num_weights = num_prev_nodes;
    node.bias = 0;
    nodes_list[i] = node;
  }

  created_layer->nodes = nodes_list;
  created_layer->num_nodes = num_nodes;
  created_layer->activ_func = activ_func;
}

void nn_push_layer(nn_network_t *net, size_t num_allocated_layers, nn_layer_t *layer)
{
  assert(net->num_layers < num_allocated_layers && "not enough allocated layers");
  net->num_layers++;
  net->layers[net->num_layers - 1] = *layer;
}

void nn_add_layer(nn_arena_t *arena, nn_network_t *net, size_t num_allocated_layers, size_t num_nodes, nn_activation_func_t activ_func)
{
  size_t num_prev_nodes = 0;
  if (net->num_layers == 0)
  {
    num_prev_nodes = num_nodes;
  }
  else
  {
    nn_layer_t *last_layer = nn_get_last_layer(net);
    num_prev_nodes = last_layer->num_nodes;
  }
  nn_layer_t *created_layer = (nn_layer_t*)nn_arena_alloc(arena, sizeof(nn_layer_t));
  nn_create_layer(arena, created_layer, num_prev_nodes, num_nodes, activ_func);
  nn_push_layer(net, num_allocated_layers, created_layer);
}

void nn_create_y_vs_x_plot(nn_network_t *net, char *str_buf, size_t str_buf_size, double x_min, double x_max, size_t num_points)
{
  char *current_line = str_buf;
  for (size_t i = 0; i < num_points; i++)
  {
    double x = map_range((double)i, 0.0, (double)num_points, x_min, x_max);
    double y = -1;
    nn_predict(net, &x, 1, &y, 1);

    size_t tot_chars_written = (current_line - str_buf);
    assert(tot_chars_written < str_buf_size);
    size_t chars_left_in_str_buf = (str_buf_size - (current_line - str_buf));
    int chars_written = snprintf(current_line, chars_left_in_str_buf, "%g %g\n ", x, y);
    assert(chars_written > 0);
    current_line += chars_written - 1;
  }
}

#endif
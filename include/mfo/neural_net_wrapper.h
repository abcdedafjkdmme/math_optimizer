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

void nn_push_layer_dyn(nn_network_t *net, nn_layer_t *layer)
{
  // first time pushing
  if(net->num_layers == 0){
    size_t new_capacity = 1;
    net->layer_capacity = new_capacity;
    net->layers = malloc(new_capacity * sizeof(nn_layer_t));
  }
  if (net->num_layers >= net->layer_capacity)
  {
    size_t new_capacity =  2 * net->layer_capacity;
    net->layer_capacity = new_capacity;
    net->layers = realloc(net->layers, new_capacity * sizeof(nn_layer_t));
  }
  net->num_layers++;
  net->layers[net->num_layers - 1] = *layer;
}


void nn_add_layer(nn_arena_t *arena, nn_network_t *net, size_t num_nodes, nn_activation_func_t activ_func)
{
  size_t num_prev_nodes = 0;
  // first time adding layer
  if (net->num_layers == 0)
  {
    num_prev_nodes = num_nodes;
  }
  else
  {
    nn_layer_t *last_layer = nn_get_last_layer(net);
    num_prev_nodes = last_layer->num_nodes;
  }
  nn_layer_t *created_layer = (nn_layer_t *)nn_arena_alloc(arena, sizeof(nn_layer_t));
  nn_create_layer(arena, created_layer, num_prev_nodes, num_nodes, activ_func);
  nn_push_layer_dyn(net, created_layer);
}

void nn_network_destroy(nn_network_t *net){
  free(net->layers); // only layers are dynamically allocated so they need to be freeed
}
#endif
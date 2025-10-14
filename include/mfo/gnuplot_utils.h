#ifndef GNUPLOT_UTILS_H
#define GNUPLOT_UTILS_H

#include <stddef.h>
#include <stdio.h>
#include <assert.h>

#include "arena.h"

void plt_create_plot_data(char *str_buf, size_t str_buf_size, double *x_var, double *y_var, size_t num_points)
{
  char *current_line = str_buf;
  for (size_t i = 0; i < num_points; i++)
  {
    double x = x_var[i];
    double y = y_var[i];
    size_t tot_chars_written = (size_t)(current_line - str_buf);
    assert(tot_chars_written < str_buf_size && "not enough memory in string buffer. increase the size of buffer");
    size_t chars_left_in_str_buf = (str_buf_size - tot_chars_written);
    int chars_written = snprintf(current_line, chars_left_in_str_buf, "%g %g\n ", x, y);
    assert(chars_written > 0);
    current_line += chars_written - 1;
  }
}

/**
 * @warning Only works with one input and one output
 */
void nn_create_y_vs_x_plot(nn_arena_t *arena, nn_network_t *net, char *str_buf, size_t str_buf_size, double x_min, double x_max, size_t num_points)
{
  size_t num_inputs = net->layers[0].num_nodes;
  assert(num_inputs == 1);
  size_t num_outputs = net->layers[net->num_layers - 1].num_nodes;
  assert(num_outputs == 1);

  double *x_vals = nn_arena_alloc(arena, num_points * sizeof(double));
  double *y_vals = nn_arena_alloc(arena, num_points * sizeof(double));

  for (size_t i = 0; i < num_points; i++)
  {
    double x = map_range((double)i, 0.0, (double)num_points, x_min, x_max);
    double y = -1;
    nn_predict(net, &x, 1, &y, 1);

    x_vals[i] = x;
    y_vals[i] = y;
  }

  plt_create_plot_data(str_buf, str_buf_size, x_vals, y_vals, num_points);

  // char *current_line = str_buf;
  // for (size_t i = 0; i < num_points; i++)
  //{
  //   double x = map_range((double)i, 0.0, (double)num_points, x_min, x_max);
  //   double y = -1;
  //   nn_predict(net, &x, 1, &y, 1);
  //
  //  size_t tot_chars_written = (current_line - str_buf);
  //  assert(tot_chars_written < str_buf_size);
  //  size_t chars_left_in_str_buf = (str_buf_size - (current_line - str_buf));
  //  int chars_written = snprintf(current_line, chars_left_in_str_buf, "%g %g\n ", x, y);
  //  assert(chars_written > 0);
  //  current_line += chars_written - 1;
  //}
}

#endif
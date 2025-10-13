#ifndef ARENA_H
#define ARENA_H

#include "stdlib.h"
#include "assert.h"

typedef struct
{
  size_t base;
  size_t current;
  size_t size;
} nn_arena_t;

void nn_arena_create(nn_arena_t *arena, size_t tot_size)
{
  arena->base = (size_t)malloc(tot_size);
  arena->current = arena->base;
  arena->size = tot_size;
}
void nn_arena_destroy(nn_arena_t *arena)
{
  free((void *)arena->base);
  arena->base = 0;
  arena->current = 0;
}

void *nn_arena_alloc(nn_arena_t *arena, size_t size)
{
  size_t old = arena->current;
  arena->current += size;
  assert(arena->current < arena->base + arena->size && "not enough memory in the arena");
  return (void *)old;
}


#endif
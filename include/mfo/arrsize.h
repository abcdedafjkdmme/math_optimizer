#ifndef ARRSIZE_H
#define ARRSIZE_H

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define ARRAY_SIZE_ROWS(x) (sizeof(x) / sizeof(x[0]))
#define ARRAY_SIZE_COLS(x) (sizeof(x[0]) / sizeof(x[0][0]))

#endif
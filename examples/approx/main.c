#include "assert.h"
#include "math.h"
#include "stdio.h"

#include "arrsize.h"
#include "approximations.h"

int main(){
  // universal approximator test
  double x = 1;
  double coffs[] = {1, 1.0, 1.0 / 2.0, 1.0 / 6.0};
  double y = poly_approx(x, coffs, ARRAY_SIZE(coffs));
  printf("universal approximator: (x,y) = (%lf, %lf) \n", x, y);
  return 0;
}
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"
#include "baseline.h"
#include "shared_mem.h"

int main(int argc, char **argv) {
  //run_baseline();
  return run_shared_mem();
}

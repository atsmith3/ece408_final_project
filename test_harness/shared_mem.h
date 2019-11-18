#ifndef SHARED_MEM_H
#define SHARED_MEM_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"

void forward_shared_mem(Tensor &y, const Tensor &x, const Tensor &w);
int run_shared_mem();

#endif

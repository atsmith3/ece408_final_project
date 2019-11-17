#ifndef BASELINE_H
#define BASELINE_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"

void forward_baseline(Tensor &y, const Tensor &x, const Tensor &w);
void generate_random();
int create_golden();

#endif

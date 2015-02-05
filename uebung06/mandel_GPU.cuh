/*
* mandel_GPU.cuh
*
*  Created on: 17.12.2014
*      Author: raub
*/

#ifndef MANDEL_GPU_CUH_
#define MANDEL_GPU_CUH_

#include "complexRange.h"

void render(uchar4*, CRange, unsigned int);
void calcGrid(dim3);

#endif /* MANDEL_GPU_CUH_ */

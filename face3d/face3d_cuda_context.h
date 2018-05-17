#pragma once

#ifdef FACE3D_USE_CUDA

namespace face3d {

void set_cuda_device(int cuda_device);
int get_cuda_device();


}
#endif

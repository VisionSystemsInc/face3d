#include "face3d_cuda_context.h"

#ifdef FACE3D_USE_CUDA

namespace face3d_cuda {

int cuda_device_ = 0;

}

void face3d::set_cuda_device(int cuda_device)
{
  face3d_cuda::cuda_device_ = cuda_device;
}

int face3d::get_cuda_device()
{
  return face3d_cuda::cuda_device_;
}

#endif

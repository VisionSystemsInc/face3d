#include <stdio.h>
#include <cuda_runtime.h>
#include "triangle_mesh.h"

__global__ void localize_points(float* V, float* d_query_points, float* d_query_points_xy,
                                float* d_locs, float* d_dists, int nvertices, int nquery_points,
                                int query_points_size, int query_points_xy_size, int threads_per_block) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  float min_dist = 100;
  float minx, miny;
  float x, y, z, r, g, b, dist;
  int start, end;
  int shared_arr_size = 1024;
  __shared__ float l_query_points[1024*3];

  x = V[3*index];
  y = V[3*index+1];
  z = V[3*index+2];

  for (int i = 0; i < nquery_points; i += shared_arr_size) {
    start = ceilf(shared_arr_size/(threads_per_block*1.0))*threadIdx.x;
    end = min(ceilf(shared_arr_size/(threads_per_block*1.0))*(threadIdx.x+1), float(shared_arr_size));

    for (int j = start; j < end && i+j < nquery_points; j++) {
      l_query_points[j*3] = d_query_points[(i+j)*3];
      l_query_points[j*3+1] = d_query_points[(i+j)*3+1];
      l_query_points[j*3+2] = d_query_points[(i+j)*3+2];
    }
    __syncthreads();

    for (int j = 0; j < shared_arr_size && i+j < nquery_points; j++) {
      r = l_query_points[j*3];
      g = l_query_points[j*3+1];
      b = l_query_points[j*3+2];
      dist = (x-r)*(x-r) + (y-g)*(y-g) + (z-b)*(z-b);
      if (dist < min_dist) {
        min_dist = dist;
        minx = d_query_points_xy[(i+j)*2];
        miny = d_query_points_xy[(i+j)*2+1];
      }
    }
    __syncthreads();
  }

  if (index < nvertices) {
    d_locs[2*index] = minx;
    d_locs[2*index+1] = miny;
    d_dists[index] = min_dist;
  }
  return;
}

namespace face3d {
  void localize_points_cuda(MatrixXfRowMajor& query_points,
                            MatrixXfRowMajor& query_points_xy,
                            MatrixXfRowMajor& V,
                            MatrixXfRowMajor& locs_eigen,
                            MatrixXfRowMajor& dists,
                            int cuda_device
                            ) {

    int V_size = V.rows()*V.cols()*sizeof(float);
    int query_points_size = query_points.rows()*query_points.cols()*sizeof(float);
    int query_points_xy_size = query_points_xy.rows()*query_points_xy.cols()*sizeof(float);
    int locs_size = locs_eigen.rows()*locs_eigen.cols()*sizeof(float);
    int dists_size = locs_eigen.rows()*sizeof(float);

    if(cudaSetDevice(cuda_device) != cudaSuccess) {
      std::cerr << "ERROR setting cuda device to " << cuda_device << std::endl;
      throw std::runtime_error("cudaSetDevice returned error");
    }

    float *d_V, *d_query_points, *d_query_points_xy, *d_locs, *d_dists;

    cudaMalloc((void **)&d_V, V_size);
    cudaMalloc((void **)&d_query_points, query_points_size);
    cudaMalloc((void **)&d_query_points_xy, query_points_xy_size);
    cudaMalloc((void **)&d_locs, locs_size);
    cudaMalloc((void **)&d_dists, dists_size);

    cudaMemcpy(d_V, V.data(), V_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_points, query_points.data(), query_points_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_points_xy, query_points_xy.data(), query_points_xy_size, cudaMemcpyHostToDevice);

    cudaMemset(d_locs, 0, locs_size);
    cudaMemset(d_dists, 0, dists_size);

    int nvertices = V.rows();
    int nquery_points = query_points.rows();

    int threads_per_block = 256;
    int blocks_per_grid = (nvertices + threads_per_block - 1) / threads_per_block;

    localize_points<<<blocks_per_grid, threads_per_block>>> (d_V, d_query_points, d_query_points_xy,
                                                             d_locs, d_dists, nvertices, nquery_points,
                                                             query_points_size, query_points_xy_size, threads_per_block);

    cudaDeviceSynchronize();

    cudaMemcpy(locs_eigen.data(), d_locs, locs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dists.data(), d_dists, dists_size, cudaMemcpyDeviceToHost);

    cudaFree(d_V); cudaFree(d_locs);
    cudaFree(d_query_points); cudaFree(d_query_points_xy);
    cudaFree(d_dists);
  }
}

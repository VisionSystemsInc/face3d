#pragma once

#include <vector>

namespace face3d {

struct estimation_results_t
{
  bool success_;
  std::vector<bool> image_success_;
  std::vector<int> vertices_found_;

  estimation_results_t(int num_images) :
    success_(false),
    image_success_(num_images, false),
    vertices_found_(num_images, 0) {}

  explicit operator bool() const { return success_; }
};
}

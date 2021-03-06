#pragma once

#include <vector>
#include <stdexcept>

#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_point_2d.h>

#include <face3d_basic/ortho_camera_parameters.h>
#include <face3d_basic/perspective_camera_parameters.h>
#include <face3d_basic/subject_sighting_coefficients.h>
#include <face3d/head_mesh.h>

#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_transforms.h>

#include "triangle_mesh.h"

namespace face3d
{

namespace camera_estimation {

bool compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                           std::vector<vgl_point_3d<double> > const& pts3d,
                           int nx, int ny,
                           ortho_camera_parameters<double> &cam_params);

// The semantics of this function are nonsensical, but it needs to be defined to
// satisfy the compiler due to the optional focal_length specification in the
// coefficient estimator.
// TODO: refactor so this hack is not necessary.
bool compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                           std::vector<vgl_point_3d<double> > const& pts3d,
                           int nx, int ny,
                           double focal_len,
                           vgl_point_2d<double> principal_pt,
                           ortho_camera_parameters<double> &cam_params);

bool compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                           std::vector<vgl_point_3d<double> > const& pts3d,
                           int nx, int ny,
                           perspective_camera_parameters<double> &cam_params);

bool compute_camera_params(std::vector<vgl_point_2d<double> > const& pts2d,
                           std::vector<vgl_point_3d<double> > const& pts3d,
                           int nx, int ny,
                           double focal_len,
                           vgl_point_2d<double> principal_pt,
                           perspective_camera_parameters<double> &cam_params);

template<class T, class CAM_T>
bool compute_camera_params_from_pncc_and_offsets(dlib::array2d<vgl_point_3d<T> > const& PNCC,
                                                 dlib::array2d<vgl_vector_3d<T> > const& offsets,
                                                 CAM_T &cam_params);

// function object for computing perspective camera with known focal length and principal pt
class compute_camera_params_calibrated
{
public:
  compute_camera_params_calibrated(double focal_len,
                                   vgl_point_2d<double> const& principal_pt)
    : focal_len_(focal_len), principal_pt_(principal_pt)
  {}

  bool operator()(std::vector<vgl_point_2d<double> > const& pts2d,
                  std::vector<vgl_point_3d<double> > const& pts3d,
                  int nx, int ny,
                  perspective_camera_parameters<double> &cam_params);
private:
  double focal_len_;
  vgl_point_2d<double> principal_pt_;
};

void compute_camera_params_bundle_adjust(std::vector<std::map<int, vgl_point_2d<double>>> const& vertex_img_locs,
                                         vnl_matrix<double> const& subject_pca_components,
                                         vnl_matrix<double> const& expression_pca_components,
                                         face3d::head_mesh &base_mesh,
                                         std::vector<face3d::perspective_camera_parameters<double>> const& init_cameras,
                                         std::vector<vgl_point_3d<double>> &points_out,
                                         face3d::subject_sighting_coefficients<face3d::perspective_camera_parameters<double>> &coeffs_out);

void compute_camera_params_bundle_adjust(std::vector<dlib::array2d<vgl_point_3d<float>>> const& PNCCs,
                                         vnl_matrix<double> const& subject_pca_components,
                                         vnl_matrix<double> const& expression_pca_components,
                                         face3d::head_mesh &base_mesh,
                                         std::vector<face3d::perspective_camera_parameters<double>> const& init_cameras,
                                         std::vector<vgl_point_3d<double>> &points_out,
                                         face3d::subject_sighting_coefficients<face3d::perspective_camera_parameters<double>> &coeffs_out);

// provide a function template for convenience. Compiler will prefer
// non-templates until template arg is explicitly given.  Needed by
// media_coefficient_estimator.
template<class CAM_T>
bool compute_camera_params_generic(std::vector<vgl_point_2d<double> > const& pts2d,
                                   std::vector<vgl_point_3d<double> > const& pts3d,
                                   int nx, int ny,
                                   CAM_T &cam_params)
{
  return compute_camera_params(pts2d, pts3d, nx, ny, cam_params);
}

// Implementation of Templated methods
template<class T, class CAM_T>
bool
compute_camera_params_from_pncc_and_offsets(dlib::array2d<vgl_point_3d<T> > const& PNCC,
                                            dlib::array2d<vgl_vector_3d<T> > const& offsets,
                                            CAM_T &cam_params)
{
  const int nx = PNCC.nc();
  const int ny = PNCC.nr();
  if ( (offsets.nc() != nx) || (offsets.nr() != ny) ) {
    std::cerr << "ERROR: compute_camera_params(): image sizes of PNCC and offset images do not match" << std::endl;
    std::cerr << "PNCC:    " << nx << " x " << ny << std::endl;
    std::cerr << "Offsets: " << offsets.nc() << " x " << offsets.nr() << std::endl;
    throw std::logic_error("Invalid inputs to compute_camera_params()");
  }
  // invalid points are represented as (0,0,0)
  const vgl_point_3d<T> invalid_loc(0,0,0);
  const T squared_len_thresh = 10.0*10.0;
  dlib::array2d<unsigned char> mask(ny, nx);
  for (int yi=0; yi<ny; ++yi) {
    for (int xi=0; xi<nx; ++xi) {
      if (std::isnan(PNCC[yi][xi].x())) {
        mask[yi][xi] = dlib::off_pixel;
      }
      else {
        if ((PNCC[yi][xi] - invalid_loc).sqr_length() < squared_len_thresh) {
          mask[yi][xi] = dlib::off_pixel;
        }
        else {
          mask[yi][xi] = dlib::on_pixel;
        }
      }
    }
  }
  // erode the mask to avoid problematic areas around edges
  dlib::array2d<unsigned char> mask_eroded(ny,nx);
  static const int strel_size = 3;
  unsigned char strel[strel_size][strel_size];
  for (int yi=0; yi<strel_size; ++yi) {
    for (int xi=0; xi<strel_size; ++xi) {
      strel[yi][xi] = dlib::on_pixel;
    }
  }
  dlib::binary_erosion(mask, mask_eroded, strel);

  std::vector<vgl_point_2d<double> > pts2d;
  std::vector<vgl_point_3d<double> > pts3d;
  for (int yi=0; yi<ny; ++yi) {
    for (int xi=0; xi<nx; ++xi) {
      if (mask_eroded[yi][xi]) {
        pts2d.push_back(vgl_point_2d<double>(xi,yi));
        vgl_point_3d<T> pt3d = PNCC[yi][xi] + offsets[yi][xi];
        pts3d.push_back(vgl_point_3d<double>(pt3d.x(), pt3d.y(), pt3d.z()));
      }
    }
  }
  return compute_camera_params(pts2d, pts3d, nx, ny, cam_params);
}

}
}

#include "offset_correction.h"
#include "camera_estimation.h"
#include <dlib/array2d.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <vgl/vgl_ray_3d.h>
#include <vgl/vgl_homg_point_2d.h>
#include <face3d_basic/ortho_camera_parameters.h>

namespace face3d {

template<class TIN, class TOUT>
void convert_3d_type(TIN const& pin, TOUT &pout) {
  pout = TOUT(pin.x(), pin.y(), pin.z());
}


bool correct_offsets(dlib::array2d<vgl_point_3d<float> > const& PNCC,
                     dlib::array2d<vgl_vector_3d<float> > const& offsets,
                     dlib::array2d<vgl_vector_3d<float> > &offsets_out)
{
  using CAM_T = ortho_camera_parameters<double>;
  //using CAM_T = perspective_camera_parameters<double>;
  const int nx = PNCC.nc();
  const int ny = PNCC.nr();
  if ((offsets.nr() != ny) || (offsets.nc() != nx)) {
    std::cerr << "Error: PNCC and offsets image sizes do not match" << std::endl;
    return false;
  }
  dlib::array2d<unsigned char> valid_mask(ny,nx);
  dlib::assign_all_pixels(valid_mask, dlib::off_pixel);

  const vgl_point_3d<float> origin(0,0,0);
  const float mag_sqrd_thresh = 50.0f*50.0f;
  for (int yi=0; yi<ny; ++yi) {
    for (int xi=0; xi<nx; ++xi) {
      // threshold distance of PNCC from 0
      float mag_sqrd = (PNCC[yi][xi] - origin).sqr_length();
      if (mag_sqrd > mag_sqrd_thresh) {
        valid_mask[yi][xi] = dlib::on_pixel;
      }
    }
  }
  // erode the mask to avoid problematic areas around edges
  dlib::array2d<unsigned char> valid_mask_eroded(ny,nx);
  static const int strel_size = 5;
  unsigned char strel[strel_size][strel_size];
  for (int y=0; y<strel_size; ++y) {
    for (int x=0; x<strel_size; ++x) {
      strel[y][x] = dlib::on_pixel;
    }
  }
  dlib::binary_erosion(valid_mask, valid_mask_eroded, strel);

  // extract valid 3d/2d pairs
  std::vector<vgl_point_2d<double> > pts2d;
  std::vector<vgl_point_3d<double> > pts3d;
  for (int yi=0; yi<ny; ++yi) {
    for (int xi=0; xi<nx; ++xi) {
      if (valid_mask_eroded[yi][xi] == dlib::on_pixel) {
        vgl_point_3d<double> p3d;
        convert_3d_type(PNCC[yi][xi] + offsets[yi][xi], p3d);
        vgl_point_2d<double> p2d(xi,yi);
        pts2d.push_back(p2d);
        pts3d.push_back(p3d);
      }
    }
  }

  // estimate camera
  CAM_T cam_params;
  camera_estimation::compute_camera_params(pts2d, pts3d, nx, ny, cam_params);
  auto cam = cam_params.to_camera();

  // enforce 3d points on camera rays
  const vgl_vector_3d<float> zero_offset(0,0,0);
  offsets_out.set_size(ny,nx);
  const double correction_thresh_sqrd = 30.0*30.0;
  for (int yi=0; yi<ny; ++yi) {
    for (int xi=0; xi<nx; ++xi) {
      if (valid_mask[yi][xi] == dlib::off_pixel) {
        offsets_out[yi][xi] = zero_offset;
        continue;
      }
      vgl_ray_3d<double> ray = cam.backproject_ray(vgl_homg_point_2d<double>(xi,yi));
      vgl_point_3d<double> p3d;
      convert_3d_type(PNCC[yi][xi] + offsets[yi][xi], p3d);
      vgl_vector_3d<double> v = p3d - ray.origin();
      double dist = dot_product(v,ray.direction());
      vgl_point_3d<double> p3d_on_ray = ray.origin() + ray.direction()*dist;
      if ((p3d_on_ray - p3d).sqr_length() <= correction_thresh_sqrd) {
        vgl_point_3d<float> p3d_on_rayf;
        convert_3d_type(p3d_on_ray, p3d_on_rayf);
        offsets_out[yi][xi] = p3d_on_rayf - PNCC[yi][xi];
      }
    }
  }
  return true;
}

}

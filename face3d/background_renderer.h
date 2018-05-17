#pragma once
#include <vector>
#include <algorithm>
#include <face3d_basic/sighting_coefficients.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/geometry.h>
#include <dlib/image_transforms.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/algo/vgl_rotation_3d.h>
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_vector_fixed.h>
#include <vpgl/vpgl_proj_camera.h>

namespace face3d {

template<class CAM_T>
class background_renderer {
public:
  // Creates a new background_renderer object. The lifetimes of images and coeffs must
  // exceed that of the background_renderer object, because references are stored.
  background_renderer(std::vector<dlib::array2d<dlib::rgb_pixel> > const& images,
                      std::vector<sighting_coefficients<CAM_T> > const& coeffs);

  template<class CAM_OUT_T>
    dlib::array2d<dlib::rgb_pixel> render(sighting_coefficients<CAM_OUT_T> const& coeffs);

private:
  std::vector<dlib::array2d<dlib::rgb_pixel> > images_;
  std::vector<sighting_coefficients<CAM_T> > coeffs_;

  // helper functions
  void composite_image(dlib::array2d<dlib::rgb_pixel> &dest,
                       dlib::array2d<dlib::rgb_pixel> const& src,
                       float weight);
};


// ---  Member Function Definitions --- //

template<class CAM_T>
background_renderer<CAM_T>::
background_renderer(std::vector<dlib::array2d<dlib::rgb_pixel> > const& images,
                    std::vector<sighting_coefficients<CAM_T> > const& coeffs)
  : images_(images.size()), coeffs_(coeffs)
{
  if (images_.size() != coeffs_.size()) {
    std::cerr << "ERROR: images_.size() == " << images_.size() << ", coeffs_size() = " << coeffs_.size() << std::endl; throw std::logic_error("Different numbers of images and coefficients");
  }
  // deep copy images
  for (int i=0; i<images.size(); ++i) {
    dlib::assign_image(images_[i], images[i]);
  }
}

template<class CAM_T>
template<class CAM_OUT_T>
dlib::array2d<dlib::rgb_pixel> background_renderer<CAM_T>::
render(sighting_coefficients<CAM_OUT_T> const& coeffs_out)
{
  if (images_.size() != coeffs_.size()) {
    std::cerr << "ERROR: render(): images_.size() == " << images_.size() << ", coeffs_size() = " << coeffs_.size() << std::endl;
    throw std::logic_error("Different numbers of images and coefficients");
  }
  const int num_input_images = images_.size();
  const int nx = coeffs_out.camera().nx();
  const int ny = coeffs_out.camera().ny();

  vnl_matrix_fixed<double,3,3> R_render = coeffs_out.camera().rotation().as_matrix();
  vnl_vector_fixed<double,3> render_camz_vnl = R_render.get_row(2);
  vgl_vector_3d<double> render_cam_z(render_camz_vnl[0], render_camz_vnl[1], render_camz_vnl[2]);
  // compute weights for each of the input images
  std::vector<vgl_vector_3d<double> > cam_zs;
  std::vector<vnl_matrix_fixed<double,3,4> > projection_matrices;
  std::vector<double> image_weights;
  double weight_sum = 0;
  const double softmax_weight = 40.0;
  for (int i=0; i<num_input_images; ++i) {
    CAM_T camera_params(coeffs_[i].camera());
    vnl_matrix_fixed<double,3,4> P = camera_params.to_camera().get_matrix();
    vnl_matrix_fixed<double,3,3> R_og = camera_params.rotation().as_matrix();
    if (std::signbit(R_og(2,0)) != std::signbit(R_render(2,0))) {
      // mirror the original image
      vnl_matrix_fixed<double,3,3> flip_x_3x3(0.0);
      vnl_matrix_fixed<double,4,4> flip_x_4x4(0.0);
      flip_x_3x3.set_diagonal(vnl_vector_fixed<double,3>(-1,1,1));
      flip_x_4x4.set_diagonal(vnl_vector_fixed<double,4>(-1,1,1,1));
      R_og *= flip_x_3x3;
      P *= flip_x_4x4;
    }
    projection_matrices.push_back(P);
    vnl_vector_fixed<double,3> cam_z_vnl = R_og.get_row(2);
    vgl_vector_3d<double> cam_z(cam_z_vnl[0], cam_z_vnl[1], cam_z_vnl[2]);
    cam_zs.push_back(cam_z);

    double dp = dot_product(cam_z, render_cam_z);
    double weight = 0;
    if (dp > 0) {
      weight = std::exp(softmax_weight * dp);
    }
    image_weights.push_back(weight);
    weight_sum += weight;
  }
  // add constant term to weight_sum to create fade to black if no nearby images
  weight_sum += std::exp(softmax_weight*0.2);

  for (double &w : image_weights) {
    // normalize to complete softmax computation
    w /= weight_sum;
  }

  auto cam_dest = coeffs_out.camera().to_camera();
  // compute homographies for images with non-zero weights
  std::vector<dlib::point_transform_affine> transforms(num_input_images);
  for (int i=0; i<num_input_images; ++i) {
    if (image_weights[i] <= 0) {
      // don't bother
      continue;
    }
    vpgl_proj_camera<double> cam_src(projection_matrices[i]);
    vgl_point_3d<double> head_center(0,0,0);
    double center_offset = 0.0; // pos -> away from camera
    vgl_vector_3d<double> up(0,1,0);
    vgl_point_3d<double> plane_center = head_center + center_offset*cam_zs[i];
    vgl_vector_3d<double> plane_x = cross_product(cam_zs[i], up);
    vgl_point_3d<double> p1 = plane_center + 100*up;
    vgl_point_3d<double> p2 = plane_center + 100*plane_x;
    vgl_point_3d<double> p3 = plane_center - 100*plane_x;
    std::vector<vgl_point_3d<double> > pts3d {p1, p2, p3};
    std::vector<dlib::vector<float,2> > pts_src;
    std::vector<dlib::vector<float,2> > pts_warped;
    for (vgl_point_3d<double> const& pt3d : pts3d) {
      vgl_point_2d<double> pt2d_src(cam_src.project(pt3d));
      vgl_point_2d<double> pt2d_warped(cam_dest.project(pt3d));
      pts_src.push_back(dlib::vector<float,2>(pt2d_src.x(), pt2d_src.y()));
      pts_warped.push_back(dlib::vector<float,2>(pt2d_warped.x(), pt2d_warped.y()));
    }
    transforms[i] = dlib::find_affine_transform(pts_warped, pts_src);
  }

  // warp images
  dlib::array2d<dlib::rgb_pixel> warped_out(ny,nx);
  dlib::assign_all_pixels(warped_out, dlib::rgb_pixel(0,0,0));
  for (int i=0; i<num_input_images; ++i) {
    if (image_weights[i] > 0) {
      dlib::array2d<dlib::rgb_pixel> img_warped(ny,nx);
      dlib::transform_image(images_[i], img_warped,
                            dlib::interpolate_bilinear(),
                            transforms[i], dlib::black_background(),
                            dlib::get_rect(img_warped));

      composite_image(warped_out, img_warped, image_weights[i]);
    }
  }
  return warped_out;
}

template<class CAM_T>
void background_renderer<CAM_T>::
composite_image(dlib::array2d<dlib::rgb_pixel> &dest,
                dlib::array2d<dlib::rgb_pixel> const& src,
                float weight)
{
  const int nx = dest.nc();
  const int ny = dest.nr();
  if ((src.nc() != nx) || (src.nr() != ny)) {
    throw std::logic_error("Sizes of dest and src images do not match");
  }
  for (int y=0; y<ny; ++y) {
    for (int x=0; x<nx; ++x) {
      dlib::rgb_pixel &pix_out = dest[y][x];
      dlib::rgb_pixel pix_src = src[y][x];
      pix_out.red = static_cast<unsigned char>(std::min(255.0f, pix_out.red + weight*pix_src.red));
      pix_out.green = static_cast<unsigned char>(std::min(255.0f, pix_out.green + weight*pix_src.green));
      pix_out.blue = static_cast<unsigned char>(std::min(255.0f, pix_out.blue + weight*pix_src.blue));
    }
  }
  return;
}

}

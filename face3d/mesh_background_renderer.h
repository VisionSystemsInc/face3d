#pragma once
#include <string>
#include <vector>
#include <limits>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <vgl/vgl_point_2d.h>
#include <vgl/vgl_point_3d.h>
#include <vgl/vgl_homg_point_2d.h>
#include <face3d_basic/sighting_coefficients.h>
#include <face3d_basic/io_utils.h>
#include <face3d/head_mesh.h>
#include <face3d/mesh_background_renderer_agnostic.h>
#include <face3d/mesh_renderer.h>
#include <igl/triangle/triangulate.h>

namespace face3d {
using TEX_T = dlib::array2d<dlib::rgb_alpha_pixel>;
template <class CAM_T, class IMG_T>
class mesh_background_renderer
{
public:
  mesh_background_renderer(IMG_T const& img,
                           sighting_coefficients<CAM_T> const& coeffs,
                           head_mesh const& mesh,
                           vnl_matrix<double> const& subject_components,
                           vnl_matrix<double> const& expression_components,
                           mesh_renderer &renderer);

  mesh_background_renderer(std::vector<IMG_T> const& imgs,
                           std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                           head_mesh const& mesh,
                           vnl_matrix<double> const & subject_components,
                           vnl_matrix<double> const & expression_components,
                           mesh_renderer &renderer);



  template<class CAM_T_OUT>
  void render(sighting_coefficients<CAM_T_OUT> const& render_coeffs,
              dlib::array2d<dlib::rgb_pixel> &background);

  void set_debug_dir(std::string const& debug_dir) { debug_mode_ = true; debug_dir_ = debug_dir; }

private:

  std::vector<dlib::array2d<dlib::rgb_pixel> > images_;
  std::vector<sighting_coefficients<CAM_T> > coeffs_;
  mesh_background_renderer_agnostic<CAM_T, IMG_T> agnostic_background_renderer_;
  head_mesh mesh_;
  mesh_renderer &renderer_;
  vnl_matrix<double> const & subject_components_;
  vnl_matrix<double> const & expression_components_;
  // the following members are set by initialize()
  bool debug_mode_;
  std::string debug_dir_;

  // helper functions
  void composite_image(dlib::array2d<dlib::rgb_pixel> &dest,
                       dlib::array2d<dlib::rgb_pixel> const& src,
                       float weight);
  void assign_border_locations(Eigen::MatrixXd &V, int border_begin, int nx, int ny,
                               int num_boundary_verts_per_side, bool flip_x);

  void adjust_border_locations(Eigen::MatrixXd &V, Eigen::MatrixXi const& F,
                               int border_begin, int nx, int ny, int num_boundary_verts_per_side);
};


template<class CAM_T, class IMG_T>
mesh_background_renderer<CAM_T, IMG_T>::
mesh_background_renderer(IMG_T const& img,
                         sighting_coefficients<CAM_T> const& coeffs,
                         head_mesh const& mesh,
                         vnl_matrix<double> const& subject_components,
                         vnl_matrix<double> const& expression_components,
                         mesh_renderer &renderer) :
  images_(1), coeffs_(1, coeffs), renderer_(renderer),
  mesh_(mesh), subject_components_(subject_components), expression_components_(expression_components),
  agnostic_background_renderer_({img}, coeffs, mesh, subject_components, expression_components, renderer),
  debug_mode_(false)
{
  // deep copy image

}

template<class CAM_T, class IMG_T>
mesh_background_renderer<CAM_T, IMG_T>::
mesh_background_renderer(std::vector<IMG_T> const& imgs,
                         std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                         head_mesh const& mesh,
                         vnl_matrix<double> const& subject_components,
                         vnl_matrix<double> const& expression_components,
                         mesh_renderer &renderer) :
  images_(imgs.size()), renderer_(renderer),
  mesh_(mesh), subject_components_(subject_components), expression_components_(expression_components),
  agnostic_background_renderer_(imgs, coeffs, mesh, subject_components, expression_components, renderer),
  debug_mode_(false)
{
  const int num_imgs = imgs.size();
  if (coeffs.size() != num_imgs) {
    throw std::logic_error("Different number of images and coefficients passed to mesh_background_renderer");
  }
  // deep copy images

}


template<class CAM_T, class IMG_T>
template<class CAM_T_OUT>
void mesh_background_renderer<CAM_T, IMG_T>::
render(sighting_coefficients<CAM_T_OUT> const& render_coeffs,
       dlib::array2d<dlib::rgb_pixel> &background){
  this->agnostic_background_renderer_.render(render_coeffs, this->subject_components_, this->expression_components_, background, this->mesh_);
}

}

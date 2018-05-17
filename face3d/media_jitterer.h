#pragma once
#include <face3d/mesh_renderer.h>
#include <face3d_basic/face3d_util.h>
#include <face3d/head_mesh.h>
#include <face3d/mesh_background_renderer.h>
#include <face3d_basic/subject_sighting_coefficients.h>
#include <face3d_basic/sighting_coefficients.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include "texture_map_operations.h"

namespace face3d {

template<class CAM_T, class IMG_T>
class media_jitterer{
public:
  media_jitterer(std::vector<IMG_T > const& images,
                 std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                 head_mesh const& base_mesh,
                 vnl_matrix<double> const& subject_pca_components,
                 vnl_matrix<double> const& expression_pca_components,
                 face3d::mesh_renderer &renderer,
                 std::string const& debug_dir="", bool complete_texture=true);

  // Constructor takes a pre-computed symmetry map
  media_jitterer(std::vector<IMG_T > const& images,
                 std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                 head_mesh const& base_mesh,
                 vnl_matrix<double> const& subject_pca_components,
                 vnl_matrix<double> const& expression_pca_components,
                 face3d::mesh_renderer &renderer,
                 dlib::array2d<dlib::vector<float,2> > const& tex_symmetry_map,
                 std::string const& debug_dir="", bool complete_texture=true);

template <class CAM_OUT_T, class IMG_OUT_T>
 bool render(subject_sighting_coefficients<CAM_OUT_T>& output_sightings,
             vnl_matrix<double> const& subject_pca_components,
             vnl_matrix<double> const& expression_pca_components,
             std::vector<IMG_OUT_T >  &jittered_images);

template <class CAM_OUT_T, class IMG_OUT_T>
bool render(CAM_OUT_T const& camera,
              vnl_vector<double> const& subject_coeffs,
              vnl_vector<double> const& expression_coeffs,
              vnl_matrix<double> const& subject_pca_components,
              vnl_matrix<double> const& expression_pca_components,
              IMG_OUT_T &render_img);

/**
 * If disabled, a "blank" texture will be rendered instead.
 * This is meant for debug/visualization purposes.
 **/
void enable_texture(bool tex_enabled);

/**
 * If enabled, a planar warp of the input images will be rendered
 * as a background of the 3-d model
 */
void enable_background(bool background_enabled) { composite_background_ = background_enabled; }


private:
  mesh_renderer &model_renderer_;
  head_mesh base_mesh_;
  mesh_background_renderer<CAM_T, IMG_T> bkgnd_renderer_;
  dlib::array2d<dlib::rgb_alpha_pixel> face_texture_;
  dlib::array2d<dlib::rgb_alpha_pixel> mouth_texture_;
  std::string debug_dir_;
  bool debug_mode_;
  bool composite_background_;
  bool complete_texture_;

  // code common to all constructors happens in init().
  void init(std::vector<IMG_T > const& images,
            std::vector<sighting_coefficients<CAM_T> > const& coeffs,
            vnl_matrix<double> const& subject_pca_components,
            vnl_matrix<double> const& expression_pca_components,
            dlib::array2d<dlib::vector<float,2> > const& face_sym_map);

  bool generate_texture(std::vector<IMG_T > const& images,
                        std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                        vnl_matrix<double> const& subject_pca_components,
                        vnl_matrix<double> const& expression_pca_components,
                        dlib::array2d<dlib::rgb_alpha_pixel> &face_tex,
                        dlib::array2d<dlib::rgb_alpha_pixel> &mouth_tex);
};


//------------  Function Definitions -----------//

template<class CAM_T, class IMG_T>
media_jitterer<CAM_T, IMG_T>::
media_jitterer(std::vector<IMG_T> const& images,
               std::vector<sighting_coefficients<CAM_T> > const& coeffs,
               head_mesh const& base_mesh,
               vnl_matrix<double> const& subject_pca_components,
               vnl_matrix<double> const& expression_pca_components,
               face3d::mesh_renderer &renderer,
               std::string const& debug_dir,
               bool complete_texture)
  : model_renderer_(renderer), base_mesh_(base_mesh),
  bkgnd_renderer_(images, coeffs, base_mesh,
                  subject_pca_components, expression_pca_components, renderer),
    debug_dir_(debug_dir), debug_mode_(debug_dir_ != ""), complete_texture_(complete_texture),
  composite_background_(true)
{
  // get texture size
  const int face_tex_nr = base_mesh_.face_mesh().texture().nr();
  const int face_tex_nc = base_mesh_.face_mesh().texture().nc();

  // fill in face texture via symmetry constraints
  dlib::array2d<dlib::vector<float,2> > face_sym_map;
  face3d::triangle_mesh const& face_mesh = base_mesh_.face_mesh();
  face3d::generate_symmetry_map(face_mesh.V(), face_mesh.F(), face_mesh.T(),
                                face_tex_nc, face_tex_nr, face_sym_map);

  init(images, coeffs, subject_pca_components, expression_pca_components, face_sym_map);
}

// Constructor takes a pre-computed symmetry map
template<class CAM_T, class IMG_T>
media_jitterer<CAM_T, IMG_T>::
media_jitterer(std::vector<IMG_T > const& images,
               std::vector<sighting_coefficients<CAM_T> > const& coeffs,
               head_mesh const& base_mesh,
               vnl_matrix<double> const& subject_pca_components,
               vnl_matrix<double> const& expression_pca_components,
               face3d::mesh_renderer &renderer,
               dlib::array2d<dlib::vector<float,2> > const& face_sym_map,
               std::string const& debug_dir, bool complete_texture)
: model_renderer_(renderer), base_mesh_(base_mesh),
  bkgnd_renderer_(images, coeffs, base_mesh,
                  subject_pca_components, expression_pca_components, renderer),
  debug_dir_(debug_dir), debug_mode_(debug_dir_ != ""), complete_texture_(complete_texture),
  composite_background_(true)
{
  init(images, coeffs, subject_pca_components, expression_pca_components, face_sym_map);
}

template<class CAM_T, class IMG_T>
void media_jitterer<CAM_T, IMG_T>::
init(std::vector<IMG_T > const& images,
     std::vector<sighting_coefficients<CAM_T> > const& coeffs,
     vnl_matrix<double> const& subject_pca_components,
     vnl_matrix<double> const& expression_pca_components,
     dlib::array2d<dlib::vector<float,2> > const& face_sym_map)
{
  // validate input
  if (images.size() != coeffs.size()) {
    throw std::logic_error("Different numbers of images and coefficients passed to media_jitterer");
  }
  if (debug_mode_) {
    for (int i=0; i<images.size(); ++i) {
      std::stringstream img_fname;
      img_fname << debug_dir_ << "/input_image_" << i << ".png";
      dlib::save_png(images[i],img_fname.str().c_str());
    }
    model_renderer_.set_debug_dir(debug_dir_);
    model_renderer_.set_debug_mode(debug_mode_);

    bkgnd_renderer_.set_debug_dir(debug_dir_);
  }
  // generate a texture map based on input sightings
  dlib::array2d<dlib::rgb_alpha_pixel> face_tex, mouth_tex;
  if(!generate_texture(images, coeffs, subject_pca_components, expression_pca_components,
                       face_tex, mouth_tex)) {
    throw std::runtime_error("Failed to generate texture map from input sightings");
  }
  if (complete_texture_)
    face3d::complete_texture(face_tex, face_sym_map, face_texture_);
  else
    face_texture_ = std::move(face_tex);

  if(debug_mode_) {
    if (complete_texture_){
      std::string dbg_tex_fname = debug_dir_ + "/face_texture_og.png";
      dlib::save_png(face_tex, dbg_tex_fname.c_str());
    }
    std::string dbg_tex_completed_fname = debug_dir_ + "/face_texture_completed.png";
    dlib::save_png(face_texture_, dbg_tex_completed_fname.c_str());
  }

  /** Note: The 3DMM-style meshes do not have a mouth mesh, so the following
   * code block will not usually be called any more. If we start using a new mesh
   * that does have a mouth mesh, and we need more speed, it will probably be worth
   * precomputing that symmetry map as well.
   */
  if (base_mesh_.has_mouth()) {
    // fill in mouth texture via symmetry constraints
    dlib::array2d<dlib::vector<float,2> > mouth_sym_map;
    face3d::triangle_mesh const& mouth_mesh = base_mesh_.mouth_mesh();
    face3d::generate_symmetry_map(mouth_mesh.V(), mouth_mesh.F(), mouth_mesh.T(),
                                  mouth_tex.nc(), mouth_tex.nr(), mouth_sym_map);
  if (complete_texture_)
    face3d::complete_texture(mouth_tex, mouth_sym_map, mouth_texture_);
  else
    mouth_texture_ = std::move(mouth_tex);

    if(debug_mode_) {
      if (complete_texture_){
        std::string dbg_tex_fname = debug_dir_ + "/mouth_texture_og.png";
        dlib::save_png(mouth_tex, dbg_tex_fname.c_str());
      }
      std::string dbg_tex_completed_fname = debug_dir_ + "/mouth_texture_completed.png";
      dlib::save_png(mouth_texture_, dbg_tex_completed_fname.c_str());
    }
  }

  enable_texture(true);
}

template<class CAM_T, class IMG_T>
bool media_jitterer<CAM_T, IMG_T>::
generate_texture(std::vector<IMG_T> const& images,
                 std::vector<sighting_coefficients<CAM_T> > const& coeffs,
                 vnl_matrix<double> const& subject_pca_components,
                 vnl_matrix<double> const& expression_pca_components,
                 dlib::array2d<dlib::rgb_alpha_pixel> &face_tex,
                 dlib::array2d<dlib::rgb_alpha_pixel> &mouth_tex)
{
  const int num_imgs = images.size();
  if (num_imgs != coeffs.size()) {
    throw std::logic_error("Number of images does not match number of coefficient sightings");
  }
  if (num_imgs == 0) {
    std::cerr << "ERROR: generate_texture() called with vector of 0 images." << std::endl;
    return false;
  }
  std::vector<dlib::array2d<dlib::rgb_alpha_pixel> >  face_textures(num_imgs);
  std::vector<dlib::array2d<dlib::rgb_alpha_pixel> >  mouth_textures(num_imgs);
  for (int img_idx=0; img_idx<num_imgs; ++img_idx) {
    // set coefficients
    base_mesh_.apply_coefficients(subject_pca_components, expression_pca_components,
                                  coeffs[img_idx].subject_coeffs(), coeffs[img_idx].expression_coeffs());

    std::vector< dlib::array2d<dlib::rgb_alpha_pixel > > textures_out;
    if (!model_renderer_.render_to_texture(base_mesh_.meshes(), images[img_idx], coeffs[img_idx].camera(),
                                           textures_out)) {
      std::cerr << "render_to_texture() returned error" << std::endl;
      return false;
    }
    if (textures_out.size() != base_mesh_.meshes().size()) {
      throw std::logic_error("Unexpected number of textures returned from render_to_texture()");
    }
    face_textures[img_idx] = std::move(textures_out[0]);
    if (base_mesh_.has_mouth()) {
      mouth_textures[img_idx] = std::move(textures_out[1]);
    }
  }
  face3d::merge_textures(face_textures, face_tex);
  if (base_mesh_.has_mouth()) {
    face3d::merge_textures(mouth_textures, mouth_tex);
  }

  return true;
}

template <class CAM_T, class IMG_T>
template <class CAM_OUT_T, class IMG_OUT_T>
bool media_jitterer<CAM_T, IMG_T>::
render(CAM_OUT_T const& camera,
       vnl_vector<double> const& subject_coeffs,
       vnl_vector<double> const& expression_coeffs,
       vnl_matrix<double> const& subject_pca_components,
       vnl_matrix<double> const& expression_pca_components,
       IMG_OUT_T &render_img)
{
  base_mesh_.apply_coefficients( subject_pca_components,
                                 expression_pca_components,
                                 subject_coeffs, expression_coeffs );

  // original images are already lit, so don't apply modulation in fragment shader
  model_renderer_.set_ambient_weight(1.0);
  dlib::array2d<dlib::rgb_alpha_pixel> face_image;
  render_aux_out aux_out;
  if (!model_renderer_.render(base_mesh_.meshes(), camera, face_image, aux_out)) {
    std::cerr << "ERROR: mesh_renderer::render() returned error" << std::endl;
    return false;
  }

  if (composite_background_) {
    dlib::array2d<dlib::rgb_pixel> background;
    bkgnd_renderer_.render(sighting_coefficients<CAM_OUT_T>(subject_coeffs, expression_coeffs, camera), background);
    // composite render image on top of background
    const int ny = face_image.nr();
    const int nx = face_image.nc();
    face3d_img_util::set_color_img_size(render_img, ny, nx);
    for (int y=0; y<ny; ++y) {
      for (int x=0; x<nx; ++x) {
        const float render_weight_mag = 10.0f; // amplify the rendered image weight by this (arbitrary) amount.
        dlib::rgb_alpha_pixel rendered_pixel = face3d_img_util::get_pixel(face_image, y, x);
        float render_weight = std::min(1.0f, render_weight_mag * static_cast<float>(rendered_pixel.alpha) / 255.0f);
        unsigned char red   = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, render_weight * rendered_pixel.red   + (1.0f - render_weight) * background[y][x].red)));
        unsigned char green = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, render_weight * rendered_pixel.green + (1.0f - render_weight) * background[y][x].green)));
        unsigned char blue  = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, render_weight * rendered_pixel.blue  + (1.0f - render_weight) * background[y][x].blue)));
        unsigned char curr_pixel [] = {red, green, blue, 255};
        face3d_img_util::assign_pixel(render_img, y, x , curr_pixel, 1);
      }
    }
  }

  return true;
}

template <class CAM_T, class IMG_T>
template <class CAM_OUT_T, class IMG_OUT_T>
bool media_jitterer<CAM_T, IMG_T>::
render(subject_sighting_coefficients<CAM_OUT_T>& output_sightings,
       vnl_matrix<double> const& subject_pca_components,
       vnl_matrix<double> const& expression_pca_components,
       std::vector<IMG_OUT_T>  &jittered_images)
{
  const int num_imgs = output_sightings.num_sightings();

  jittered_images.resize(num_imgs);
  for (int i=0; i<num_imgs; ++i) {
    render(output_sightings.camera(i),
           output_sightings.subject_coeffs(),
           output_sightings.expression_coeffs(i),
           subject_pca_components, expression_pca_components,
           jittered_images[i]);
  }
  return true;
}

template<class CAM_T, class IMG_T>
void media_jitterer<CAM_T, IMG_T>::
enable_texture(bool tex_enabled)
{
  if (tex_enabled) {
    base_mesh_.face_mesh().set_texture(face_texture_);
    if (base_mesh_.has_mouth()) {
      base_mesh_.mouth_mesh().set_texture(mouth_texture_);
    }
  }
  else {
   dlib::array2d<dlib::rgb_pixel> blank_tex(100,100);
   dlib::assign_all_pixels(blank_tex, dlib::rgb_pixel(50,255,50));
    base_mesh_.face_mesh().set_texture(blank_tex);
    if (base_mesh_.has_mouth()) {
      base_mesh_.mouth_mesh().set_texture(blank_tex);
    }
  }
}

} // namespace face3d

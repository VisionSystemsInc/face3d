#pragma once
#include <face3d/pose_jitterer.h>
#include <face3d_basic/perspective_camera_parameters.h>
#include <face3d/mesh_background_renderer_agnostic.h>

using TEX_T = dlib::array2d<dlib::rgb_alpha_pixel>;
namespace face3d{
  template<class CAM_T, class IMG_T>
  class novel_view_jitterer{
  public:
    novel_view_jitterer(std::vector<IMG_T > const& images,
                        std::vector<face3d::textured_triangle_mesh<TEX_T>> & subject_meshes,
                        std::vector<CAM_T> & cam_params,
                        std::string debug_dir="", unsigned gl_device_id=0, bool complete_texture=true, bool composite_background=true)
      :cam_params_(cam_params), subject_meshes_(subject_meshes), images_(images.size()),
       debug_dir_(debug_dir), gl_device_id_(gl_device_id), complete_texture_(complete_texture), composite_background_(composite_background){
      // deep copy images.
      for (int i=0; i<images.size(); ++i) {
        dlib::assign_image(images_[i], images[i]);
      }

      const int face_tex_nr = subject_meshes[0].texture().nr();
      const int face_tex_nc = subject_meshes[0].texture().nc();

      face3d::generate_symmetry_map(subject_meshes_[0].V(), subject_meshes_[0].F(), subject_meshes_[0].T(),
                                  face_tex_nc, face_tex_nr, face_sym_map_);
      this->debug_mode_ = debug_dir != "" ? true : false;
    }
    template <class CAM_OUT_T, class IMG_OUT_T>
    bool render(std::vector<CAM_OUT_T> const& novel_camera,
                std::vector<IMG_OUT_T> &render_img);

    template <class CAM_OUT_T, class IMG_OUT_T>
    bool render(CAM_OUT_T const& novel_camera,
                IMG_OUT_T &render_img);

    bool generate_texture(std::vector<IMG_T > const& images,
                          std::vector<face3d::textured_triangle_mesh<TEX_T> > & subject_meshes,
                          std::vector<CAM_T> & cam_params,
                          dlib::array2d<dlib::rgb_alpha_pixel> &face_tex);

    void override_texture(std::vector<TEX_T>& new_face_textures);
    void init_renderer(){
      bool already_initted = true;
      if (!model_renderer_){
        model_renderer_.reset(new face3d::mesh_renderer(this->gl_device_id_, true));
        already_initted = false;
      }
      if (!bkgnd_renderer_){
        bkgnd_renderer_.reset(new face3d::mesh_background_renderer_agnostic<CAM_T, IMG_T>(this->images_, this->subject_meshes_,this->cam_params_, *this->model_renderer_));
        already_initted = false;
      }
      if (! already_initted)
        init(images_, subject_meshes_, cam_params_, face_sym_map_);
    }

  protected:
    void init(std::vector<IMG_T > const& images,
              std::vector<face3d::textured_triangle_mesh<TEX_T> > & subject_meshes,
              std::vector<CAM_T> & cam_params,
              dlib::array2d<dlib::vector<float,2> > const& face_sym_map);


    void enable_texture(face3d::textured_triangle_mesh<TEX_T> & mesh, bool tex_enabled);
    std::string debug_dir_;
    unsigned gl_device_id_;
    std::shared_ptr<face3d::mesh_renderer> model_renderer_;
    std::shared_ptr<mesh_background_renderer_agnostic<CAM_T, IMG_T> > bkgnd_renderer_;
    dlib::array2d<dlib::rgb_alpha_pixel> face_texture_;
    bool debug_mode_, complete_texture_, composite_background_;
    std::vector<IMG_T > images_;
    std::vector<face3d::textured_triangle_mesh<TEX_T> > subject_meshes_;
    std::vector<CAM_T>  cam_params_;
    dlib::array2d<dlib::vector<float,2> > face_sym_map_;

  };

  template<class CAM_T, class IMG_T>
  void novel_view_jitterer<CAM_T, IMG_T>::
  init(std::vector<IMG_T > const& images,
       std::vector<face3d::textured_triangle_mesh<TEX_T>> & subject_meshes,
       std::vector<CAM_T> & cam_params,
       dlib::array2d<dlib::vector<float,2> > const& face_sym_map)
  {
    // validate input
    if (images.size() != subject_meshes.size()) {
      throw std::logic_error("Different numbers of images and meshes passed to media_jitterer");
    }
    if (debug_mode_) {
      for (int i=0; i<images.size(); ++i) {
        std::stringstream img_fname;
        img_fname << debug_dir_ << "/input_image_" << i << ".png";
        dlib::save_png(images[i],img_fname.str().c_str());
      }
      model_renderer_->set_debug_dir(debug_dir_);
      model_renderer_->set_debug_mode(debug_mode_);
      bkgnd_renderer_->set_debug_dir(debug_dir_);
    }
    // generate a texture map based on input sightings
    dlib::array2d<dlib::rgb_alpha_pixel> face_tex;
    if(!generate_texture(images, subject_meshes, cam_params, face_tex)){
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
    for (auto& mesh : subject_meshes)
      enable_texture(mesh, true);
  }

  template<class CAM_T, class IMG_T>
  bool novel_view_jitterer<CAM_T, IMG_T>::
  generate_texture(std::vector<IMG_T > const& images,
                   std::vector<face3d::textured_triangle_mesh<TEX_T> > & subject_meshes,
                   std::vector<CAM_T> & cam_params,
                   dlib::array2d<dlib::rgb_alpha_pixel> &face_tex)
  {
    const int num_imgs = images.size();
    if (num_imgs != subject_meshes.size()) {
      throw std::logic_error("Number of images does not match number of coefficient sightings");
    }
    if (num_imgs == 0) {
      std::cerr << "ERROR: generate_texture() called with vector of 0 images." << std::endl;
      return false;
    }
    std::vector<dlib::array2d<dlib::rgb_alpha_pixel> >  face_textures(num_imgs);
    for (int img_idx=0; img_idx<num_imgs; ++img_idx) {
      // set coefficients
      std::vector< dlib::array2d<dlib::rgb_alpha_pixel > > textures_out;
      std::vector<face3d::textured_triangle_mesh<TEX_T> > meshes_in; meshes_in.push_back(subject_meshes[img_idx]);
      if (!model_renderer_->render_to_texture(meshes_in, images[img_idx], cam_params[img_idx],
                                              textures_out)) {
        std::cerr << "render_to_texture() returned error" << std::endl;
        return false;
      }
      if (textures_out.size() != meshes_in.size()) {
        throw std::logic_error("Unexpected number of textures returned from render_to_texture()");
      }
      face_textures[img_idx] = std::move(textures_out[0]);
    }
    face3d::merge_textures(face_textures, face_tex);
    return true;
  }

  template <class CAM_T, class IMG_T>
  template <class CAM_OUT_T, class IMG_OUT_T>
  bool novel_view_jitterer<CAM_T, IMG_T>::
  render(std::vector<CAM_OUT_T> const& novel_cameras,
         std::vector<IMG_OUT_T> &render_vec){
    render_vec.resize(novel_cameras.size());
    bool success = true;
    for (unsigned i=0; i<novel_cameras.size(); i++)
      success &= this->render(novel_cameras[i], render_vec[i]);
    return true;
  }

  template <class CAM_T, class IMG_T>
  template <class CAM_OUT_T, class IMG_OUT_T>
  bool novel_view_jitterer<CAM_T, IMG_T>::
  render(CAM_OUT_T const& novel_camera,
         IMG_OUT_T &render_img)
  {

    //TODO enforce meshes have same number of vertices and faces

    // fill in face texture via symmetry constraints from the first mesh

    // original images are already lit, so don't apply modulation in fragment shader
    this->init_renderer();
    model_renderer_->set_ambient_weight(1.0);
    dlib::array2d<dlib::rgb_alpha_pixel> face_image;
    render_aux_out aux_out;
    if (!model_renderer_->render(subject_meshes_, novel_camera, face_image, aux_out)) {
      std::cerr << "ERROR: mesh_renderer::render() returned error" << std::endl;
      return false;
    }

    if (composite_background_) {
      dlib::array2d<dlib::rgb_pixel> background;
      bkgnd_renderer_->render(novel_camera, background, subject_meshes_[0]);
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
    }else{
      dlib::assign_image(render_img, face_image);
    }

    return true;
  }

  template<class CAM_T, class IMG_T>
  void novel_view_jitterer<CAM_T, IMG_T>::
  enable_texture(face3d::textured_triangle_mesh<TEX_T> & mesh, bool tex_enabled)
  {
    if (tex_enabled) {
      mesh.set_texture(face_texture_);
    }else {
      dlib::array2d<dlib::rgb_pixel> blank_tex(100,100);
      dlib::assign_all_pixels(blank_tex, dlib::rgb_pixel(50,255,50));
      mesh.set_texture(blank_tex);
    }
  }

  template<class CAM_T, class IMG_T>
  void novel_view_jitterer<CAM_T, IMG_T>::
  override_texture(std::vector<TEX_T> & new_face_textures)
  {
    if (new_face_textures.size() != this->subject_meshes_.size()){
      std::cout<<"Cannot assign new textures to subject meshes due to size mismatch "<<new_face_textures.size()<<" vs "<<this->subject_meshes_.size()<< std::endl;
        return;
      }
    for (unsigned i=0; i< new_face_textures.size(); i++){
      this->subject_meshes_[i].set_texture(new_face_textures[i]);
    }
  }
}

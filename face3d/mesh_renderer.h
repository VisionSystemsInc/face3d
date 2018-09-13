#ifndef mesh_renderer_h_included_
#define mesh_renderer_h_included_

//std
#include <string>
#include <vector>

//vxl
#include <vil/vil_image_view.h>
#include <vgl/vgl_point_3d.h>

#include <dlib/array2d.h>
#include <dlib/pixel.h>

#include <Eigen/Dense>

#ifndef __APPLE__
#include <glad/glad.h>
#else
#define GLFW_INCLUDE_GLCOREARB
#endif

#if FACE3D_USE_EGL
#include <EGL/egl.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <face3d_basic/ortho_camera_parameters.h>
#include <face3d_basic/face3d_img_util.h>
#include <face3d_basic/perspective_camera_parameters.h>
#include "head_mesh.h"
#include "mesh_io.h"

namespace face3d {

struct render_aux_out
{
  dlib::array2d<vgl_point_3d<float> > *img3d_ = nullptr;
  dlib::array2d<vgl_vector_3d<float> > *normals_ = nullptr;
  dlib::array2d<GLint> *face_idx_ = nullptr;
  dlib::array2d<vgl_point_3d<float> > *face_bc_ = nullptr;
  dlib::array2d<vgl_point_3d<float> > *uv_ = nullptr;
};

class mesh_renderer
{
public:

  mesh_renderer();
  mesh_renderer(unsigned device_id);
#if !FACE3D_USE_EGL
  // constructor used for sharing GLFW OpenGL contexts
  mesh_renderer(GLFWwindow* window);
#endif

  ~mesh_renderer();

  // avoid creating copies. If needed, we'll need to figure out how to deal
  // with the problem of shared (or not) OpenGL context.
  mesh_renderer(mesh_renderer const& other) = delete;
  mesh_renderer(mesh_renderer const&& other) = delete;
  mesh_renderer& operator = (mesh_renderer const& other) = delete;
  mesh_renderer& operator = (mesh_renderer const&& other) = delete;


  // render with all available outputs.  Pass null for unneeded outputs.
  template<class CAM_T, class RENDER_IMG_T, class TEX_T>
  bool render(std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
              CAM_T const& cam_params,
              RENDER_IMG_T &img, render_aux_out &aux_out);

  // convenience version of render with no aux_out return
  template<class CAM_T, class RENDER_IMG_T, class TEX_T>
  bool render(std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
              CAM_T const& cam_params,
              RENDER_IMG_T &img)
  {
    render_aux_out aux_out;
    return render(meshes, cam_params, img, aux_out);
  }

  template<class CAM_T, class IMG_T_IN, class IMG_T_OUT, class TEX_T>
  bool render_to_texture(std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
                         IMG_T_IN const& image,
                         CAM_T const& cam_params,
                         std::vector<IMG_T_OUT> &textures_out);

  template<class RENDER_IMG_T, class TEX_T>
  bool render_2d(Eigen::MatrixXd const& V, Eigen::MatrixXi const& F,
                 Eigen::MatrixXd const& UV, TEX_T const& texture,
                 int nx, int ny, float zmin, float zmax, RENDER_IMG_T &img);

  template <class IMG_T>
  void copy_rgba_image(dlib::array2d<dlib::rgb_alpha_pixel> const& src, IMG_T &dest);

  void set_light_dir(vgl_vector_3d<double> light_dir){light_dir_ = light_dir;}
  void set_ambient_weight(float weight){light_ambient_weight_ = weight;}
  void set_background_color(dlib::rgb_pixel const& color) { background_color_ = color;}

  void set_debug_dir(std::string const& dbg_dir) { debug_dir_ = dbg_dir; }
  void set_debug_mode(bool dbg_mode) { debug_mode_ = dbg_mode; }
  static std::mutex renderer_mutex_;

private:

  void init_renderer();
  void make_context_current();

#if FACE3D_USE_EGL
  void init_context_EGL();
  // egl stuff
  EGLDisplay egl_display_;
  EGLSurface egl_surface_;
  EGLContext egl_context_;
#else
  void init_context_GLFW();
  bool gl_context_owned_;
  GLFWwindow* glfw_window_;
#endif

  bool compile_shader(GLuint shader, std::string const& shader_filename) const;

  template<class TEX_T>
  bool draw_mesh(face3d::textured_triangle_mesh<TEX_T> const& mesh);

  template<class TEX_T>
  bool draw_mesh_2d(Eigen::MatrixXd const& V, Eigen::MatrixXi const& F,
                    Eigen::MatrixXd const& UV, TEX_T const& tex);

  template<class TEX_T>
  bool draw_mesh_triangles(face3d::textured_triangle_mesh<TEX_T> const& mesh);

  bool draw_triangles_2d(Eigen::MatrixXd const& V, Eigen::MatrixXi const& F, Eigen::MatrixXd const& UV);

  void set_camera_uniform_variables(GLuint shader_prog, ortho_camera_parameters<double> const& cam_params);
  void set_camera_uniform_variables(GLuint shader_prog, perspective_camera_parameters<double> const& cam_params);

  GLuint get_shader_prog(ortho_camera_parameters<double> const&) { return shader_prog_ortho_;}
  GLuint get_shader_prog(perspective_camera_parameters<double> const&) { return shader_prog_perspective_;}
  GLuint get_rtt_shader_prog(ortho_camera_parameters<double> const&) { return rtt_shader_prog_ortho_;}
  GLuint get_rtt_shader_prog(perspective_camera_parameters<double> const&) { return rtt_shader_prog_perspective_;}

  GLuint get_2d_shader_prog() { return shader_prog_2d_; }

  void check_gl_error(const char* filename, int line);

  template<class PIX_T>
  void gl_read_pixels_rgba(int nx, int ny, dlib::array2d<PIX_T> &img);
  template<class PIX_T>
  void gl_read_pixels_rgba(int nx, int ny, dlib::matrix<PIX_T> &img);

  void gl_read_pixels_rgba(int nx, int ny, vil_image_view<float> &img);

  template<class PIX_T>
    void gl_tex_image_2d_rgba(int nx, int ny, dlib::array2d<PIX_T> const&img, bool copy_data);
  template<class PIX_T>
    void gl_tex_image_2d_rgba(int nx, int ny, dlib::matrix<PIX_T> const&img, bool copy_data);

    void gl_tex_image_2d_rgba(int nx, int ny, vil_image_view<float> const&img, bool copy_data);

  void gl_tex_image_2d_rgb(int nx, int ny, dlib::array2d<dlib::rgb_pixel> const& img, bool copy_data);
  void gl_tex_image_2d_rgb(int nx, int ny, dlib::matrix<dlib::rgb_pixel> const& img, bool copy_data);
  void gl_tex_image_2d_rgb(int nx, int ny, vil_image_view<float> const&img, bool copy_data);

  template<class T>
  void get_image_dims(vil_image_view<T> const& img, int &nx, int& ny);
  template<class T>
  void get_image_dims(dlib::array2d<T> const& img, int &nx, int& ny);
  template<class T>
  void get_image_dims(dlib::matrix<T> const& img, int &nx, int& ny);

  // perspective-specific parameters
  double focal_len_;
  vgl_point_2d<double> principal_pt_;

  // ortho-specific parameters
  double scene_radius_;
  vgl_point_3d<double> scene_center_;

  dlib::array2d<GLfloat> last_depth_;

  vgl_vector_3d<double> light_dir_;
  float light_ambient_weight_;

  bool debug_mode_;
  std::string debug_dir_;
  int device_id_;
  dlib::rgb_pixel background_color_;
  // standard rendering shader program (compile once and store)
  GLuint shader_prog_ortho_;
  GLuint shader_prog_perspective_;
  // render to texture shader program (compile once and store)
  GLuint rtt_shader_prog_ortho_;
  GLuint rtt_shader_prog_perspective_;
  GLuint shader_prog_2d_;
};


// ----  Templated Member Function Definitions ---- //
template<class T>
void face3d::mesh_renderer::get_image_dims(vil_image_view<T> const& img, int &nx, int& ny)
{
  nx = img.ni();
  ny = img.nj();
}


template<class T>
void face3d::mesh_renderer::get_image_dims(dlib::array2d<T> const& img, int &nx, int& ny)
{
  nx = img.nc();
  ny = img.nr();
}

template<class T>
void face3d::mesh_renderer::get_image_dims(dlib::matrix<T> const& img, int &nx, int& ny)
{
  nx = img.nc();
  ny = img.nr();
}

template<class CAM_T, class RENDER_IMG_T, class TEX_T>
bool face3d::mesh_renderer::render(std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
                                   CAM_T const& cam_params,
                                   RENDER_IMG_T &img, render_aux_out &aux_out)
{
  // make sure we're starting off in a clean state
  check_gl_error(__FILE__, __LINE__);

  const int nx = cam_params.nx();
  const int ny = cam_params.ny();

  if (debug_mode_) {
    std::cout << "rendering with camera parameters: " << cam_params << std::endl;
  }
  make_context_current();

  GLuint color_tex, pt_tex, norm_tex, face_idx_tex, face_bc_tex, uv_tex, depth_tex;

  glGenTextures(1, &color_tex);
  glBindTexture(GL_TEXTURE_2D, color_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl_tex_image_2d_rgba(nx, ny, img, false);
  check_gl_error(__FILE__, __LINE__);

  glGenTextures(1, &depth_tex);
  glBindTexture(GL_TEXTURE_2D, depth_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, nx, ny, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  check_gl_error(__FILE__, __LINE__);

  glGenTextures(1, &pt_tex);
  glBindTexture(GL_TEXTURE_2D, pt_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, nx, ny, 0, GL_RGB, GL_FLOAT, NULL);
  check_gl_error(__FILE__, __LINE__);

  glGenTextures(1, &norm_tex);
  glBindTexture(GL_TEXTURE_2D, norm_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, nx, ny, 0, GL_RGB, GL_FLOAT, NULL);
  check_gl_error(__FILE__, __LINE__);

  glGenTextures(1, &face_idx_tex);
  glBindTexture(GL_TEXTURE_2D, face_idx_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, nx, ny, 0, GL_RED_INTEGER, GL_INT, NULL);
  check_gl_error(__FILE__, __LINE__);

  glGenTextures(1, &face_bc_tex);
  glBindTexture(GL_TEXTURE_2D, face_bc_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, nx, ny, 0, GL_RGB, GL_FLOAT, NULL);
  check_gl_error(__FILE__, __LINE__);

  glGenTextures(1, &uv_tex);
  glBindTexture(GL_TEXTURE_2D, uv_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, nx, ny, 0, GL_RGB, GL_FLOAT, NULL);
  check_gl_error(__FILE__, __LINE__);

  GLuint fb;
  glGenFramebuffers(1, &fb);
  glBindFramebuffer(GL_FRAMEBUFFER, fb);
  //Attach 2D texture to this FBO
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, pt_tex, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, norm_tex, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, face_idx_tex, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, face_bc_tex, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT5, GL_TEXTURE_2D, uv_tex, 0);
  check_gl_error(__FILE__, __LINE__);

  //-------------------------
  //Attach depth texture to FBO
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
  check_gl_error(__FILE__, __LINE__);

  GLuint draw_buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5};
  glDrawBuffers(6, draw_buffers);
  check_gl_error(__FILE__, __LINE__);

  GLenum errcode;
  {
    std::lock_guard<std::mutex> guard(this->renderer_mutex_);
    if((errcode = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE) {
      std::cerr << "ERROR: Framebuffer status != GL_FRAMEBUFFER_COMPLETE" << std::endl;
      std::cerr << "glCheckFramebufferStatus returned 0x" << std::hex << errcode << std::dec << std::endl;
      check_gl_error(__FILE__, __LINE__);
      return false;
    }
    else {
      if (debug_mode_)
        std::cout << "Framebuffer complete." << std::endl;
    }
  }
  ///  RENDER
  glViewport(0, 0, nx, ny);

  // set the background color
  GLfloat color_bg[] = {
    static_cast<GLfloat>(background_color_.red)/255,
    static_cast<GLfloat>(background_color_.green)/255,
    static_cast<GLfloat>(background_color_.blue)/255,
    0.0f};
  GLfloat pt_bg[] = {0.0f, 0.0f, 0.0f};
  GLint face_bg[] = {-1};
  glClearBufferfv(GL_COLOR, 0, color_bg);
  glClearBufferfv(GL_COLOR, 1, pt_bg);
  glClearBufferfv(GL_COLOR, 2, pt_bg);
  glClearBufferiv(GL_COLOR, 3, face_bg);
  glClearBufferfv(GL_COLOR, 4, pt_bg);
  glClear(GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  glBindFramebuffer(GL_FRAMEBUFFER, fb);

  GLuint shader_prog = get_shader_prog(cam_params);
  glUseProgram(shader_prog);

  // set shader variables
  set_camera_uniform_variables(shader_prog, cam_params);
  glUniform3f(glGetUniformLocation(shader_prog, "light_dir"),
              light_dir_.x(), light_dir_.y(), light_dir_.z());
  glUniform1f(glGetUniformLocation(shader_prog, "light_ambient_weight"),
              light_ambient_weight_);
  glUniform1i(glGetUniformLocation(shader_prog, "tex"), 0);

  int face_idx_offset = 0;
  for (face3d::textured_triangle_mesh<TEX_T> const& mesh : meshes) {
    GLint face_idx_offset_loc = glGetUniformLocation(shader_prog, "face_idx_offset");
    // not all shaders use this variable, so need to make sure location is valid
    if (face_idx_offset_loc >= 0) {
      glUniform1i(face_idx_offset_loc, face_idx_offset);
    }
    if (!draw_mesh(mesh)) {
      std::cerr << "ERROR drawing mesh" << std::endl;
      return false;
    }
    face_idx_offset += mesh.F().rows();
  }
  check_gl_error(__FILE__, __LINE__);

  // READ IMAGE BACK
  img.set_size(ny, nx);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  gl_read_pixels_rgba(nx, ny, img);
  check_gl_error(__FILE__, __LINE__);

  // READ 3D Points back
  if (aux_out.img3d_) {
    aux_out.img3d_->set_size(ny, nx);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    std::vector<GLfloat> buff_4DF(nx*ny*4);
    glReadPixels(0,0,nx, ny, GL_RGBA, GL_FLOAT, &(buff_4DF[0]));
    const GLfloat* buff_ptr = &(buff_4DF[0]);
    vgl_point_3d<float>* img3d_ptr = &((*aux_out.img3d_)[0][0]);
    for (int yi=0; yi<ny; ++yi) {
      for (int xi=0; xi<nx; ++xi) {
        *img3d_ptr++ = vgl_point_3d<float>(*buff_ptr, *(buff_ptr+1), *(buff_ptr+2));
        buff_ptr += 4;
      }
    }
    check_gl_error(__FILE__, __LINE__);
  }

  // READ 3D Normals back
  if (aux_out.normals_) {
    aux_out.normals_->set_size(ny, nx);
    glReadBuffer(GL_COLOR_ATTACHMENT2);
    std::vector<GLfloat> buff_4DF(nx*ny*4);
    glReadPixels(0,0, nx, ny, GL_RGBA, GL_FLOAT, &(buff_4DF[0]));
    const GLfloat* buff_ptr = &(buff_4DF[0]);
    vgl_vector_3d<float>* normals_ptr = &((*aux_out.normals_)[0][0]);
    for (int yi=0; yi<ny; ++yi) {
      for (int xi=0; xi<nx; ++xi) {
        *normals_ptr++ = vgl_vector_3d<float>(*buff_ptr, *(buff_ptr+1), *(buff_ptr+2));
        buff_ptr += 4;
      }
    }
    check_gl_error(__FILE__, __LINE__);
  }

  if (aux_out.face_idx_) {
    // READ Face indices back
    aux_out.face_idx_->set_size(ny, nx);
    std::vector<GLint> buff_4DI(nx*ny*4);
    glReadBuffer(GL_COLOR_ATTACHMENT3);
    glReadPixels(0,0, nx, ny, GL_RGBA_INTEGER, GL_INT, &(buff_4DI[0]));
    const GLint* buff_ptr = &(buff_4DI[0]);
    int* face_idx_ptr = &((*aux_out.face_idx_)[0][0]);
    for (int yi=0; yi<ny; ++yi) {
      for (int xi=0; xi<nx; ++xi) {
        *face_idx_ptr++ = *buff_ptr++;
      }
    }
    check_gl_error(__FILE__, __LINE__);
  }

  // READ Face barycentric coordinates
  if (aux_out.face_bc_) {
    aux_out.face_bc_->set_size(ny,nx);
    glReadBuffer(GL_COLOR_ATTACHMENT4);
    std::vector<GLfloat> buff_4DF(nx*ny*4);
    glReadPixels(0,0, nx, ny, GL_RGBA, GL_FLOAT, &(buff_4DF[0]));
    const GLfloat* buff_ptr = &(buff_4DF[0]);
    vgl_point_3d<float> *face_bc_ptr = &((*aux_out.face_bc_)[0][0]);
    for (int yi=0; yi<ny; ++yi) {
      for (int xi=0; xi<nx; ++xi) {
        *face_bc_ptr++ = vgl_point_3d<float>(*buff_ptr, *(buff_ptr+1), *(buff_ptr+2));
        buff_ptr += 4;
      }
    }
    check_gl_error(__FILE__, __LINE__);
  }

  // READ UV coordinates
  if (aux_out.uv_) {
    aux_out.uv_->set_size(ny,nx);
    glReadBuffer(GL_COLOR_ATTACHMENT5);
    std::vector<GLfloat> buff_4DF(nx*ny*4);
    glReadPixels(0,0, nx, ny, GL_RGBA, GL_FLOAT, &(buff_4DF[0]));
    const GLfloat* buff_ptr = &(buff_4DF[0]);
    vgl_point_3d<float> *uv_ptr = &((*aux_out.uv_)[0][0]);
    for (int yi=0; yi<ny; ++yi) {
      for (int xi=0; xi<nx; ++xi) {
        *uv_ptr++ = vgl_point_3d<float>(*buff_ptr, *(buff_ptr+1), *(buff_ptr+2));
        buff_ptr += 4;
      }
    }
    check_gl_error(__FILE__, __LINE__);
  }

  // READ DEPTH BACK
  last_depth_.set_size(ny, nx);
  glBindTexture(GL_TEXTURE_2D, depth_tex);
  glReadPixels(0,0, nx, ny, GL_DEPTH_COMPONENT, GL_FLOAT, &(last_depth_[0][0]));
  check_gl_error(__FILE__, __LINE__);

  // clean up
  glDeleteTextures(1,&color_tex);
  glDeleteTextures(1,&pt_tex);
  glDeleteTextures(1,&norm_tex);
  glDeleteTextures(1,&depth_tex);
  glDeleteTextures(1,&face_idx_tex);
  glDeleteTextures(1,&face_bc_tex);
  glDeleteTextures(1,&uv_tex);
  // unbind fb by switching to 0 (back buffer)
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fb);
  check_gl_error(__FILE__, __LINE__);

  return true;
}

template<class CAM_T, class IMG_T_IN, class IMG_T_OUT, class TEX_T>
bool mesh_renderer::
render_to_texture(std::vector<face3d::textured_triangle_mesh<TEX_T> > const& meshes,
                  IMG_T_IN const& image,
                  CAM_T const& cam_params,
                  std::vector<IMG_T_OUT> &textures_out)
{
  textures_out.clear();

  // first, render original image to get depth values
  // only output we care about is the last_depth_ member
  dlib::array2d<dlib::rgb_alpha_pixel> img;
  render_aux_out aux_out;
  render(meshes, cam_params, img, aux_out);

  int img_nx, img_ny;
  get_image_dims(image,img_nx,img_ny);

  if ((cam_params.nx() != img_nx) || (cam_params.ny() != img_ny)) {
    std::cerr << "ERROR: image dimensions in camera parameters do not match image size:";
    std::cerr << " image: " << img_nx << "x" << img_ny;
    std::cerr << " cam_params: " << cam_params.nx() << "x" << cam_params.ny() << std::endl;
    return false;
  }

  make_context_current();
  check_gl_error(__FILE__, __LINE__);

  for (face3d::textured_triangle_mesh<TEX_T> const& mesh : meshes) {

    GLuint color_tex, depth_tex;
    const int tex_nx = mesh.texture().nc();
    const int tex_ny = mesh.texture().nr();

    IMG_T_OUT tex_out;

    glGenTextures(1, &color_tex);
    glBindTexture(GL_TEXTURE_2D, color_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl_tex_image_2d_rgba(tex_nx, tex_ny, tex_out, false);
    check_gl_error(__FILE__, __LINE__);

    glGenTextures(1, &depth_tex);
    glBindTexture(GL_TEXTURE_2D, depth_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, tex_nx, tex_ny, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    check_gl_error(__FILE__, __LINE__);

    GLuint fb;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    //Attach 2D texture to this FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
    //Attach depth texture to FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);

    GLuint draw_buffers[] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, draw_buffers);

    GLenum errcode;
    {
      std::lock_guard<std::mutex> guard(this->renderer_mutex_);
      if((errcode = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR: Framebuffer status != GL_FRAMEBUFFER_COMPLETE" << std::endl;
        std::cerr << "glCheckFramebufferStatus returned 0x" << std::hex << errcode << std::dec << std::endl;
        check_gl_error(__FILE__, __LINE__);
        return false;
      }
    }
    ///  RENDER
    glViewport(0, 0, tex_nx, tex_ny);

    // set the background color
    GLfloat tex_bg[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearBufferfv(GL_COLOR, 0, tex_bg);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    // get the texture
    GLuint img_tex;
    glGenTextures(1, &img_tex);
    glBindTexture(GL_TEXTURE_2D, img_tex);
    // Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    gl_tex_image_2d_rgb(img_nx, img_ny, image, true);

    // sanity check on last_depth_ size
    if ((last_depth_.nr() != img_ny) || (last_depth_.nc() != img_nx)) {
      throw std::runtime_error("Depth and Image sizes do not match!");
    }

    GLuint img_depth_tex;
    glGenTextures(1, &img_depth_tex);
    glBindTexture(GL_TEXTURE_2D, img_depth_tex);
    // Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	// Set texture wrapping to GL_REPEAT (usually basic wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, img_nx, img_ny, 0, GL_DEPTH_COMPONENT, GL_FLOAT, &(last_depth_[0][0]));

    glBindTexture(GL_TEXTURE_2D, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, fb);

    GLuint rtt_shader_prog = get_rtt_shader_prog(cam_params);
    glUseProgram(rtt_shader_prog);

    // set shader variables
    set_camera_uniform_variables(rtt_shader_prog, cam_params);

    // convert camera rotation to 3x3 matrix of GLfloats
    vnl_quaternion<double> qd = cam_params.rotation().as_quaternion();
    vnl_quaternion<GLfloat> qf(qd[0],qd[1],qd[2],qd[3]);
    vnl_matrix_fixed<GLfloat,3,3> R = vgl_rotation_3d<GLfloat>(qf).as_matrix();
    vnl_vector_fixed<GLfloat,3> cam_z = R.get_column(2);
    glUniform3f(glGetUniformLocation(rtt_shader_prog, "camera_dir"),
                                     -cam_z[0], -cam_z[1], -cam_z[2]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, img_tex);
    glUniform1i(glGetUniformLocation(rtt_shader_prog, "img"), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, img_depth_tex);
    glUniform1i(glGetUniformLocation(rtt_shader_prog, "img_depth"), 1);

    if(!draw_mesh_triangles(mesh)) {
      std::cerr << "ERROR rendering mesh triangles" << std::endl;
      return false;
    }

    // READ IMAGE BACK
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    gl_read_pixels_rgba(tex_nx, tex_ny, tex_out);
    textures_out.push_back(std::move(tex_out));

    // clean up
    glDeleteTextures(1,&color_tex);
    glDeleteTextures(1,&depth_tex);
    glDeleteTextures(1,&img_tex);
    glDeleteTextures(1,&img_depth_tex);

    // unbind fb by switching to 0 (back buffer)
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fb);
  }

  return true;
}

template<class RENDER_IMG_T, class TEX_T>
bool mesh_renderer::
render_2d(Eigen::MatrixXd const& V, Eigen::MatrixXi const& F,
          Eigen::MatrixXd const& UV, TEX_T const& texture,
          int nx, int ny, float zmin, float zmax,
          RENDER_IMG_T &img)
{
  make_context_current();
  GLuint color_tex, depth_tex;
  glGenTextures(1, &color_tex);
  glBindTexture(GL_TEXTURE_2D, color_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  //NULL means reserve texture memory, but texels are undefined
  gl_tex_image_2d_rgba(nx, ny, img, false);
  check_gl_error(__FILE__, __LINE__);

  glGenTextures(1, &depth_tex);
  glBindTexture(GL_TEXTURE_2D, depth_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
  //NULL means reserve texture memory, but texels are undefined
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, nx, ny, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
  check_gl_error(__FILE__, __LINE__);

  GLuint fb;
  glGenFramebuffers(1, &fb);
  glBindFramebuffer(GL_FRAMEBUFFER, fb);
  //Attach 2D texture to this FBO
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
  //Attach depth texture to FBO
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0);
  check_gl_error(__FILE__, __LINE__);

  GLuint draw_buffers[] = {GL_COLOR_ATTACHMENT0,};
  glDrawBuffers(1, draw_buffers);

  if(GLenum errcode = glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "ERROR: Framebuffer status != GL_FRAMEBUFFER_COMPLETE" << std::endl;
    std::cerr << "glCheckFramebufferStatus returned 0x" << std::hex << errcode << std::dec << std::endl;
    check_gl_error(__FILE__, __LINE__);
    return false;
  }
  check_gl_error(__FILE__, __LINE__);

  ///  RENDER
  glViewport(0, 0, nx, ny);

  // set the background color
  GLfloat color_bg[] = {0.0, 0.0, 0.0, 0.0f};
  glClearBufferfv(GL_COLOR, 0, color_bg);
  glClear(GL_DEPTH_BUFFER_BIT);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST);

  glBindFramebuffer(GL_FRAMEBUFFER, fb);

  GLuint shader_prog = get_2d_shader_prog();
  glUseProgram(shader_prog);

  // set shader variables
  glUniform1i(glGetUniformLocation(shader_prog, "tex"), 0);
  glUniform2f(glGetUniformLocation(shader_prog, "img_dims"), nx, ny);
  glUniform2f(glGetUniformLocation(shader_prog, "depth_range"), zmin-0.1f, zmax+0.1f);

  if (!draw_mesh_2d(V,F,UV,texture)) {
    std::cerr << "ERROR drawing 2D mesh" << std::endl;
    return false;
  }
  check_gl_error(__FILE__, __LINE__);

  // READ IMAGE BACK
  img.set_size(ny, nx);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  gl_read_pixels_rgba(nx, ny, img);
  check_gl_error(__FILE__, __LINE__);

  // clean up
  glDeleteTextures(1,&color_tex);
  glDeleteTextures(1,&depth_tex);
  // unbind fb by switching to 0 (back buffer)
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fb);

  return true;
}

template<class PIX_T>
void mesh_renderer::gl_tex_image_2d_rgba(int nx, int ny, dlib::array2d<PIX_T> const& img, bool copy_data)
{
  GLvoid const* data_ptr = nullptr;
  if (copy_data) {
    if (face3d_img_util::pixel_traits<PIX_T>::nplanes != 4) {
      throw std::logic_error("non-4plane image passed to gl_tex_image_2d_rgba");
    }
    if (sizeof(typename face3d_img_util::pixel_traits<PIX_T>::dtype) != sizeof(unsigned char)) {
      throw std::logic_error("gl_tex_image_2d_rgba: dlib images must have byte pixel types");
    }
    if ((img.nr() != ny) || (img.nc() != nx)) {
      throw std::logic_error("gl_tex_image_2d_rgba: image size does not match arguments");
    }
    data_ptr = &(img[0][0]);
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, data_ptr);
}

template<class PIX_T>
void mesh_renderer::gl_tex_image_2d_rgba(int nx, int ny, dlib::matrix<PIX_T> const& img, bool copy_data)
{
  GLvoid const* data_ptr = nullptr;
  if (copy_data) {
    if (face3d_img_util::pixel_traits<PIX_T>::nplanes != 4) {
      throw std::logic_error("non-4plane image passed to gl_tex_image_2d_rgba");
    }
    if (sizeof(typename face3d_img_util::pixel_traits<PIX_T>::dtype) != sizeof(unsigned char)) {
      throw std::logic_error("gl_tex_image_2d_rgba: dlib images must have byte pixel types");
    }
    if ((img.nr() != ny) || (img.nc() != nx)) {
      throw std::logic_error("gl_tex_image_2d_rgba: image size does not match arguments");
    }
    data_ptr = &(img[0][0]);
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nx, ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, data_ptr);
}

template<class PIX_T>
void mesh_renderer::gl_read_pixels_rgba(int nx, int ny, dlib::array2d<PIX_T> &img)
{
  if (face3d_img_util::pixel_traits<PIX_T>::nplanes == 4) {
    // image has matching number of planes. read straight into image memory.
    img.set_size(ny,nx);
    glReadPixels(0,0, img.nc(), img.nr(), GL_RGBA, GL_UNSIGNED_BYTE, &(img[0][0]));
    check_gl_error(__FILE__, __LINE__);
  }
  else if(face3d_img_util::pixel_traits<PIX_T>::nplanes == 3) {
    // need to get rgba into an rgb image.
    dlib::array2d<dlib::rgb_alpha_pixel> temp(ny,nx);
    glReadPixels(0,0, img.nc(), img.nr(), GL_RGBA, GL_UNSIGNED_BYTE, &(temp[0][0]));
    check_gl_error(__FILE__, __LINE__);
    // copy to output
    copy_rgba_image(temp, img);
  }
  else {
    std::cerr << "ERROR: img has " <<  face3d_img_util::pixel_traits<PIX_T>::nplanes << " planes." << std::endl;
    throw std::logic_error("gl_read_pixels_rgba called with unexpected number of planes");
  }
}

template<class PIX_T>
void mesh_renderer::gl_read_pixels_rgba(int nx, int ny, dlib::matrix<PIX_T> &img)
{
  if (face3d_img_util::pixel_traits<PIX_T>::nplanes == 4) {
    // image has matching number of planes. read straight into image memory.
    img.set_size(ny,nx);
    glReadPixels(0,0, img.nc(), img.nr(), GL_RGBA, GL_UNSIGNED_BYTE, &(img[0][0]));
    check_gl_error(__FILE__, __LINE__);
  }
  else if(face3d_img_util::pixel_traits<PIX_T>::nplanes == 3) {
    // need to get rgba into an rgb image.
    dlib::array2d<dlib::rgb_alpha_pixel> temp;
    glReadPixels(0,0, img.nc(), img.nr(), GL_RGBA, GL_UNSIGNED_BYTE, &(temp[0][0]));
    check_gl_error(__FILE__, __LINE__);
    // copy to output
    copy_rgba_image(temp, img);
  }
  else {
    std::cerr << "ERROR: img has " <<  face3d_img_util::pixel_traits<PIX_T>::nplanes << " planes." << std::endl;
    throw std::logic_error("gl_read_pixels_rgba called with unexpected number of planes");
  }
}

template <class IMG_T>
void mesh_renderer::copy_rgba_image(dlib::array2d<dlib::rgb_alpha_pixel> const& src, IMG_T &dest)
{
  const int nr = src.nr();
  const int nc = src.nc();
  face3d_img_util::set_color_img_size(dest, nr, nc);
  // don't use dlib::assign_image here, because we don't want alpha matting.
  for (int r=0; r<nr; ++r) {
    for (int c=0; c<nc; ++c) {
      dlib::rgb_alpha_pixel const& pix(src[r][c]);
      unsigned char pixel_ptr [] = {pix.red, pix.green, pix.blue, pix.alpha};
      face3d_img_util::assign_pixel(dest, r, c, pixel_ptr, 1);
    }
  }
}


template<class TEX_T>
bool face3d::mesh_renderer::draw_mesh_triangles(face3d::textured_triangle_mesh<TEX_T> const& mesh)
{
  triangle_mesh::VTYPE const& V = mesh.V();
  const int num_verts = V.rows();

  triangle_mesh::TTYPE const& T = mesh.T();
  if (T.rows() != num_verts) {
    std::cout << "ERROR: Mesh has " << T.rows() << " texture coordinates. Expecting " << num_verts << std::endl;
    throw std::runtime_error("mesh does not have correct number of texture coordinates");
  }

  triangle_mesh::NTYPE const& N = mesh.N();
  if (N.rows() != num_verts) {
    throw std::runtime_error("mesh does not have correct number of vertex normals");
  }

  std::vector<GLfloat> verts(num_verts*8);
  for (int i=0; i<num_verts; ++i) {
    // vertex positions
    verts[i*8+0] = V(i,0);
    verts[i*8+1] = V(i,1);
    verts[i*8+2] = V(i,2);
    // vertex normals
    verts[i*8+3] = N(i,0);
    verts[i*8+4] = N(i,1);
    verts[i*8+5] = N(i,2);
    // texture coordinates
    verts[i*8+6] = T(i,0);
    verts[i*8+7] = T(i,1);
  }

  triangle_mesh::FTYPE const& F = mesh.F();
  const int num_faces = F.rows();
  std::vector<GLuint> faces(num_faces*3);
  for (int i=0; i<num_faces; ++i) {
    faces[i*3+0] = F(i,0);
    faces[i*3+1] = F(i,1);
    faces[i*3+2] = F(i,2);
  }

  GLuint VBO, VAO, EBO;
  glGenVertexArrays(1,&VAO);
  glGenBuffers(1,&VBO);
  glGenBuffers(1,&EBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*verts.size(), &verts[0], GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*faces.size(), &faces[0], GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(GLfloat), (GLvoid*)(3*sizeof(GLfloat)));
  glEnableVertexAttribArray(1);

  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(GLfloat), (GLvoid*)(6*sizeof(GLfloat)));
  glEnableVertexAttribArray(2);

  // draw
  glDrawElements(GL_TRIANGLES, num_faces*3, GL_UNSIGNED_INT, 0);

  // unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // clean up
  glDeleteBuffers(1,&VBO);
  glDeleteBuffers(1,&EBO);
  glDeleteVertexArrays(1,&VAO);

  return true;
}


template<class TEX_T>
bool mesh_renderer::draw_mesh(face3d::textured_triangle_mesh<TEX_T> const& mesh)
{
  glActiveTexture(GL_TEXTURE0);
  // get the texture
  GLuint mesh_tex;
  glGenTextures(1, &mesh_tex);
  glBindTexture(GL_TEXTURE_2D, mesh_tex);
  // Set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT (usually basic wrapping method)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // Set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  TEX_T const& tex_img = mesh.texture();
  int tex_nx, tex_ny;
  get_image_dims(tex_img, tex_nx, tex_ny);
  gl_tex_image_2d_rgba(tex_nx, tex_ny, tex_img, true);

  // set up back-face culling
  glFrontFace(GL_CW); // Note: This is not the default
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  // Do the drawing
  if(!draw_mesh_triangles(mesh)) {
    std::cerr << "ERROR drawing mesh triangles" << std::endl;
    return false;
  }

  // clean up
  glDeleteTextures(1,&mesh_tex);
  return true;
}

template<class TEX_T>
bool mesh_renderer::
draw_mesh_2d(Eigen::MatrixXd const& V, Eigen::MatrixXi const& F,
             Eigen::MatrixXd const& UV, TEX_T const& tex_img)
{
  glActiveTexture(GL_TEXTURE0);
  // get the texture
  GLuint mesh_tex;
  glGenTextures(1, &mesh_tex);
  glBindTexture(GL_TEXTURE_2D, mesh_tex);
  // Set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// Set texture wrapping to GL_REPEAT (usually basic wrapping method)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // Set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  dlib::array2d<dlib::rgb_alpha_pixel> tex_img_walpha;
  dlib::assign_image(tex_img_walpha, tex_img);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex_img.nc(), tex_img.nr(), 0, GL_RGBA, GL_UNSIGNED_BYTE, &(tex_img_walpha[0][0]));

  // disable back-face culling
  glDisable(GL_CULL_FACE);

  // Do the drawing
  if(!draw_triangles_2d(V,F,UV)) {
    std::cerr << "ERROR drawing mesh triangles" << std::endl;
    return false;
  }

  // clean up
  glDeleteTextures(1,&mesh_tex);
  return true;
}



}

#endif

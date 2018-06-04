#include "mesh_renderer.h"
#include <face3d_basic/io_utils.h>
#include <face3d_basic/image_conversion.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

#include <igl/readPLY.h>
#include <igl/readOBJ.h>
#include <igl/writePLY.h>
#include <igl/readOFF.h>

#include <vil/vil_image_view.h>

#if FACE3D_USE_EGL
#include <EGL/eglext.h>
#include <EGL/eglplatform.h>
#endif

using namespace face3d;
using std::string;
using std::vector;
using face3d::ortho_camera_parameters;

std::mutex face3d::mesh_renderer::renderer_mutex_;
face3d::mesh_renderer::mesh_renderer()
  : light_dir_(0,0.5,0.5), light_ambient_weight_(0.5), debug_mode_(false),
    background_color_(50,150,50),device_id_(0)
{
#if !FACE3D_USE_EGL
  glfw_window_ = nullptr;
#endif
  init_renderer();
}

face3d::mesh_renderer::mesh_renderer(unsigned device_id)
  : light_dir_(0,0.5,0.5), light_ambient_weight_(0.5), debug_mode_(false),
    background_color_(50,150,50),device_id_(device_id)
{
#if !FACE3D_USE_EGL
  glfw_window_ = nullptr;
#endif
  init_renderer();
}

#if !FACE3D_USE_EGL
face3d::mesh_renderer::mesh_renderer(GLFWwindow* window)
  : light_dir_(0,0.5,0.5), light_ambient_weight_(0.5), debug_mode_(false),
  background_color_(50,150,50), glfw_window_(window)
{
  init_renderer();
}
#endif

#if FACE3D_USE_EGL
void face3d::mesh_renderer::init_context_EGL()
{
#define USE_EGL_EXT
#ifdef USE_EGL_EXT
  PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
   PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC)eglGetProcAddress("eglQueryDeviceAttribEXT");
   PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT = (PFNEGLQUERYDEVICESTRINGEXTPROC) eglGetProcAddress("eglQueryDeviceStringEXT");

  EGLint num_egl_devices = -1;

  //Get number of devices
  if (!eglQueryDevicesEXT(0, NULL, &num_egl_devices)) {
    std::cerr << "ERROR: eglQueryDevices() returned error." << std::endl;
    throw std::runtime_error("EGL Initialization Error");
  }
  std::cout << "Found " << num_egl_devices << " EGL devices." << std::endl;
  if (num_egl_devices <= 0) {
    std::cerr << "ERROR: num_egl_devices = " << num_egl_devices << std::endl;
    throw std::runtime_error("Found 0 EGL Devices");
  }
  std::vector<EGLDeviceEXT> egl_devices(num_egl_devices);
   if (num_egl_devices <= this->device_id_) {
     std::cerr << "ERROR: num_egl_devices = " << num_egl_devices <<" while requested device is "<<this->device_id_<< std::endl;
    throw std::runtime_error("Invalid Device ID index");
  }
   eglQueryDevicesEXT(num_egl_devices, &egl_devices[0], &num_egl_devices);
   const char *devstr= eglQueryDeviceStringEXT(egl_devices[this->device_id_], EGL_DRM_DEVICE_FILE_EXT);
   std::stringstream dev_str;
   dev_str<<devstr;

   std::cout << "Using EGL Device Index: " << this->device_id_ << " ("<<dev_str.str()<<")"<<std::endl;

  // 1. initialize the EGL display
  egl_display_ = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_devices[this->device_id_], 0);
#else
  egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
#endif

  EGLint major=-1, minor=-1;

  if (egl_display_ == EGL_NO_DISPLAY) {
    throw std::runtime_error("Failed to get an EGL Display");
  }
  if(eglInitialize(egl_display_, &major, &minor) == EGL_FALSE) {
    throw std::runtime_error("Failed to initialize EGL Display");
  }
  std::cout << "initialized EGL, version = " << major << "." << minor << std::endl;

  // 2. Select an appropriate configuration
#if 1
  static const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 24,
    EGL_CONFORMANT, EGL_OPENGL_ES2_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_NONE
  };
#else
  static const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_NONE
  };
#endif

  static const EGLint pbufferAttribs[] = {
    EGL_WIDTH, 64,
    EGL_HEIGHT, 64,
    EGL_NONE,
  };

  EGLint num_egl_configs;
  EGLConfig egl_config;
  eglChooseConfig(egl_display_, configAttribs, &egl_config, 1, &num_egl_configs);

  std::cout << "Found " << num_egl_configs << " EGL configs." << std::endl;

//#define CREATE_EGL_SURFACE
#ifdef CREATE_EGL_SURFACE
  // 3. Create a surface
  egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config, pbufferAttribs);
  //egl_surface_ = eglCreateWindowSurface(egl_display_, egl_config, (NativeWindowType)NULL, NULL);
  if (egl_surface_ == EGL_NO_SURFACE) {
    throw std::runtime_error("eglCreateWindowSurface() failed");
  }
#else
  egl_surface_ = EGL_NO_SURFACE;
#endif

  // 4. Bind the API
  if(eglBindAPI(EGL_OPENGL_ES_API) == EGL_FALSE) {
    throw std::runtime_error("eglBindAPI failed");
  }

#define USE_GLES2
#ifdef USE_GLES2
  EGLint context_attributes[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
#else
  EGLint *context_attributes = NULL;

#endif

  // 5. Create a context
  egl_context_ = eglCreateContext(egl_display_, egl_config, EGL_NO_CONTEXT, context_attributes);
  if (egl_context_ == EGL_NO_CONTEXT) {
    throw std::runtime_error("eglCreateContext() failed");
  }
}
#else
void face3d::mesh_renderer::init_context_GLFW()
{
  if (glfw_window_ == nullptr) {
    gl_context_owned_ = true;
    if(!glfwInit()) {
      throw std::runtime_error("Error initializing GLFW context");
    }
    glfwSetErrorCallback([](int error, const char* description){std::cerr << "GLFW Error: " << description << std::endl;});
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    glfw_window_ = glfwCreateWindow(1,1,"Render Window", NULL, NULL);
    if (!glfw_window_) {
      throw std::runtime_error("Error creating GLFW window");
    }
    std::cerr << "Created GLFW Window" << std::endl;
  }
  else {
    gl_context_owned_ = false;
  }
}
#endif


void face3d::mesh_renderer::make_context_current()
{
#if FACE3D_USE_EGL
  if(eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_) != EGL_TRUE) {
    std::cerr << "ERROR: eglMakeCurrent() returned error" << std::endl;
    throw std::runtime_error("eglMakeCurrent failed");
  }
#else
  if (!glfw_window_) {
    throw std::runtime_error("make_context_current() called with invalid glfw_window");
  }
  glfwMakeContextCurrent(glfw_window_);
#endif
}

void face3d::mesh_renderer::init_renderer()
{
  std::lock_guard<std::mutex> guard(this->renderer_mutex_);
#if FACE3D_USE_EGL
  init_context_EGL();
#else
  init_context_GLFW();
#endif

  make_context_current();

#ifndef __APPLE__
#ifdef USE_GLES2
  if(!gladLoadGLES2Loader((GLADloadproc)eglGetProcAddress)) {
    throw std::runtime_error("GLAD failed to load OpenGL ES2");
  }
  std::cout << "Loaded OpenGL ES2" << std::endl;
  std::cout << "OpenGL Version " << GLVersion.major << "." << GLVersion.minor << " loaded." << std::endl;
#else
  if(!gladLoadGL()) {
    throw std::runtime_error("GLAD failed to load OpenGL");
  }
  std::cout << "OpenGL Version " << GLVersion.major << "." << GLVersion.minor << " loaded." << std::endl;
#endif
#endif

  // allow reading and writing of arbitrary-sized textures
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1);

  // compile shaders
  const std::string shader_src_dir { FACE3D_SHADER_DIR };

  // compile render shader program
  {
    const std::string vertex_shader_fname_ortho = shader_src_dir + "/vertex_shader_ortho.glsl";
    GLuint vertex_shader_ortho = glCreateShader(GL_VERTEX_SHADER);
    if(!compile_shader(vertex_shader_ortho, vertex_shader_fname_ortho)) {
      throw std::runtime_error("ERROR: ortho vertex shader failed to compile.");
    }
    const std::string vertex_shader_fname_perspective = shader_src_dir + "/vertex_shader_perspective.glsl";
    GLuint vertex_shader_perspective = glCreateShader(GL_VERTEX_SHADER);
    if(!compile_shader(vertex_shader_perspective, vertex_shader_fname_perspective)) {
      throw std::runtime_error("ERROR: perspective vertex shader failed to compile.");
    }
    const std::string geometry_shader_fname = shader_src_dir + "/geometry_shader.glsl";
    GLuint geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
    if(!compile_shader(geometry_shader, geometry_shader_fname)) {
      throw std::runtime_error("ERROR: geometry shader failed to compile.");
    }
    const std::string fragment_shader_fname = shader_src_dir + "/fragment_shader.glsl";
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    if(!compile_shader(fragment_shader, fragment_shader_fname)) {
      throw std::runtime_error("ERROR: fragment shader failed to compile.");
    }
    const std::string vertex_shader_fname_2d = shader_src_dir + "/vertex_shader_2d.glsl";
    GLuint vertex_shader_2d = glCreateShader(GL_VERTEX_SHADER);
    if(!compile_shader(vertex_shader_2d, vertex_shader_fname_2d)) {
      throw std::runtime_error("ERROR: 2d vertex shader failed to compile.");
    }
    const std::string fragment_shader_fname_2d = shader_src_dir + "/fragment_shader_2d.glsl";
    GLuint fragment_shader_2d = glCreateShader(GL_FRAGMENT_SHADER);
    if(!compile_shader(fragment_shader_2d, fragment_shader_fname_2d)) {
      throw std::runtime_error("ERROR: 2d fragment shader failed to compile.");
    }
    {
      shader_prog_ortho_ = glCreateProgram();
      glAttachShader(shader_prog_ortho_, vertex_shader_ortho);
      glAttachShader(shader_prog_ortho_, geometry_shader);
      glAttachShader(shader_prog_ortho_, fragment_shader);
      glLinkProgram(shader_prog_ortho_);
      GLint link_success;
      glGetProgramiv(shader_prog_ortho_, GL_LINK_STATUS, &link_success);
      if (debug_mode_ || !link_success) {
        char info_log[512];
        glGetProgramInfoLog(shader_prog_ortho_, 512, NULL, info_log);
        std::cout << info_log << std::endl;
      }
      if (!link_success) {
        throw std::runtime_error("Ortho shader program failed to link.");
      }
    }
    {
      shader_prog_perspective_ = glCreateProgram();
      glAttachShader(shader_prog_perspective_, vertex_shader_perspective);
      glAttachShader(shader_prog_perspective_, geometry_shader);
      glAttachShader(shader_prog_perspective_, fragment_shader);
      glLinkProgram(shader_prog_perspective_);
      GLint link_success;
      glGetProgramiv(shader_prog_perspective_, GL_LINK_STATUS, &link_success);
      if (debug_mode_ || !link_success) {
        char info_log[512];
        glGetProgramInfoLog(shader_prog_perspective_, 512, NULL, info_log);
        std::cout << info_log << std::endl;
      }
      if (!link_success) {
        throw std::runtime_error("Perspective shader program failed to link.");
      }
    }
    {
      shader_prog_2d_ = glCreateProgram();
      glAttachShader(shader_prog_2d_, vertex_shader_2d);
      glAttachShader(shader_prog_2d_, fragment_shader_2d);
      glLinkProgram(shader_prog_2d_);
      GLint link_success;
      glGetProgramiv(shader_prog_2d_, GL_LINK_STATUS, &link_success);
      if (debug_mode_ || !link_success) {
        char info_log[512];
        glGetProgramInfoLog(shader_prog_2d_, 512, NULL, info_log);
        std::cout << info_log << std::endl;
      }
      if (!link_success) {
        throw std::runtime_error("2D shader program failed to link.");
      }
    }
      glDeleteShader(vertex_shader_ortho);
      glDeleteShader(vertex_shader_perspective);
      glDeleteShader(geometry_shader);
      glDeleteShader(fragment_shader);
      glDeleteShader(vertex_shader_2d);
      glDeleteShader(fragment_shader_2d);
  }
  // compile render to texture shader program
  {
    const std::string rtt_vertex_shader_ortho_fname = shader_src_dir + "/render_to_texture_vertex_shader_ortho.glsl";
    GLuint rtt_vertex_shader_ortho = glCreateShader(GL_VERTEX_SHADER);
    if(!compile_shader(rtt_vertex_shader_ortho, rtt_vertex_shader_ortho_fname)) {
      throw std::runtime_error("ERROR: RTT ortho vertex shader failed to compile.");
    }
    const std::string rtt_vertex_shader_perspective_fname = shader_src_dir + "/render_to_texture_vertex_shader_perspective.glsl";
    GLuint rtt_vertex_shader_perspective = glCreateShader(GL_VERTEX_SHADER);
    if(!compile_shader(rtt_vertex_shader_perspective, rtt_vertex_shader_perspective_fname)) {
      throw std::runtime_error("ERROR: RTT perspective vertex shader failed to compile.");
    }
    const std::string rtt_fragment_shader_fname = shader_src_dir + "/render_to_texture_fragment_shader.glsl";
    GLuint rtt_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    if(!compile_shader(rtt_fragment_shader, rtt_fragment_shader_fname)) {
      throw std::runtime_error("ERROR: RTT fragment shader failed to compile.");
    }

    {
      rtt_shader_prog_ortho_ = glCreateProgram();
      glAttachShader(rtt_shader_prog_ortho_, rtt_vertex_shader_ortho);
      glAttachShader(rtt_shader_prog_ortho_, rtt_fragment_shader);
      glLinkProgram(rtt_shader_prog_ortho_);
      GLint link_success;
      glGetProgramiv(rtt_shader_prog_ortho_, GL_LINK_STATUS, &link_success);
      if (debug_mode_ || !link_success) {
        char info_log[512];
        glGetProgramInfoLog(shader_prog_ortho_, 512, NULL, info_log);
        std::cout << info_log << std::endl;
      }
      if (!link_success) {
        throw std::runtime_error("RTT ortho shader failed to link.");
      }
    }
    {
      rtt_shader_prog_perspective_ = glCreateProgram();
      glAttachShader(rtt_shader_prog_perspective_, rtt_vertex_shader_perspective);
      glAttachShader(rtt_shader_prog_perspective_, rtt_fragment_shader);
      glLinkProgram(rtt_shader_prog_perspective_);
      GLint link_success;
      glGetProgramiv(rtt_shader_prog_perspective_, GL_LINK_STATUS, &link_success);
      if (debug_mode_ || !link_success) {
        char info_log[512];
        glGetProgramInfoLog(shader_prog_perspective_, 512, NULL, info_log);
        std::cout << info_log << std::endl;
      }
      if (!link_success) {
        throw std::runtime_error("RTT perspective shader failed to link.");
      }
    }
    glDeleteShader(rtt_vertex_shader_perspective);
    glDeleteShader(rtt_vertex_shader_ortho);
    glDeleteShader(rtt_fragment_shader);
  }
  std::cout << "mesh_renderer initialized." << std::endl;
}

face3d::mesh_renderer::~mesh_renderer()
{
  std::lock_guard<std::mutex> guard(this->renderer_mutex_);
  this->make_context_current();
  glDeleteProgram(shader_prog_ortho_);
  glDeleteProgram(rtt_shader_prog_ortho_);
  glDeleteProgram(shader_prog_perspective_);
  glDeleteProgram(rtt_shader_prog_perspective_);
  glDeleteProgram(shader_prog_2d_);

#if FACE3D_USE_EGL
  std::cout << "Destroying EGL Context" << std::endl;
  eglDestroySurface(egl_display_, egl_surface_);
  eglDestroyContext(egl_display_, egl_context_);
  // eglTerminate(egl_display_);
#else
  if (gl_context_owned_) {
    glfwTerminate();
  }
#endif
}

bool mesh_renderer::compile_shader(GLuint shader, std::string const& shader_filename) const
{
  std::string shader_src;
  if(!io_utils::read_string(shader_filename, shader_src)) {
    std::cerr << "ERROR reading shader from " << shader_filename << std::endl;
    return false;
  }
  const char *shader_src_c = shader_src.c_str();
  glShaderSource(shader, 1, &shader_src_c, NULL);
  glCompileShader(shader);
  GLint compile_success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_success);
  if (!compile_success) {
    GLchar infoLog[512];
    glGetShaderInfoLog(shader, 512, NULL, infoLog);
    std::cerr << "ERROR: Failed to compile shader " << shader_filename << ":\n" << infoLog << std::endl;
    return false;
  }
  if (debug_mode_) {
    std::cout << "compiled " << shader_filename << std::endl;
  }
  return true;
}


bool face3d::mesh_renderer::
draw_triangles_2d(Eigen::MatrixXd const& V,
                  Eigen::MatrixXi const& F,
                  Eigen::MatrixXd const& UV)
{
  const int num_verts = V.rows();

  if (UV.rows() != num_verts) {
    std::cout << "ERROR: 2D Mesh has " << UV.rows() << " texture coordinates. Expecting " << num_verts << std::endl;
    throw std::runtime_error("mesh does not have correct number of texture coordinates");
  }
  if (V.cols() != 3) {
    std::cout << "ERROR: 2D Mesh vertices matrix has " << V.cols() << " columns, expecting 3" << std::endl;
    throw std::runtime_error("mesh does not have correct number of vertex dimensions");
  }

  std::vector<GLfloat> verts(num_verts*5);
  for (int i=0; i<num_verts; ++i) {
    int voff = i*5;
    // vertex positions
    verts[voff+0] = V(i,0);
    verts[voff+1] = V(i,1);
    verts[voff+2] = V(i,2);
    // texture coordinates
    verts[voff+3] = UV(i,0);
    verts[voff+4] = UV(i,1);
  }

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

  // position (2d)
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(0);

  // texture coords (2d)
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (GLvoid*)(3*sizeof(GLfloat)));
  glEnableVertexAttribArray(1);

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


void mesh_renderer::
set_camera_uniform_variables(GLuint shader_prog, ortho_camera_parameters<double> const& cam_params)
{
  const float depth_range = 300000.0f;
  glUniform1f(glGetUniformLocation(shader_prog, "cam_scale"),
              cam_params.scale());
  float offset_z = depth_range / 2;
  glUniform3f(glGetUniformLocation(shader_prog, "cam_trans"),
              cam_params.offset().x(), cam_params.offset().y(), offset_z);
  glUniform3f(glGetUniformLocation(shader_prog, "img_dims"),
              cam_params.nx(), cam_params.ny(), depth_range);

  // convert camera rotation to 3x3 matrix of GLfloats
  vnl_quaternion<double> qd = cam_params.rotation().as_quaternion();
  vnl_quaternion<GLfloat> qf(qd[0],qd[1],qd[2],qd[3]);
  vnl_matrix_fixed<GLfloat,3,3> R = vgl_rotation_3d<GLfloat>(qf).as_matrix();
  glUniformMatrix3fv(glGetUniformLocation(shader_prog, "cam_rot"),
                     1, GL_TRUE, &R(0,0));
}

void mesh_renderer::
set_camera_uniform_variables(GLuint shader_prog, perspective_camera_parameters<double> const& cam_params)
{
  vgl_point_3d<double> cam_center(cam_params.to_camera().camera_center());
  float dist = static_cast<float>((cam_center - vgl_point_3d<double>(0,0,0)).length());
  const float depth_range[2] = {dist-250,dist+250};

  glUniform1f(glGetUniformLocation(shader_prog, "cam_focal_len"),
              cam_params.focal_len());
  glUniform2f(glGetUniformLocation(shader_prog, "cam_principal_pt"),
              cam_params.principal_point().x(), cam_params.principal_point().y());
  glUniform2f(glGetUniformLocation(shader_prog, "img_dims"),
              cam_params.nx(), cam_params.ny());
  glUniform2f(glGetUniformLocation(shader_prog, "depth_range"),
              depth_range[0], depth_range[1]);

  // convert camera rotation to 3x3 matrix of GLfloats
  vnl_quaternion<double> qd = cam_params.rotation().as_quaternion();
  vnl_quaternion<GLfloat> qf(qd[0],qd[1],qd[2],qd[3]);
  vnl_matrix_fixed<GLfloat,3,3> R = vgl_rotation_3d<GLfloat>(qf).as_matrix();
  glUniformMatrix3fv(glGetUniformLocation(shader_prog, "cam_rot"),
                     1, GL_TRUE, &R(0,0));
  // camera translation
  vgl_vector_3d<double> trans = cam_params.translation();
  glUniform3f(glGetUniformLocation(shader_prog, "cam_trans"),
              trans.x(), trans.y(), trans.z());
}

void mesh_renderer::check_gl_error(const char* filename, int line)
{
  GLenum gl_error = glGetError();
  if(gl_error != GL_NO_ERROR) {
    std::cerr << "ERROR: glGetError() returned 0x" << std::hex << gl_error << std::dec;
    std::cerr << " at " << filename << ", line " << line << std::endl;
    throw std::runtime_error("mesh_renderer detected an OpenGL error.");
  }
}

void mesh_renderer::gl_read_pixels_rgba(int nx, int ny, vil_image_view<float> &img)
{
  // need to create a new vil_image_view here to guarantee interleaved mode
  img = vil_image_view<GLfloat>(nx,ny,1,4);
  img.fill(-1.0f);
  glReadPixels(0, 0, nx, ny, GL_RGBA, GL_FLOAT, img.top_left_ptr());
  check_gl_error(__FILE__, __LINE__);
}

void mesh_renderer::gl_tex_image_2d_rgba(int nx, int ny, vil_image_view<float> const& img, bool copy_data)
{
  GLvoid const* data_ptr = nullptr;
  if (copy_data) {
    if (img.nplanes() != 4) {
      throw std::logic_error("non-4plane image passed to gl_tex_image_2d_rgba");
    }
    if ((img.nj() != ny) || (img.ni() != nx)) {
      throw std::logic_error("gl_tex_image_2d_rgba: image size does not match arguments");
    }
    data_ptr = img.top_left_ptr();
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, nx, ny, 0, GL_RGBA, GL_FLOAT, data_ptr);
}


void mesh_renderer::gl_tex_image_2d_rgb(int nx, int ny, dlib::array2d<dlib::rgb_pixel> const& img, bool copy_data)
{
  GLvoid const* data_ptr = NULL;
  if (copy_data) {
    data_ptr = &(img[0][0]);
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, nx, ny, 0, GL_RGB, GL_UNSIGNED_BYTE, data_ptr);
}

void mesh_renderer::gl_tex_image_2d_rgb(int nx, int ny, dlib::matrix<dlib::rgb_pixel> const& img, bool copy_data)
{
  GLvoid const* data_ptr = NULL;
  if (copy_data) {
    data_ptr = &(img(0,0));
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, nx, ny, 0, GL_RGB, GL_UNSIGNED_BYTE, data_ptr);
}

void mesh_renderer::gl_tex_image_2d_rgb(int nx, int ny, vil_image_view<float> const& img, bool copy_data)
{
  GLvoid const* data_ptr = NULL;
  if (copy_data) {
    if (img.nplanes() != 3) {
      std::cerr << "ERROR: gl_tex_image_2d_rgb(): img.nplanes() = " << img.nplanes() << ". Expecting 3." << std::endl;
      throw std::logic_error("gl_tex_image_2d_rgb input must be 3-plane image");
    }
    // make sure pixels are interleaved
    if (img.planestep() != 1) {
      std::cerr << "ERROR: gl_tex_image_2d_rgb(): img.planestep() == " << img.planestep() << ".  Expecting 1 (interleaved pixels)" << std::endl;
      throw std::logic_error("gl_tex_image_2d_rgb input must have interleaved pixels");
    }
    data_ptr = img.top_left_ptr();
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, nx, ny, 0, GL_RGB, GL_FLOAT, data_ptr);
}
